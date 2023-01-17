!pip install allennlp
!pip show torch torchtext allennlp
!echo '\n'
!ls ../input
!ls ../input/movie-review-sentiment-analysis-kernels-only
import pandas as pd
import numpy as np

from tqdm import tqdm_notebook
import allennlp.common.tqdm as tqdm
tqdm._tqdm = tqdm_notebook
train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep="\t", dtype={"phrase": np.str})
test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep="\t", dtype={"phrase": np.str})
len(train)
train.where(train['Phrase'] == " ").dropna()
train.drop(2005, inplace=True)
train_len = train.shape[0]
print("len of train =", train_len)
print('Average count of phrases per sentence in train is {0:.0f}.'.format(train.groupby('SentenceId')['Phrase'].count().mean()))
print('Average count of phrases per sentence in test is {0:.0f}.'.format(test.groupby('SentenceId')['Phrase'].count().mean()))
print('Number of phrases in train: {}. Number of sentences in train: {}.'.format(train.shape[0], len(train.SentenceId.unique())))
print('Number of phrases in test: {}. Number of sentences in test: {}.'.format(test.shape[0], len(test.SentenceId.unique())))
print('Average word length of phrases in train is {0:.0f}.'.format(np.mean(train['Phrase'].apply(lambda x: len(x.split())))))
print('Average word length of phrases in test is {0:.0f}.'.format(np.mean(test['Phrase'].apply(lambda x: len(x.split())))))
#idx_separator = 124805
idx_separator = int(len(train) * 0.8)
idxs = np.random.permutation(train.shape[0])
train_idxs = idxs[:idx_separator]
val_idxs = idxs[idx_separator:]
train_df = train.iloc[train_idxs,:]
val_df = train.iloc[val_idxs,:]

print("train_df len =", len(train_df))
print("val_df len =", len(val_df))
train_df.to_csv("../input/train.csv")
val_df.to_csv("../input/val.csv")
!ls ../input
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, ELMoTokenCharactersIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer, CharacterTokenizer
from allennlp.data.fields import *
from allennlp.data import Vocabulary
class MRDatasetReader(DatasetReader):
    def __init__(self,
             tokenizer: Tokenizer = None,
             token_indexers: Dict[str, TokenIndexer] = None,
             lazy: bool = False) -> None:
        super().__init__(lazy)

        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
    
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            df = pd.read_csv(file_path, dtype={"PhraseId": np.int, "Phrase": np.str, "Sentiment": np.int8, "SentenceId": np.int})
            #print("count = ", len(train.where(train['Sentiment'] == 0).dropna()))
            for i, item in df.iterrows():
                phrase_id = item["PhraseId"]
                sentence_id = item["SentenceId"]
                phrase = item["Phrase"]
                sentiment = item["Sentiment"]
                yield self.text_to_instance(phrase_id, sentence_id, phrase, sentiment)
            
    def text_to_instance(self, phrase_id, sentence_id, phrase, sentiment) -> Instance:
        tokenized_phrase = self._tokenizer.tokenize(phrase)
        
        phrase_field = TextField(tokenized_phrase, self._token_indexers)
        #phrase_id_field = MetadataField(phrase_id)
        #sentence_id_field = MetadataField(sentence_id)
        fields = {
            "phrase": phrase_field
        }
        
        #print(f"sentiment = {sentiment} | sentiment-1 = {sentiment-1}")
        fields["labels"] = LabelField(str(int(sentiment)))
        #fields["labels"] = LabelField(sentiment, skip_indexing=True)
        
        return Instance(fields)
glove_pretrained_embedding = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz"
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder, ElmoTokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules import Highway
from allennlp.modules.elmo import Elmo
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import *
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy
from allennlp.data.iterators import *
from allennlp.training.trainer import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from allennlp.training.learning_rate_schedulers import LearningRateWithMetricsWrapper
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
class MRModel(Model):
    def __init__(self, vocab, text_field_embedder, input_size, hidden_size, dropout = 0.0):
        super().__init__(vocab)
        
        self.text_field_embedder = text_field_embedder
        
        lstm1 = nn.LSTM(input_size=input_size, hidden_size=HIDDEN_DIM, num_layers=1, bidirectional=True, batch_first=True)
        self.lstm1 = PytorchSeq2VecWrapper(lstm1)
        
        #self.lin1 = nn.Linear(in_features=self.lstm1.get_output_dim(), out_features=50)        
        self.lin1 = nn.Linear(in_features=self.lstm1.get_output_dim(), out_features=5)
        
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = CategoricalAccuracy()
        
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x
        
    def forward(self, 
                phrase: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None):
        mask = get_text_field_mask(phrase)
        embedded_phrase = self.text_field_embedder(phrase)
        encoded_phrase = self.lstm1(embedded_phrase, mask)
    
        x = self.dropout(F.relu(self.lin1(encoded_phrase)))
        
        tag_logits = F.softmax(x, dim=1)

        output = {"tag_logits": tag_logits}

        if labels is not None:
            self.accuracy(tag_logits, labels)
            output["loss"] = self.loss(tag_logits, labels)

        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
token_indexers = { 
    "tokens": SingleIdTokenIndexer(lowercase_tokens=True),
    "elmo": ELMoTokenCharactersIndexer(namespace="elmo"),
    "token_characters": TokenCharactersIndexer(character_tokenizer=CharacterTokenizer(byte_encoding="utf-8", start_tokens=[259], end_tokens=[260]))
}

reader = MRDatasetReader(token_indexers=token_indexers)

train_dataset = reader.read("../input/train.csv")
val_dataset = reader.read("../input/val.csv")
vocab = Vocabulary.from_instances(train_dataset + val_dataset)
tokens_token_embedder_weight = _read_pretrained_embeddings_file(file_uri=glove_pretrained_embedding, embedding_dim=100, vocab=vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
tokens_token_embedder = Embedding(embedding_dim=100, trainable=False, weight=tokens_token_embedder_weight, num_embeddings=vocab.get_vocab_size('tokens'))
elmo_token_embedder = ElmoTokenEmbedder(options_file=options_file, weight_file=weight_file, do_layer_norm=False)
token_characters = TokenCharactersEncoder(embedding=Embedding(embedding_dim=EMBEDDING_DIM, num_embeddings=vocab.get_vocab_size('tokens')), 
                                          encoder=CnnEncoder(embedding_dim=EMBEDDING_DIM, num_filters=50, ngram_filter_sizes=[4,5]), 
                                          dropout=0.2)

token_embedders_config = {
    "tokens": tokens_token_embedder,
    "elmo": elmo_token_embedder,
    "token_characters": token_characters
}

tf_embedder = BasicTextFieldEmbedder(token_embedders=token_embedders_config)
model = MRModel(vocab, tf_embedder, input_size=1224, hidden_size=HIDDEN_DIM, dropout=0.2)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)

iterator = BucketIterator(batch_size=256, sorting_keys=[("phrase", "num_tokens")])
iterator.index_with(vocab)

learning_rate_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=val_dataset,
                  patience=3,
                  shuffle=True,
                  num_epochs=5,
                  cuda_device=0,
                  learning_rate_scheduler=LearningRateWithMetricsWrapper(learning_rate_scheduler))
trainer.train()