# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install ../input/sacremoses/sacremoses-master
!pip install ../input/transformers/transformers-master
from tqdm import tqdm

import torch 

import numpy as np



import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from transformers import BertTokenizer, BertConfig



from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import warnings

warnings.filterwarnings('ignore')
TRAIN_FILE_NAME = "../input/tensorflow2-question-answering/simplified-nq-train.jsonl"

TEST_FILE_NAME = "../input/tensorflow2-question-answering/simplified-nq-test.jsonl"



VOCAB_SIZE = 30522

TEXT_MAX_SEQUENCE_LENGTH = 500

QUESTION_MAX_SEQUENCE_LENGTH = 25



TRAIN_BATCH_SIZE = 256

VAL_BATCH_SIZE = 64



DATALOADER_NUM_WORKERS = 8



TOKENIZER = BertTokenizer.from_pretrained("../input/bert-config/vocab.txt")

CONFIG = BertConfig.from_pretrained("../input/bert-config/bert_config.json")

# print(CONFIG)



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
import re

import random

import pandas as pd

import json

from math import floor

from tqdm import tqdm

import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertConfig



import multiprocessing

from functools import partial



import torch

from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score





def print_number_of_trainable_parameters(model):

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    params = sum([np.prod(p.size()) for p in model_parameters])

    print("Number of trainable parameters in the model are : {}".format(params))

    return



def get_results_dict(y_test, y_pred):

    results = {

        "f1": f1_score(y_test, y_pred, average="macro"),

        "precision": precision_score(y_test, y_pred, average="macro"),

        "recall": recall_score(y_test, y_pred, average="macro"),

        "accuracy": accuracy_score(y_test, y_pred)

    }

    return results

    

class InputExample(object):

    """A single training/test example for question answering."""

    

    def __init__(self, text, question, label, example_id, document_url):

        self.text = text

        self.question = question

        self.label = label

        self.example_id = example_id

        self.document_url = document_url





def cleanhtml(raw_html):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, '', raw_html)

    cleantext = re.sub("\s+", " ", cleantext)

    return cleantext





def get_generator(file_name, read_batch_size):

    with open(file_name, "r") as file_:

        batch = []

        

        for idx, item in enumerate(file_):

            if len(batch) == read_batch_size:

                batch = []

            

            batch.append(json.loads(item))

            

            if len(batch) == read_batch_size:

                yield batch

                



def create_input_examples_from_rec(rec, negative_sampling_percent=0.5):

    text = rec["document_text"].split()

    question_text = rec["question_text"].strip()

    question_text = cleanhtml(" ".join(question_text)).strip().lower()

    document_url = rec["document_url"]

    example_id = rec["example_id"]



    long_answer_start_token = rec["annotations"][0]["long_answer"]["start_token"]

    long_answer_end_token = rec["annotations"][0]["long_answer"]["end_token"]

    long_answer_candidate_idx = rec["annotations"][0]["long_answer"]['candidate_index']



    long_answer_candidates = rec["long_answer_candidates"]



    temp_input_examples_list = []



    # removing true label 

    if long_answer_start_token != -1:

        long_answer_candidates = long_answer_candidates[: long_answer_candidate_idx] \

        + long_answer_candidates[long_answer_candidate_idx + 1 :]



        # adding true label

        temp_text = text[long_answer_start_token:long_answer_end_token]

        temp_text = cleanhtml(" ".join(temp_text)).strip().lower()

        

        if len(temp_text.split()) > 0 and len(question_text.split()) > 0:

            temp_input_examples_list.append(

                InputExample(

                    text=temp_text, 

                    question=question_text,

                    label=1,

                    example_id=example_id,

                    document_url=document_url

                )

            )



    num_negative_samples = floor(len(long_answer_candidates) * negative_sampling_percent)

    sampled_negative_samples = random.sample(long_answer_candidates, num_negative_samples)



    # adding negative samples

    for candidate in sampled_negative_samples:

        candidate_start = candidate["start_token"]  

        candidate_end = candidate["end_token"]



        temp_text = text[candidate_start:candidate_end]

        temp_text = cleanhtml(" ".join(temp_text)).strip().lower()

        if len(temp_text.split()) > 0 and len(question_text.split()) > 0:

            temp_input_examples_list.append(

                InputExample(

                    text=temp_text, 

                    question=question_text,

                    label=0,

                    example_id=example_id,

                    document_url=document_url

                )

            )

    return temp_input_examples_list





class InputFeature(object):

    def __init__(

        self, text_input_ids, ques_input_ids, 

        text_seq_length, ques_seq_length,label

    ):

        self.text_input_ids = text_input_ids

        self.ques_input_ids = ques_input_ids

        self.text_seq_length = text_seq_length

        self.ques_seq_length = ques_seq_length

        self.label = label





def convert_one_example_to_feature(

    example, tokenizer,

    text_max_seq_length=None,

    ques_max_seq_length=None

):

    # featurinzing text

    text_input_words = tokenizer.tokenize(example.text)

    text_input_ids = tokenizer.convert_tokens_to_ids(text_input_words)

    text_seq_length = len(text_input_words)

    

    if text_max_seq_length:

        if text_seq_length > text_max_seq_length:

            text_input_ids = text_input_ids[:text_max_seq_length]

            text_seq_length = text_max_seq_length

        else:

            text_input_ids = text_input_ids + [tokenizer.pad_token_id]*(

                text_max_seq_length - text_seq_length

            )

    

    # featurizing question

    ques_input_words = tokenizer.tokenize(example.question)

    ques_input_ids = tokenizer.convert_tokens_to_ids(ques_input_words)

    ques_seq_length = len(ques_input_words)

    

    if ques_max_seq_length:

        if ques_seq_length > ques_max_seq_length:

            ques_input_ids = ques_input_ids[:ques_max_seq_length]

            ques_seq_length = ques_max_seq_length

        else:

            ques_input_ids = ques_input_ids + [tokenizer.pad_token_id]*(

                ques_max_seq_length - ques_seq_length

            )

    

    feature = InputFeature(

        text_input_ids=text_input_ids,

        ques_input_ids=ques_input_ids,

        label = example.label,

        text_seq_length=text_seq_length,

        ques_seq_length=ques_seq_length

    )

    return feature

    



def load_cache_examples_multiprocessing(

    examples, tokenizer, text_max_seq_length, ques_max_seq_length

):

    pool = multiprocessing.Pool()

    features = pool.map(

        partial(

            convert_one_example_to_feature,

            tokenizer=tokenizer,

            text_max_seq_length=text_max_seq_length,

            ques_max_seq_length=ques_max_seq_length

        ),

        examples

    )

    pool.close()

    

    # convert to Tensors and build dataset

    all_text_input_ids = torch.tensor(

        [f.text_input_ids for f in features], dtype=torch.long

    )

    all_ques_input_ids = torch.tensor(

        [f.ques_input_ids for f in features], dtype=torch.long

    )

    all_text_seq_lengths = torch.tensor(

        [f.text_seq_length for f in features], dtype=torch.long

    )

    all_ques_seq_lengths = torch.tensor(

        [f.ques_seq_length for f in features], dtype=torch.long

    )

    all_labels = torch.tensor(

        [f.label for f in features], dtype=torch.float

    )

    dataset = TensorDataset(

        all_text_input_ids, all_ques_input_ids, all_text_seq_lengths,

        all_ques_seq_lengths, all_labels

    )

    return dataset
import torch 

import torch.optim as optim

import torch.nn as nn

import torch.nn.functional as F





# takes the last hidden_state as the question encoding



class BiLstmEncoder(nn.Module):

    def __init__(

        self, vocab_size, embedding_dim,

        hidden_size, num_layers, output_dim,

        bidirectional, dropout_ratio=0.2

    ):

        super(BiLstmEncoder, self).__init__()

        self.vocab_size = vocab_size

        self.embedding_dim = embedding_dim

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.bidirectional = bidirectional

        self.dropout_ratio = dropout_ratio

        self.output_dim = output_dim



        self.embedding = nn.Embedding(

            num_embeddings=vocab_size,

            embedding_dim= embedding_dim

        )

        self.lstm_layer = nn.LSTM(

            input_size=embedding_dim,

            num_layers=num_layers,

            hidden_size=hidden_size,

            bidirectional=bidirectional,

            dropout=dropout_ratio,

            batch_first=True

        )



        if bidirectional is True: 

            self.linear_combiner = nn.Linear(num_layers*2*hidden_size, output_dim)

        else:

            self.linear_combiner = nn.Linear(num_layers*1*hidden_size, output_dim)



        self.dropout_layer = nn.Dropout(dropout_ratio)

    

    def forward(self, question, ques_seq_lengths=None):

        # question = [batch_size, sent_length]



        batch_size = question.shape[0]

    

        embedded = self.embedding(question)

        # embedded = [batch_size, sent_length, embedding_dim]



        _, (hidden, _) = self.lstm_layer(embedded)

        # hidden = [num_layers * num_directions, batch_size, hidden_size]



        hidden = hidden.view(batch_size, -1)

        # hidden = [batch_size, num_layers * num_directions * hidden_size]



        combined_context = self.linear_combiner(self.dropout_layer(hidden))

        # combined_context = [batch_size, output_dim]



        return combined_context





class Attention(nn.Module):

    """

    Computes a weighted average of channels across timesteps (1 parameter pr. channel).

    """

    def __init__(

        self, attention_size, device

    ):

        super(Attention, self).__init__()

        self.attention_size = attention_size

        self.device = device



        self.attention = nn.Parameter(torch.rand(attention_size))



    def forward(self, inputs, input_lengths):

        # inputs = [batch_size, max_seq_length, attention_size]

        # input_lengths = [batch_size]



        max_seq_length = inputs.shape[1]



        attn = torch.matmul(inputs, self.attention)

        # attn = [batch_size, maz_seq_len]



        idxes = torch.arange(0, max_seq_length, out=torch.LongTensor(max_seq_length)).unsqueeze(0).to(self.device)

        mask = torch.autograd.Variable((idxes < input_lengths.unsqueeze(1)).float()).to(self.device)

        # mask = [batch_size, max_seq_length]

        # idxes = [batch_size, max_seq_length]



        attn_masked = attn.masked_fill(mask == 0, -1e10)

        attention_weights = F.softmax(attn_masked, dim=1)

        # attention_weights = [batch_size, max_seq_length]



        # apply attention weights

        weighted = torch.bmm(attention_weights.unsqueeze(1), inputs)

        # weighted = [batch_size, 1, attention_size]



        weighted = weighted.squeeze(1)

        # weighted_outputs = [batch_size, attention_size]



        return (weighted, attention_weights)





class AttentiveBilstm(nn.Module):

    def __init__(

        self, max_seq_length,

        vocab_size, embedding_dim,

        hidden_size, num_layers, output_dim,

        bidirectional, device, dropout_ratio=0.2

    ):

        super(AttentiveBilstm, self).__init__()

        self.max_seq_length = max_seq_length

        self.vocab_size = vocab_size

        self.embedding_dim = embedding_dim

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.output_dim = output_dim

        self.bidirectional = bidirectional

        self.device = device

        self.dropout_ratio = dropout_ratio



        self.embedding = nn.Embedding(

            num_embeddings=vocab_size,

            embedding_dim=embedding_dim

        )



        self.lstm_layer = nn.LSTM(

            input_size=embedding_dim,

            num_layers=num_layers,

            hidden_size=hidden_size,

            bidirectional=bidirectional,

            dropout=dropout_ratio

        )

        

        if bidirectional is True:

            self.attention_layer = Attention(

                attention_size=hidden_size*2,

                device=device

            )

            self.linear_combiner = nn.Linear(

                hidden_size*2, output_dim

            )

        else:

            self.attention_layer = Attention(

                attention_size=hidden_size*1,

                device=device

            )

            self.linear_combiner = nn.Linear(

                hidden_size, output_dim

            ) 



        self.dropout_layer = nn.Dropout(dropout_ratio)

    

    def forward(self, question, seq_lengths):

        # question = [batch_size, max_seq_length]

        # seq_lengths = [batch_size]



        embedded = self.dropout_layer(self.embedding(question))

        # embedded = [batch_size, max_seq_length, embedding_dim]

        print(embedded.shape)



        # permuting for pad packed easiness

        embedded = embedded.permute(1, 0, 2)

        # embedded = [max_seq_length, batch_size, embedding_dim]



        packed_embedded = nn.utils.rnn.pack_padded_sequence(

            embedded, seq_lengths, enforce_sorted=False

        )

    

        packed_outputs, (_, _) = self.lstm_layer(packed_embedded)



        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, total_length=self.max_seq_length)

        # outputs = [max_seq_length, batch_size, num_directions*hidden_size]



        # outputs are permuted again because attention layer needs batch_first

        (weighted_outputs, attention_weights) = self.attention_layer(outputs.permute(1, 0, 2), seq_lengths)

        # weighted_outputs = [batch_size, attention_size]

        

        weighted_outputs = self.linear_combiner(weighted_outputs)

        # weighted_outputs = [batch_size, output_dim]



        return (weighted_outputs, attention_weights)





class DeepMoji(nn.Module):

    def __init__(

        self, vocab_size, embedding_dim, hidden_state_size,

        num_layers, output_dim, device,

        dropout_ratio=0.5, bidirectional=True

    ):



        super(DeepMoji, self).__init__()

        self.vocab_size = vocab_size

        self.embedding_dim = embedding_dim

        self.hidden_state_size = hidden_state_size

        self.num_layers=num_layers

        self.output_dim = output_dim

        self.dropout_ratio = dropout_ratio

        self.bidirectional = bidirectional

        self.output_dim = output_dim

        self.device = device

        

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        

        self.bilstm_one = nn.LSTM(

            input_size=embedding_dim,

            hidden_size=hidden_state_size,

            bidirectional=bidirectional,

            num_layers=num_layers,

            dropout=dropout_ratio

        )

        

        self.lstm_one_context_combiner_layer = nn.Linear(

            2*hidden_state_size, hidden_state_size

        )

        

        self.bilstm_two = nn.LSTM(

            input_size=hidden_state_size,

            hidden_size=hidden_state_size,

            bidirectional=bidirectional,

            num_layers=num_layers,

            dropout=dropout_ratio

        )

        self.lstm_two_context_combiner_layer = nn.Linear(

            2*hidden_state_size, hidden_state_size

        )

        

        self.attn_layer = Attention(hidden_state_size*2 + embedding_dim, device)



        self.output_layer = nn.Linear(hidden_state_size*2 + embedding_dim, output_dim)

        

        self.dropout_layer = nn.Dropout(dropout_ratio)

    

    def forward(self, inp, src_len):

        # inp = [batch_size, sent_length]

        # src_len = [batch_size]

        

        embedded = self.dropout_layer(self.embedding(inp)).permute(1, 0, 2)

        # embedded = [sent_length, batch_size, embedding_dim]

        

        embedded_packed = torch.nn.utils.rnn.pack_padded_sequence(

            embedded, src_len, enforce_sorted=False

        )

        

        bilstm_out_1_packed, (_, _) = self.bilstm_one(embedded_packed)

        # bilstm_out_1 = [seq_len, batch_size, 2 * hidden_state_size] 

        

        bilstm_out_1, _ = nn.utils.rnn.pad_packed_sequence(

            bilstm_out_1_packed, total_length=embedded.shape[0])



        bilstm_out_1_context_combined = self.lstm_one_context_combiner_layer(

            bilstm_out_1

        )

        # bilstm_out_1_context_combined = [seq_len, batch_size, hidden_state_size]



        bilstm_2_input_packed = torch.nn.utils.rnn.pack_padded_sequence(

            bilstm_out_1_context_combined, src_len, enforce_sorted=False

        )

        bilstm_out_2_packed, (_, _) = self.bilstm_two(bilstm_2_input_packed) 

        # bilstm_out_2_packed = [seq_len, batch_size, 2 * hidden_state_size]

        

        bilstm_out_2, _ = nn.utils.rnn.pad_packed_sequence(

            bilstm_out_2_packed, total_length=embedded.shape[0]

        )

        

        bilstm_out_2_context_combined = self.lstm_two_context_combiner_layer(

            bilstm_out_2

        )

        # bilstm_out_2_context_combined = [seq_len, batch_size, hidden_state_size]

    

        bilstm_stacked = torch.cat(

            (

                bilstm_out_1_context_combined,

                bilstm_out_2_context_combined

            ), 

            dim=2

        )

        # bilstm_stacked = [seq_len, batch_size, 2 * hidden_state_size]



        # stacking embedded to bilstm_stacked

        bilstm_and_embedded_stacked = torch.cat(

            (

                bilstm_stacked,

                embedded

            ), 

            dim=2

        )    

        # bilstm_and_embedded_stacked = [seq_len, batch_size, 2 * hidden_state_size + embedding_dim]

        

                

        (weighted_outputs, attention_weights) = self.attn_layer(

            bilstm_and_embedded_stacked.permute(1, 0, 2), src_len

        )

        # weighted_outputs = [batch_size, attention_size]



        outputs = self.output_layer(weighted_outputs)

        # outputs = [batch_size, output_dim]



        return outputs





class QAModel(nn.Module):

    def __init__(

        self, ques_embedding_dim,

        text_embedding_dim,

        ques_embedder,

        text_embedder, device,

        first_combiner_size=100,

        dropout_ratio=0.2

    ):

        super(QAModel, self).__init__()



        self.text_embedding_dim = text_embedding_dim

        self.ques_embedding_dim = ques_embedding_dim

        self.text_embedder = text_embedder

        self.ques_embedder = ques_embedder

        self.device = device



        self.linear_combiner_1 = nn.Linear(

            text_embedding_dim + ques_embedding_dim, first_combiner_size

        )

        self.linear_combiner_2 = nn.Linear(

            first_combiner_size, 1

        )

        self.dropout_layer = nn.Dropout(dropout_ratio)

    

    def forward(

        self, text_input_ids, ques_input_ids,

        text_seq_lengths, ques_seq_lengths

    ):

        # text_input_ids = [batch_size, text_max_seq_length]

        # ques_input_ids = [batch_size, ques_max_seq_length]

        # text_seq_lengths = [batch_size]

        # quest_seq_lengths = [batch_size]

        # labels = [batch_size]

        

        text_encoded = self.text_embedder(text_input_ids, text_seq_lengths)

        # text_encoded = [batch_size, text_embedding_dim]

        

        ques_encoded = self.ques_embedder(ques_input_ids, ques_seq_lengths)

        # ques_encoded = [batch_size, ques_embedding_dim]

        

        # stacking both

        stacked_ques_text = torch.cat(

            (

                ques_encoded,

                text_encoded

            ), 

            dim=1

        )

        # stacked_ques_text = [batch_size, ques_embedding_dim + text_embedding_dim]

        

        fc_out_1 = self.linear_combiner_1(self.dropout_layer(stacked_ques_text))

        # fc_out_1 = [batch_size, first_combiner_size]

        

        output = self.linear_combiner_2(fc_out_1)

        # output = [batch_size, 1]

        

        return output
ques_encoder = BiLstmEncoder(

    vocab_size=VOCAB_SIZE, embedding_dim=50,

    hidden_size=64, num_layers=1, output_dim=100,

    bidirectional=True

)



text_encoder = DeepMoji(

    vocab_size=VOCAB_SIZE, embedding_dim=300, hidden_state_size=256,

    num_layers=2, output_dim=500,

    dropout_ratio=0.5, bidirectional=True, device=DEVICE

)



model = QAModel(

    ques_embedding_dim=100,

    text_embedding_dim=500,

    ques_embedder=ques_encoder,

    text_embedder=text_encoder,

    first_combiner_size=200,

    dropout_ratio=0.2, device=DEVICE

)
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters())

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
model = model.to(DEVICE)

criterion = criterion.to(DEVICE)
def train_epoch_from_generator(

    model, train_generator, optimizer, criterion,

    negative_sampling_percent, tokenizer,

    scheduler, print_stats_at_step=20,

    train_batch_size=TRAIN_BATCH_SIZE,

    generator_num_workers=DATALOADER_NUM_WORKERS,

    text_max_seq_length=TEXT_MAX_SEQUENCE_LENGTH,

    ques_max_seq_length=QUESTION_MAX_SEQUENCE_LENGTH,

    device=DEVICE

):

    tr_loss = 0.0

    avg_tr_loss = 0.0

    

    preds = None

    out_label_ids = None

    

    model.train()

    step = 0

    for generator_idx, generator_batch in enumerate(train_generator):

        all_data = []



        for rec in generator_batch:

            all_data.extend(create_input_examples_from_rec(rec, negative_sampling_percent=0.2))



        generator_train_dataset = load_cache_examples_multiprocessing(

            all_data, tokenizer, text_max_seq_length=text_max_seq_length,

            ques_max_seq_length=ques_max_seq_length

        )



        generator_random_sampler = RandomSampler(generator_train_dataset)

        generator_data_loader = DataLoader(

            generator_train_dataset, sampler=generator_random_sampler,

            batch_size=train_batch_size,

            num_workers=generator_num_workers

        )

        

        generator_batch_iterator = tqdm(generator_data_loader)

        for batch_idx, batch_data in enumerate(generator_batch_iterator):

            if batch_data[0].shape[0] == train_batch_size:

                model.zero_grad()

                

                batch_data = tuple(t.to(device) for t in batch_data)

                inputs = {

                    "text_input_ids": batch_data[0],

                    "ques_input_ids": batch_data[1],

                    "text_seq_lengths": batch_data[2], 

                    "ques_seq_lengths": batch_data[3]

                }

                true_labels = batch_data[4]

                

                # getting outputs 

                logits = model(**inputs).squeeze(1)

                

                # propagating loss backwards and scheduler and optimizer steps

                loss = criterion(logits, true_labels)

                loss.backward()

                optimizer.step()

                scheduler.step()

                step_loss = loss.item()

                

                tr_loss += step_loss

                avg_tr_loss += step_loss

                

                # for calculation of results matrix

                if preds is None:

                    preds = torch.round(F.sigmoid(logits)).detach().cpu().numpy()

                    out_label_ids = true_labels.detach().cpu().numpy()

                else:

                    preds = np.append(

                        preds,

                        torch.round(F.sigmoid(logits)).detach().cpu().numpy(),

                        axis=0

                    )

                    out_label_ids = np.append(

                        out_label_ids,

                        true_labels.detach().cpu().numpy(),

                        axis=0

                    )

                if step % print_stats_at_step == 0:

                    tr_loss = tr_loss / print_stats_at_step

                    results = get_results_dict(out_label_ids, preds)

                    # writing on bar

                    generator_batch_iterator.set_description(

                        f'Tr Iter: {step}, avg_step_loss: {tr_loss:.4f}, avg_tr_loss: {(avg_tr_loss / (step + 1)):.4f}, tr_f1: {results["f1"]:.4f}, tr_prec: {results["precision"]:.4f}, tr_rec: {results["recall"]:.4f}, tr_acc: {results["accuracy"]:.4f}'

                    )

                    tr_loss = 0.0

                    preds = None

                    out_label_ids = None

                step += 1

                

        print(f"{generator_idx + 1} generator is completed.")

        if generator_idx > 1:

            break

    return
data_generator = get_generator(TRAIN_FILE_NAME, 100)
train_epoch_from_generator(

    model, data_generator, optimizer=optimizer, criterion=criterion,

    negative_sampling_percent=0.1, tokenizer=TOKENIZER,

    scheduler=scheduler, print_stats_at_step=20,

    train_batch_size=TRAIN_BATCH_SIZE,

    generator_num_workers=DATALOADER_NUM_WORKERS,

    text_max_seq_length=TEXT_MAX_SEQUENCE_LENGTH,

    ques_max_seq_length=QUESTION_MAX_SEQUENCE_LENGTH,

    device=DEVICE

)