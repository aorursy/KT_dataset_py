# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import shutil



from allennlp.common.params import Params

from allennlp.common.util import prepare_environment, dump_metrics

from allennlp.data.iterators import BasicIterator, BucketIterator

from allennlp.data.token_indexers import SingleIdTokenIndexer

from allennlp.data.token_indexers import PretrainedBertIndexer

from allennlp.data.tokenizers import WordTokenizer

from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter

from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder

from allennlp.modules.token_embedders import Embedding

from allennlp.training import Trainer

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch

from pytorch_pretrained_bert.optimization import BertAdam

from sklearn.model_selection import train_test_split

import torch

import torch.optim as optim



# from utility script

from swemencoder import SWEMEncoder

from toxiccommentclassificationreader import ToxicCommentClassificationReader

from toxiccommentpredictor import ToxicCommentPredictor

from toxicbaseclassifier import ToxicBaseClassifier

from toxicbertclassifier import ToxicBertClassifier
# from https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py

import math

import torch

from torch.optim.optimizer import Optimizer, required



class RAdam(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.buffer = [[None, None, None] for ind in range(10)]

        super(RAdam, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(RAdam, self).__setstate__(state)



    def step(self, closure=None):



        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:



            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data.float()

                if grad.is_sparse:

                    raise RuntimeError('RAdam does not support sparse gradients')



                p_data_fp32 = p.data.float()



                state = self.state[p]



                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p_data_fp32)

                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                else:

                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)

                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)



                state['step'] += 1

                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:

                    N_sma, step_size = buffered[1], buffered[2]

                else:

                    buffered[0] = state['step']

                    beta2_t = beta2 ** state['step']

                    N_sma_max = 2 / (1 - beta2) - 1

                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                    buffered[1] = N_sma



                    # more conservative since it's an approximated value

                    if N_sma >= 5:

                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    else:

                        step_size = group['lr'] / (1 - beta1 ** state['step'])

                    buffered[2] = step_size



                if group['weight_decay'] != 0:

                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)



                # more conservative since it's an approximated value

                if N_sma >= 5:            

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                else:

                    p_data_fp32.add_(-step_size, exp_avg)



                p.data.copy_(p_data_fp32)



        return loss
# from https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py

import itertools as it

from torch.optim import Optimizer



class Lookahead(Optimizer):

    def __init__(self, base_optimizer,alpha=0.5, k=6):

        if not 0.0 <= alpha <= 1.0:

            raise ValueError(f'Invalid slow update rate: {alpha}')

        if not 1 <= k:

            raise ValueError(f'Invalid lookahead steps: {k}')

        self.optimizer = base_optimizer

        self.param_groups = self.optimizer.param_groups

        self.alpha = alpha

        self.k = k

        for group in self.param_groups:

            group["step_counter"] = 0

        self.slow_weights = [[p.clone().detach() for p in group['params']]

                                for group in self.param_groups]



        for w in it.chain(*self.slow_weights):

            w.requires_grad = False

            

        self.state = base_optimizer.state



    def step(self, closure=None):

        loss = None

        if closure is not None:

            loss = closure()

        loss = self.optimizer.step()

        for group,slow_weights in zip(self.param_groups,self.slow_weights):

            group['step_counter'] += 1

            if group['step_counter'] % self.k != 0:

                continue

            for p,q in zip(group['params'],slow_weights):

                if p.grad is None:

                    continue

                q.data.add_(self.alpha,p.data - q.data)

                p.data.copy_(q.data)

                self.state = self.optimizer.state

        return loss
params = Params({})

prepare_environment(params)
reader = ToxicCommentClassificationReader(token_indexers={

    "tokens1": SingleIdTokenIndexer(),

    "tokens2": SingleIdTokenIndexer(),

})

all_dataset = reader.read('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

train_dataset, validation_dataset = train_test_split(all_dataset, test_size=0.2, random_state=11)

vocab = Vocabulary.from_instances(train_dataset, min_count={'tokens': 3})

iterator = BucketIterator(

    batch_size=512,

    sorting_keys=[("tokens", "num_tokens")],

)

iterator.index_with(vocab)
glove_params = Params({

    'pretrained_file': '../input/glove-stanford/glove.twitter.27B.200d.txt',

    'embedding_dim': 200,

    'trainable': False

})

fasttext_params = Params({

    'pretrained_file': '../input/fatsttext-common-crawl/crawl-300d-2M/crawl-300d-2M.vec',

    'embedding_dim': 300,

    'trainable': False

})

glove_embedding = Embedding.from_params(vocab, glove_params)

fasttext_embedding = Embedding.from_params(vocab, fasttext_params)

word_embeddings = BasicTextFieldEmbedder({"tokens1": glove_embedding, "tokens2": fasttext_embedding})

seq2vec_encoder = BagOfEmbeddingsEncoder(embedding_dim=word_embeddings.get_output_dim())
model = ToxicBaseClassifier(

    text_field_embedder=word_embeddings,

    seq2seq_encoder=None,

    seq2vec_encoder=seq2vec_encoder,

    dropout=0.5,

    num_labels=6,

    vocab=vocab

)

model.cuda()



trainer = Trainer(

    model=model,

    optimizer=Lookahead(RAdam(model.parameters())),

    iterator=iterator,

    train_dataset=train_dataset,

    validation_dataset=validation_dataset,

    cuda_device=0,

    num_epochs=1000,

    grad_norm=5.0,

    grad_clipping=1.0,

    patience=10

)

metrics = trainer.train()
print('metrics: {}'.format(metrics))

print('best_validation_loss: {}'.format(metrics['best_validation_loss']))
seq2vec_encoder = SWEMEncoder(embedding_dim=word_embeddings.get_output_dim())
model = ToxicBaseClassifier(

    text_field_embedder=word_embeddings,

    seq2seq_encoder=None,

    seq2vec_encoder=seq2vec_encoder,

    dropout=0.5,

    num_labels=6,

    vocab=vocab

)

model.cuda()



trainer = Trainer(

    model=model,

    optimizer=Lookahead(RAdam(model.parameters())),

    iterator=iterator,

    train_dataset=train_dataset,

    validation_dataset=validation_dataset,

    cuda_device=0,

    num_epochs=1000,

    grad_norm=5.0,

    grad_clipping=1.0,

    patience=10

)

metrics = trainer.train()
print('metrics: {}'.format(metrics))

print('best_validation_loss: {}'.format(metrics['best_validation_loss']))
lstm = torch.nn.LSTM(

    bidirectional=True,

    input_size=word_embeddings.get_output_dim(),

    hidden_size=40,

    num_layers=2,

    batch_first=True

)

seq2seq_encoder = PytorchSeq2SeqWrapper(lstm)

seq2vec_encoder = SWEMEncoder(embedding_dim=seq2seq_encoder.get_output_dim())
model = ToxicBaseClassifier(

    text_field_embedder=word_embeddings,

    seq2seq_encoder=seq2seq_encoder,

    seq2vec_encoder=seq2vec_encoder,

    dropout=0.5,

    num_labels=6,

    vocab=vocab

)

model.cuda()



trainer = Trainer(

    model=model,

    optimizer=Lookahead(RAdam(model.parameters())),

    iterator=iterator,

    train_dataset=train_dataset,

    validation_dataset=validation_dataset,

    cuda_device=0,

    num_epochs=1000,

    grad_norm=5.0,

    grad_clipping=1.0,

    patience=10

)

metrics = trainer.train()
print('metrics: {}'.format(metrics))

print('best_validation_loss: {}'.format(metrics['best_validation_loss']))
BERT_MODEL_PATH = '../input/bertpretrained/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'

WORK_DIR = "../working/"

convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(

    BERT_MODEL_PATH + 'bert_model.ckpt',

    BERT_MODEL_PATH + 'bert_config.json',

    WORK_DIR + 'pytorch_model.bin'

)

shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + 'bert_config.json')
token_indexer = PretrainedBertIndexer(

    pretrained_model=BERT_MODEL_PATH,

    max_pieces=128,

    do_lowercase=True,

)



tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())



reader = ToxicCommentClassificationReader(

    tokenizer=tokenizer,

    token_indexers={"bert": token_indexer}

)



all_dataset = reader.read('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

train_dataset, validation_dataset = train_test_split(all_dataset, test_size=0.2, random_state=11)



iterator = BucketIterator(

    batch_size=64,

    sorting_keys=[("tokens", "num_tokens")],

)



vocab = Vocabulary()

iterator.index_with(vocab)



model = ToxicBertClassifier(

    vocab=vocab,

    bert_model=WORK_DIR,

    num_labels=6

)
model.cuda()



trainer = Trainer(model=model,

                  optimizer=Lookahead(RAdam(model.parameters())),

                  iterator=iterator,

                  train_dataset=train_dataset,

                  validation_dataset=validation_dataset,

                  cuda_device=0,

                  num_epochs=1000,

                  grad_norm=5.0,

                  grad_clipping=1.0,

                  patience=3)

metrics = trainer.train()
print('metrics: {}'.format(metrics))

print('best_validation_loss: {}'.format(metrics['best_validation_loss']))
test_dataset = reader.read('../input/jigsaw-toxic-comment-classification-challenge/test.csv')

seq_iterator = BasicIterator(batch_size=64)

seq_iterator.index_with(vocab)
predictor = ToxicCommentPredictor(model, seq_iterator, cuda_device=0)

test_preds = predictor.predict(test_dataset)

submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = test_preds

submission.to_csv('submission.csv', index=False)