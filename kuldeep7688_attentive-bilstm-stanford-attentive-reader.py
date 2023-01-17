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

import random

import numpy as np



import torch 

import torch.optim as optim

import torch.nn as nn

import torch.nn.functional as F

from pathlib import Path



from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from transformers import BertTokenizer, BertConfig



from collections import defaultdict

from dataclasses import dataclass

import functools

import gc

import itertools

import json

from multiprocessing import Pool

import os

from pathlib import Path

import random

import re

import shutil

import subprocess

import time

from typing import Callable, Dict, List, Generator, Tuple



import numpy as np

import pandas as pd



# from pandas.io.json._json import JsonReader

from pandas.io.json._json import JsonReader

from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm



import torch

from torch import nn, optim

from torch.utils.data import Dataset, Subset, DataLoader



from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix



import warnings

warnings.filterwarnings('ignore')
def print_number_of_trainable_parameters(model):

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    params = sum([np.prod(p.size()) for p in model_parameters])

    print("Number of trainable parameters in the model are : {}".format(params))

    return





class Example(object):

    def __init__(

        self,

        example_id,

        doc_start,

        question_len,

        text_len,

        input_text_ids,

        input_question_ids,

        start_position,

        end_position,

        class_label,

        doc_position

    ):

        self.example_id = example_id

        self.doc_start = doc_start

        self.question_len = question_len

        self.text_len = text_len

        self.input_text_ids = input_text_ids

        self.input_question_ids = input_question_ids

        self.start_position = start_position

        self.end_position = end_position

        self.class_label = class_label

        self.doc_position = doc_position



        

def convert_data(

    line: str,

    tokenizer: object,

    max_seq_len: int,

    max_question_len: int,

    doc_stride: int

) -> List[Example]:

    """Convert dictionary data into list of training data.



    Parameters

    ----------

    line : str

        Training data.

    tokenizer : transformers.BertTokenizer

        Tokenizer for encoding texts into ids.

    max_seq_len : int

        Maximum input sequence length.

    max_question_len : int

        Maximum input question length.

    doc_stride : int

        When splitting up a long document into chunks, how much stride to take between chunks.

    """



    def _find_short_range(short_answers: List[Dict]) -> Tuple[int, int]:

        answers = pd.DataFrame(short_answers)

        start_min = answers['start_token'].min()

        end_max = answers['end_token'].max()

        return start_min, end_max



    # model input

    data = json.loads(line)

    doc_words = data['document_text'].split()

    question_tokens = tokenizer.tokenize(data['question_text'])

    question_len = len(question_tokens)

    if len(question_tokens) > max_question_len:

        question_tokens = question_tokens[:max_question_len]

        question_len = max_question_len

    else:

        question_tokens = question_tokens + [tokenizer.pad_token]*(max_question_len - question_len)

        



    # tokenized index of i-th original token corresponds to original_to_tokenized_index[i]

    # if a token in original text is removed, its tokenized index indicates next token

    original_to_tokenized_index = []

    tokenized_to_original_index = []

    all_doc_tokens = []  # tokenized document text

    for i, word in enumerate(doc_words):

        original_to_tokenized_index.append(len(all_doc_tokens))

        if re.match(r'<.+>', word):  # remove paragraph tag

            continue

        sub_tokens = tokenizer.tokenize(word)

        for sub_token in sub_tokens:

            tokenized_to_original_index.append(i)

            all_doc_tokens.append(sub_token)



    # model output: (class_label, start_position, end_position)

    annotations = data['annotations'][0]

    if annotations['yes_no_answer'] in ['YES', 'NO']:

        class_label = annotations['yes_no_answer'].lower()

        start_position = annotations['long_answer']['start_token']

        end_position = annotations['long_answer']['end_token']

    elif annotations['short_answers']:

        class_label = 'short'

        start_position, end_position = _find_short_range(annotations['short_answers'])

    elif annotations['long_answer']['candidate_index'] != -1:

        class_label = 'long'

        start_position = annotations['long_answer']['start_token']

        end_position = annotations['long_answer']['end_token']

    else:

        class_label = 'no_answer'

        start_position = -1

        end_position = -1



    # convert into tokenized index

    if start_position != -1 and end_position != -1:

        start_position = original_to_tokenized_index[start_position]

        end_position = original_to_tokenized_index[end_position]



    # make sure at least one object in `examples`

    examples = []

    

    # take chunks with a stride of `doc_stride`

    for doc_idx, doc_start in enumerate(range(0, len(all_doc_tokens), doc_stride)):

        doc_end = doc_start + max_seq_len

        # if truncated document does not contain annotated range

        if not (doc_start <= start_position and end_position <= doc_end):

            start, end, label = -1, -1, 'no_answer'

        else:

            start = start_position - doc_start 

            end = end_position - doc_start

            label = class_label



        assert -1 <= start <= max_seq_len, f'start position is out of range: {start}'

        assert -1 <= end <= max_seq_len, f'end position is out of range: {end}'



        text_tokens = all_doc_tokens[doc_start:doc_end]

        text_len = len(text_tokens)

        if text_len > max_seq_len:

            text_tokens = text_tokens[:max_seq_len]

            text_len = max_seq_len

        else:

            text_tokens = text_tokens + [tokenizer.pad_token]*(max_seq_len - text_len)



        examples.append(

            Example(

                example_id=data['example_id'],

                doc_start=doc_start,

                question_len=question_len,

                text_len = text_len,

                input_text_ids=tokenizer.convert_tokens_to_ids(text_tokens),

                input_question_ids=tokenizer.convert_tokens_to_ids(question_tokens),

                start_position=start,

                end_position=end,

                class_label=label,

                doc_position=doc_idx / (len(all_doc_tokens) / doc_stride)

            )

        )

    return examples





class JsonChunkReader(JsonReader):

    """JsonReader provides an interface for reading in a JSON file.

    """

    

    def __init__(

        self,

        filepath_or_buffer: str,

        convert_data: Callable[[str], List[Example]],

        orient: str = None,

        typ: str = 'frame',

        dtype: bool = None,

        convert_axes: bool = None,

        convert_dates: bool = True,

        keep_default_dates: bool = True,

        numpy: bool = False,

        precise_float: bool = False,

        date_unit: str = None,

        encoding: str = None,

        lines: bool = True,

        chunksize: int = 2000,

        compression: str = None,

    ):

        super(JsonChunkReader, self).__init__(

            str(filepath_or_buffer),

            orient=orient, typ=typ, dtype=dtype,

            convert_axes=convert_axes,

            convert_dates=convert_dates,

            keep_default_dates=keep_default_dates,

            numpy=numpy, precise_float=precise_float,

            date_unit=date_unit, encoding=encoding,

            lines=lines, chunksize=chunksize,

            compression=compression

        )

        self.convert_data = convert_data

        

    def __next__(self):

        lines = list(itertools.islice(self.data, self.chunksize))

        if lines:

            with Pool(2) as p:

                obj = p.map(self.convert_data, lines)

            return obj



        self.close()

        raise StopIteration

        



class TextDataset(Dataset):

    """Dataset for [TensorFlow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering).

    

    Parameters

    ----------

    examples : list of Example

        The whole Dataset.

    """

    

    def __init__(self, examples: List[Example]):

        self.examples = examples

        

    def __len__(self) -> int:

        return len(self.examples)

    

    def __getitem__(self, index):

        annotated = list(

            filter(lambda example: example.class_label != 'no_answer', self.examples[index]))

        if len(annotated) == 0:

            return random.choice(self.examples[index])

        return random.choice(annotated)



    

def collate_fn(examples: List[Example]) -> List[List[torch.Tensor]]:

    # input tokens

    max_len_text = max([len(example.input_text_ids) for example in examples])

    max_len_ques = max([len(example.input_question_ids) for example in examples])

    

    # lengths 

    question_lengths = np.array([example.question_len for example in examples])

    text_lengths = np.array([example.text_len for example in examples])

    doc_positions = np.array([example.doc_position for example in examples])

    

    # tokens

    text_tokens = np.zeros((len(examples), max_len_text), dtype=np.int64)

    question_tokens = np.zeros((len(examples), max_len_ques), dtype=np.int64)

    for i, example in enumerate(examples):

        # text tokens 

        row = example.input_text_ids

        text_tokens[i, :len(row)] = row

        # question tokens 

        row2 = example.input_question_ids

        question_tokens[i, :len(row2)] = row2

         

    # output labels

    start_positions = np.array([example.start_position for example in examples])

    end_positions = np.array([example.end_position for example in examples])

    start_positions = np.where(start_positions >= max_len_text, -1, start_positions)

    end_positions = np.where(end_positions >= max_len_text, -1, end_positions)

    

    all_labels = ['long', 'no', 'short', 'no_answer', 'yes']

    class_labels = [all_labels.index(example.class_label) for example in examples]



    input_and_labels = [

        torch.from_numpy(text_tokens),

        torch.from_numpy(question_tokens),

        torch.from_numpy(text_lengths),

        torch.from_numpy(question_lengths),

        torch.from_numpy(doc_positions),

        torch.LongTensor(start_positions),

        torch.LongTensor(end_positions),

        torch.LongTensor(class_labels)

    ]



    return input_and_labels



def triple_loss_function(

    preds, labels, st_crtierion,

    en_criterion, class_criterion

):

    start_preds, end_preds, class_preds = preds

    start_labels, end_labels, class_labels = labels

    

    start_loss = st_crtierion(start_preds, start_labels)

    end_loss = en_criterion(end_preds, end_labels)

    class_loss = class_criterion(class_preds, class_labels)

    return start_loss + end_loss + class_loss



def get_results_dict(preds, labels):

    start_preds, end_preds, class_preds = preds

    start_labels, end_labels, class_labels = labels

    results = {

#         "f1_start_tokens": f1_score(start_labels, start_preds),

#         "f1_end_tokens": f1_score(end_labels, end_preds),

        "f1_class_labels": f1_score(class_labels, class_preds, average='macro'),

        "acc_s_tok": accuracy_score(start_labels, start_preds),

        "acc_e_tok": accuracy_score(end_labels, end_preds),

        "acc_c_lab": accuracy_score(class_labels, class_preds)

    }

    return results
class Attention(nn.Module):

    """

    Computes a weighted average of channels across timesteps (1 parameter pr. channel).

    """

    def __init__(

        self, attention_size,

        pad_token_id,

        device

    ):

        super(Attention, self).__init__()

        self.attention_size = attention_size

        self.pad_token_id = pad_token_id

        self.device = device



        self.attention = nn.Parameter(torch.rand(attention_size))



    def forward(self, inputs, input_lengths):

        # inputs = [batch_size, max_seq_length, attention_size]

        # input_lengths = [batch_size]



        max_seq_length = inputs.shape[1]



        attn = torch.matmul(inputs, self.attention)

        # attn = [batch_size, max_seq_len]



        idxes = torch.arange(0, max_seq_length, out=torch.LongTensor(max_seq_length)).unsqueeze(0).to(self.device)

        mask = torch.autograd.Variable((idxes < input_lengths.unsqueeze(1)).float()).to(self.device)

        # mask = [batch_size, max_seq_length]

        # idxes = [batch_size, max_seq_length]



        attn_masked = attn.masked_fill(mask == self.pad_token_id, -1e10)

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

        self, max_seq_length, embedding_dim,

        attention_size, pad_token_id,

        hidden_size, num_layers,

        output_dim, bidirectional,

        dropout_ratio=0.2,

        common_embedding_layer=None,

        device=torch.device("cpu")

    ):

        super(AttentiveBilstm, self).__init__()

        self.max_seq_length = max_seq_length

        self.embedding_dim = embedding_dim

        self.hidden_size = hidden_size

        self.attention_size = attention_size

        self.pad_token_id = pad_token_id

        self.num_layers = num_layers

        self.output_dim = output_dim

        self.bidirectional = bidirectional

        self.dropout_ratio = dropout_ratio

        self.device = device



        self.lstm_layer = nn.LSTM(

            input_size=embedding_dim,

            num_layers=num_layers,

            hidden_size=hidden_size,

            bidirectional=bidirectional,

            dropout=dropout_ratio

        )

            

        if bidirectional is True:

            self.hidden2attention = nn.Linear(

                2*hidden_size, attention_size

            )

            self.attention_layer = Attention(

                attention_size=attention_size,

                pad_token_id=pad_token_id,

                device=self.device

            )

        else:

            self.hidden2attention = nn.Linear(

                1*hidden_size, attention_size

            )

            self.attention_layer = Attention(

                attention_size=attention_size,

                pad_token_id=pad_token_id,

                device=self.device

            )



        self.attention2output = nn.Linear(

            attention_size, output_dim

        )

        self.dropout_layer = nn.Dropout(dropout_ratio)

    

    def forward(self, ques_embedded, seq_lengths):

        # question = [batch_size, max_seq_length, embedding_dim]

        # seq_lengths = [batch_size]



        # permuting for pad packed easiness

        embedded = ques_embedded.permute(1, 0, 2)

        # embedded = [max_seq_length, batch_size, embedding_dim]



        packed_embedded = nn.utils.rnn.pack_padded_sequence(

            embedded, seq_lengths, enforce_sorted=False

        )

    

        packed_outputs, (_, _) = self.lstm_layer(packed_embedded)



        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, total_length=self.max_seq_length)

        # outputs = [max_seq_length, batch_size, num_directions*hidden_size]



        # using hidden2attention

        outputs = self.hidden2attention(outputs.permute(1, 0, 2))

        # outputs = [batch_size, max_seq_length, attention_size]



        # outputs are permuted again because attention layer needs batch_first

        (weighted_outputs, attention_weights) = self.attention_layer(outputs, seq_lengths)

        # weighted_outputs = [batch_size, attention_size]



        outputs = self.attention2output(weighted_outputs)



        return (outputs, attention_weights)

    



# Stanford Attentive Reader

# will be used in start and end token prediction

class StanfordAttentiveReader(nn.Module):

    def __init__(

        self, max_seq_length, embedding_dim,

        question_encoding_dim, pad_token_id,

        hidden_size, num_layers,

        bidirectional, dropout_ratio=0.2,

        common_embedding_layer=None,

        device=torch.device("cpu")

    ):

        super(StanfordAttentiveReader, self).__init__()

        self.max_seq_length = max_seq_length        

        self.embedding_dim = embedding_dim

        self.question_encoding_dim = question_encoding_dim

        self.pad_token_id = pad_token_id

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.bidirectional = bidirectional

        self.dropout_ratio = dropout_ratio

        self.device = device

        

        self.lstm_layer = nn.LSTM(

            input_size=embedding_dim,

            num_layers=num_layers,

            hidden_size=hidden_size,

            bidirectional=bidirectional,

            dropout=dropout_ratio

        )



        if bidirectional:

            self.bilinear_attention_start = nn.Parameter(

                torch.rand(

                    question_encoding_dim,

                    2*hidden_size

                )

            )

            self.bilinear_attention_end = nn.Parameter(

                torch.rand(

                    question_encoding_dim,

                    2*hidden_size

                )

            )

        else:

            self.bilinear_attention_start = nn.Parameter(

                torch.rand(

                    question_encoding_dim,

                    hidden_size

                )

            )

            self.bilinear_attention_end = nn.Parameter(

                torch.rand(

                    question_encoding_dim,

                    hidden_size

                )

            )

        

        self.dropout_layer = nn.Dropout(dropout_ratio)



    def forward(

        self, text_embedded=None, ques_encoded=None,

        text_lengths=None

    ):

        # text_embedded = [batch_size, max_text_seq_length, embedding_dim] 

        # ques_encoded = [batch_size, question_encoding_dim]

        # text_lengths = [batch_size]



        # permuting for pad packed easiness

        embedded = text_embedded.permute(1, 0, 2)

        # embedded = [max_text_seq_length, batch_size, embedding_dim]



        packed_embedded = nn.utils.rnn.pack_padded_sequence(

            embedded, text_lengths, enforce_sorted=False

        )

    

        packed_outputs, (_, _) = self.lstm_layer(packed_embedded)



        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, total_length=self.max_seq_length)

        # outputs = [max_text_seq_length, batch_size, num_directions*hidden_size]



        outputs = outputs.permute(1, 2, 0)

        # outputs = [batch_size, num_directions*hidden_size, max_text_seq_length]



        # calculating bilinear attentions

        # for starts

        start_outputs = self.calculate_bilinear_attention(

            self.bilinear_attention_start, outputs,

            ques_encoded, text_lengths

        )

        # for ends

        end_outputs = self.calculate_bilinear_attention(

            self.bilinear_attention_end, outputs,

            ques_encoded, text_lengths

        )

        # start_outputs = [batch_size, max_text_seq_len]

        # end_outputs = [batch_size, max_text_seq_len]

        return [start_outputs, end_outputs]

    

    def calculate_bilinear_attention(

        self, bilinear_attention_weights, lstm_outputs,

        ques_encoding, text_lengths

    ):  

        ques_encoding = ques_encoding.unsqueeze(1)

        # ques_encoding = []

        

        energy = torch.matmul(ques_encoding, bilinear_attention_weights)

        # energy = [batch_size, 1, 2*hidden_size]



        attended = torch.matmul(energy, lstm_outputs).squeeze(1)

        # attended = [batch_size, max_seq_text_length]



        # preparing masks

        idxes = torch.arange(0, self.max_seq_length, out=torch.LongTensor(self.max_seq_length)).unsqueeze(0).to(self.device)

        mask = torch.autograd.Variable((idxes < text_lengths.unsqueeze(1)).float()).to(self.device)

        # mask = [batch_size, max_seq_length]

        # idxes = [batch_size, max_seq_length]



        attention_masked = attended.masked_fill(mask == self.pad_token_id, -1e10)

        # attention_masked = [batch_size, max_text_seq_length]



        attention_out = F.softmax(attention_masked, dim=1)

        # attention_out = [batch_size, max_text_seq_length]

        return attention_out

    



class QALstmJointModel(nn.Module):

    def __init__(

        self,

        vocab_size, embedding_dim, num_labels,

        linear_layer_out, dropout_ratio,



        max_ques_seq_length, ques_pad_token_id,

        ques_encoding_attention_size,

        ques_encoding_dim,

        ques_enc_hidden_size, ques_enc_num_layers,

        ques_enc_bidirectional, ques_enc_dropout_ratio,



        max_text_seq_length,

        text_encoding_attention_size, text_encoding_dim,

        text_pad_token_id, 

        text_enc_hidden_size, text_enc_num_layers,

        text_enc_bidirectional, text_enc_dropout_ratio,

        text_reader_hidden_size, text_reader_num_layers,

        text_reader_bidirectional, text_reader_dropout_ratio,

        device=torch.device("cpu")

    ):

        super(QALstmJointModel, self).__init__()

        self.max_ques_seq_length = max_ques_seq_length

        self.max_text_seq_length = max_text_seq_length

        self.text_encoding_dim = text_encoding_dim

        self.ques_encoding_dim = ques_encoding_dim

        self.device = device



        # common embedding layer    

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        # self.embedding_layer_norm = nn.LayerNorm(embedding_dim)



        # question encoder

        self.question_encoder = AttentiveBilstm(

            max_seq_length=max_ques_seq_length,

            embedding_dim=embedding_dim,

            attention_size=ques_encoding_attention_size,

            pad_token_id=ques_pad_token_id,

            hidden_size=ques_enc_hidden_size, num_layers=ques_enc_num_layers,

            output_dim=ques_encoding_dim,

            bidirectional=ques_enc_bidirectional, dropout_ratio=ques_enc_dropout_ratio,

            common_embedding_layer=self.embedding_layer,

            device=self.device

        )

        

        # text encoder

        self.text_encoder = AttentiveBilstm(

            max_seq_length=max_text_seq_length,

            embedding_dim=embedding_dim,

            attention_size=text_encoding_attention_size,

            pad_token_id=text_pad_token_id,

            hidden_size=text_enc_hidden_size,

            num_layers=text_enc_num_layers,

            output_dim=text_encoding_dim,

            bidirectional=text_enc_bidirectional,

            dropout_ratio=text_enc_dropout_ratio,

            common_embedding_layer=self.embedding_layer,

            device=self.device

        )



        # attentive reader

        self.reader = StanfordAttentiveReader(

            max_seq_length=max_text_seq_length,

            embedding_dim=embedding_dim,

            question_encoding_dim=ques_encoding_dim,

            pad_token_id=text_pad_token_id,

            hidden_size=text_reader_hidden_size,

            num_layers=text_reader_num_layers,

            bidirectional=text_reader_bidirectional,

            dropout_ratio=text_reader_dropout_ratio,

            common_embedding_layer=self.embedding_layer,

            device=self.device

        )



        # linear layer before classifier

        self.linear1 = nn.Linear(

            ques_encoding_dim + text_encoding_dim,

            linear_layer_out

        )

        # label classifier

        self.label_classifier_layer = nn.Linear(

            linear_layer_out,

            num_labels

        )

        # dropout layer

        self.dropout_layer = nn.Dropout(dropout_ratio)

    

    def forward(

        self,

        text_input_ids, ques_input_ids,

        text_lengths, ques_lengths

    ):

        # text_input_ids = [batch_size, max_text_seq_len]

        # ques_input_ids = [batch_size, max_ques_seq_len]

        # text_lengths = [batch_size]

        # ques_lengths = [batch_size]



        # embedding

        text_embedded = self.dropout_layer(

            self.embedding_layer(text_input_ids)

        )

        ques_embedded = self.dropout_layer(

            self.embedding_layer(ques_input_ids)

        )

        text_encoding, _ = self.text_encoder(

            text_embedded, text_lengths

        )

        # text_encoding = [batch_size, text_encoding_dim]



        ques_encoding, _ = self.question_encoder(

            ques_embedded, ques_lengths

        )

        # ques_encoding = [batch_size, ques_encoding_dim]



        ques_and_text_encoding = torch.cat(

            (ques_encoding, text_encoding),

            dim=1

        )

        # ques_and_text_encoding = [batch_size, ques_encoding_dim + text_encoding_dim]



        linear_1_out = self.linear1(self.dropout_layer(ques_and_text_encoding))

        # linear_1_out = [batch_size, linear_layer_out]



        classifier_out = self.label_classifier_layer(linear_1_out)

        # classifier_out = [batch_size, num_labels]



        # attentive outputs 

        start_outputs, end_outputs = self.reader(

            text_embedded, ques_encoding,

            text_lengths

        )

        # start_outputs = [batch_size, max_text_seq_len]

        # end_outputs = [batch_size, max_text_seq_len]



        return classifier_out, start_outputs, end_outputs
DATA_DIR = Path('../input/tensorflow2-question-answering/')

DATA_PATH = DATA_DIR / 'simplified-nq-train.jsonl'





SEED = 1029

valid_size = 0

train_size = 307373 - valid_size



CHUNKSIZE = 1000

MAX_SEQ_LEN = 500

MAX_QUESTION_LEN = 64

DOC_STRIDE = 128



NUM_LABELS = 5

TRAIN_BATCH_SIZE = 64

DATALOADER_NUM_WORKERS = 4



bert_model = 'bert-base-uncased'

do_lower_case = 'uncased' in bert_model

# device = torch.device('cuda')



TOKENIZER = BertTokenizer.from_pretrained("../input/bert-config/vocab.txt", do_lower_case=do_lower_case)

CONFIG = BertConfig.from_pretrained("../input/bert-config/bert_config.json")



# output_model_file = 'bert_pytorch.bin'

# output_optimizer_file = 'bert_pytorch_optimizer.bin'

# output_amp_file = 'bert_pytorch_amp.bin'



random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

# torch.cuda.manual_seed(seed)

# torch.backends.cudnn.deterministic = True



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)
model = QALstmJointModel(

    vocab_size=TOKENIZER.vocab_size, embedding_dim=300, num_labels=NUM_LABELS,

    linear_layer_out=100, dropout_ratio=0.2,



    max_ques_seq_length=MAX_QUESTION_LEN, ques_pad_token_id=TOKENIZER.pad_token_id,

    ques_encoding_attention_size=64,

    ques_encoding_dim=64,

    ques_enc_hidden_size=32, ques_enc_num_layers=2,

    ques_enc_bidirectional=True, ques_enc_dropout_ratio=0.2,



    max_text_seq_length=MAX_SEQ_LEN,

    text_encoding_attention_size=256, text_encoding_dim=256,

    text_pad_token_id=TOKENIZER.pad_token_id, 

    text_enc_hidden_size=256, text_enc_num_layers=2,

    text_enc_bidirectional=True, text_enc_dropout_ratio=0.2,



    text_reader_hidden_size=256, text_reader_num_layers=2,

    text_reader_bidirectional=True, text_reader_dropout_ratio=0.2,

    device=DEVICE

)
print_number_of_trainable_parameters(model)
st_criterion = nn.CrossEntropyLoss(ignore_index=-1)

en_criterion = nn.CrossEntropyLoss(ignore_index=-1)

class_criterion = nn.CrossEntropyLoss()



optimizer = optim.Adam(model.parameters())

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
model = model.to(DEVICE)

st_criterion = st_criterion.to(DEVICE)

en_criterion = en_criterion.to(DEVICE)

class_criterion = class_criterion.to(DEVICE)
print_stats_at_step = 10

accumulation_steps = 2

tr_loss = 0.0

avg_tr_loss = 0.0



start_preds = None

start_out_label_ids = None



end_preds = None

end_out_label_ids = None



class_preds = None

class_out_label_ids = None



convert_func = functools.partial(

    convert_data,

    tokenizer=TOKENIZER,

    max_seq_len=MAX_SEQ_LEN,

    max_question_len=MAX_QUESTION_LEN,

    doc_stride=DOC_STRIDE

)



data_reader = JsonChunkReader(DATA_PATH, convert_func, chunksize=CHUNKSIZE)



step = 0

model.train()

for epoch in range(1):

    print(f'Epoch going on is {epoch}')

    for examples in data_reader:

        train_dataset = TextDataset(examples)

        train_loader = DataLoader(

            train_dataset, batch_size=TRAIN_BATCH_SIZE,

            shuffle=True, collate_fn=collate_fn, 

            num_workers=DATALOADER_NUM_WORKERS

        )



        generator_batch_iterator = tqdm(train_loader)

        for batch_idx, batch_data in enumerate(generator_batch_iterator):

            if batch_data[0].shape[0] == TRAIN_BATCH_SIZE:



                batch_data = tuple(t.to(DEVICE) for t in batch_data)

                inputs = {

                    "text_input_ids": batch_data[0],

                    "ques_input_ids": batch_data[1],

                    "text_lengths": batch_data[2], 

                    "ques_lengths": batch_data[3]

                }

                true_start_positions = batch_data[5]

                true_end_positions = batch_data[6]

                true_class_labels = batch_data[7]



                # getting outputs 

                class_outputs, start_outputs, end_outputs = model(**inputs)

                # propagating loss backwards and scheduler and optimizer steps

                loss = triple_loss_function(

                    [start_outputs, end_outputs, class_outputs],

                    [true_start_positions, true_end_positions, true_class_labels],

                    st_criterion, en_criterion, class_criterion

                )

                loss.backward()

                if (step + 1) % accumulation_steps == 0:

                    optimizer.step()

                    scheduler.step()

                    model.zero_grad()



                step_loss = loss.item()

                tr_loss += step_loss

                avg_tr_loss += step_loss



                # for calculation of results matrix

                temp_start_preds = torch.max(F.softmax(start_outputs, dim=-1), 1)[1]

                start_mask_arr = temp_start_preds.ne(-1)

                temp_start_preds = temp_start_preds.masked_select(start_mask_arr)

                true_start_positions = true_start_positions.masked_select(start_mask_arr)



                temp_end_preds = torch.max(F.softmax(start_outputs, dim=-1), 1)[1]

                end_mask_arr = temp_end_preds.ne(-1)

                temp_end_preds = temp_end_preds.masked_select(end_mask_arr)

                true_end_positions = true_end_positions.masked_select(end_mask_arr)

                if start_preds is None:

                    start_preds = temp_start_preds.detach().cpu().numpy()

                    start_out_label_ids = true_start_positions.detach().cpu().numpy()



                    end_preds = temp_end_preds.detach().cpu().numpy()

                    end_out_label_ids = true_end_positions.detach().cpu().numpy()



                    class_preds = torch.max(F.softmax(class_outputs, dim=-1), 1)[1].detach().cpu().numpy()

                    class_out_label_ids = true_class_labels.detach().cpu().numpy()

                else:

                    start_preds = np.append(

                        start_preds,

                        temp_start_preds.detach().cpu().numpy(),

                        axis=0

                    )

                    start_out_label_ids = np.append(

                        start_out_label_ids,

                        true_start_positions.detach().cpu().numpy(),

                        axis=0

                    )

                    end_preds = np.append(

                        end_preds,

                        temp_end_preds.detach().cpu().numpy(),

                        axis=0

                    )

                    end_out_label_ids = np.append(

                        end_out_label_ids,

                        true_end_positions.detach().cpu().numpy(),

                        axis=0

                    )

                    class_preds = np.append(

                        class_preds,

                        torch.max(F.softmax(class_outputs, dim=-1), 1)[1].detach().cpu().numpy(),

                        axis=0

                    )

                    class_out_label_ids = np.append(

                        class_out_label_ids,

                        true_class_labels.detach().cpu().numpy(),

                        axis=0

                    )



                if step % print_stats_at_step == 0:

                    tr_loss = tr_loss / print_stats_at_step

                    results = get_results_dict(

                        [start_preds, end_preds, class_preds],

                        [start_out_label_ids, end_out_label_ids, class_out_label_ids]

                    )

                    # writing on bar

                    generator_batch_iterator.set_description(

                        f'Tr Iter: {step}, avg_step_loss: {tr_loss:.4f}, avg_tr_loss: {(avg_tr_loss / (step + 1)):.4f}, f1_c_labels: {results["f1_class_labels"]:.4f}, acc_s_tok: {results["acc_s_tok"]:.4f}, acc_e_tok: {results["acc_e_tok"]:.4f}, acc_c_lab: {results["acc_c_lab"]:.4f}'

                    )

                    tr_loss = 0.0

                    start_preds = None

                    start_out_label_ids = None

                    end_preds = None

                    end_out_label_ids = None

                    class_preds = None

                    class_out_label_ids = None

                step += 1

        del examples, train_dataset, train_loader

        break       

#         print(f"{generator_idx + 1} generator is completed.")