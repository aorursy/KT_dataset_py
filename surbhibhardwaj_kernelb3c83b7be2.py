# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import collections

import unicodedata

import six

### IMPORTS

import torch

import os

from torch.utils import data

from tqdm import tqdm

import torch.nn.init as init

from sklearn.metrics import average_precision_score,f1_score

from datetime import datetime

import logging

import pickle

from sklearn import model_selection

from datetime import datetime

from sklearn import metrics

import sys

package_dir_a = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"

sys.path.insert(0, package_dir_a)



from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch, BertModel





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import shutil

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

sub = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
### TOKENIZER

def convert_to_unicode(text):

    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""

    if six.PY3:

        if isinstance(text, str):

            return text

        elif isinstance(text, bytes):

            return text.decode("utf-8", "ignore")

        else:

            raise ValueError("Unsupported string type: %s" % (type(text)))

    elif six.PY2:

        if isinstance(text, str):

            return text.decode("utf-8", "ignore")

        elif isinstance(text, unicode):

            return text

        else:

            raise ValueError("Unsupported string type: %s" % (type(text)))

    else:

        raise ValueError("Not running on Python2 or Python 3?")





def printable_text(text):

    """Returns text encoded in a way suitable for print or `tf.logging`."""



    # These functions want `str` for both Python2 and Python3, but in one case

    # it's a Unicode string and in the other it's a byte string.

    if six.PY3:

        if isinstance(text, str):

            return text

        elif isinstance(text, bytes):

            return text.decode("utf-8", "ignore")

        else:

            raise ValueError("Unsupported string type: %s" % (type(text)))

    elif six.PY2:

        if isinstance(text, str):

            return text

        elif isinstance(text, unicode):

            return text.encode("utf-8")

        else:

            raise ValueError("Unsupported string type: %s" % (type(text)))

    else:

        raise ValueError("Not running on Python2 or Python 3?")





def load_vocab(vocab_file):

    """Loads a vocabulary file into a dictionary."""

    vocab = collections.OrderedDict()

    index = 0

    with open(vocab_file, "r") as reader:

        while True:

            token = convert_to_unicode(reader.readline())

            if not token:

                break

            token = token.strip()

            vocab[token] = index

            index += 1

    return vocab





def convert_tokens_to_ids(vocab, tokens):

    """Converts a sequence of tokens into ids using the vocab."""

    ids = []

    for token in tokens:

        ids.append(vocab[token])

    return ids





def whitespace_tokenize(text):

    """Runs basic whitespace cleaning and splitting on a peice of text."""

    text = text.strip()

    if not text:

        return []

    tokens = text.split()

    return tokens





class FullTokenizer(object):

    """Runs end-to-end tokenziation."""



    def __init__(self, vocab_file, do_lower_case=True):

        self.vocab = load_vocab(vocab_file)

        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)



    def tokenize(self, text):

        split_tokens = []

        for token in self.basic_tokenizer.tokenize(text):

            for sub_token in self.wordpiece_tokenizer.tokenize(token):

                split_tokens.append(sub_token)



        return split_tokens



    def convert_tokens_to_ids(self, tokens):

        return convert_tokens_to_ids(self.vocab, tokens)





class BasicTokenizer(object):

    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""



    def __init__(self, do_lower_case=True):

        """Constructs a BasicTokenizer.



        Args:

          do_lower_case: Whether to lower case the input.

        """

        self.do_lower_case = do_lower_case



    def tokenize(self, text):

        """Tokenizes a piece of text."""

        text = convert_to_unicode(text)

        text = self._clean_text(text)

        orig_tokens = whitespace_tokenize(text)

        split_tokens = []

        for token in orig_tokens:

            if self.do_lower_case:

                token = token.lower()

                token = self._run_strip_accents(token)

            split_tokens.extend(self._run_split_on_punc(token))



        output_tokens = whitespace_tokenize(" ".join(split_tokens))

        return output_tokens



    def _run_strip_accents(self, text):

        """Strips accents from a piece of text."""

        text = unicodedata.normalize("NFD", text)

        output = []

        for char in text:

            cat = unicodedata.category(char)

            if cat == "Mn":

                continue

            output.append(char)

        return "".join(output)



    def _run_split_on_punc(self, text):

        """Splits punctuation on a piece of text."""

        chars = list(text)

        i = 0

        start_new_word = True

        output = []

        while i < len(chars):

            char = chars[i]

            if _is_punctuation(char):

                output.append([char])

                start_new_word = True

            else:

                if start_new_word:

                    output.append([])

                start_new_word = False

                output[-1].append(char)

            i += 1



        return ["".join(x) for x in output]



    def _clean_text(self, text):

        """Performs invalid character removal and whitespace cleanup on text."""

        output = []

        for char in text:

            cp = ord(char)

            if cp == 0 or cp == 0xfffd or _is_control(char):

                continue

            if _is_whitespace(char):

                output.append(" ")

            else:

                output.append(char)

        return "".join(output)





class WordpieceTokenizer(object):

    """Runs WordPiece tokenization."""



    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):

        self.vocab = vocab

        self.unk_token = unk_token

        self.max_input_chars_per_word = max_input_chars_per_word



    def tokenize(self, text):

        """Tokenizes a piece of text into its word pieces.



        This uses a greedy longest-match-first algorithm to perform tokenization

        using the given vocabulary.



        For example:

          input = "unaffable"

          output = ["un", "##aff", "##able"]



        Args:

          text: A single token or whitespace separated tokens. This should have

            already been passed through `BasicTokenizer.



        Returns:

          A list of wordpiece tokens.

        """



        text = convert_to_unicode(text)



        output_tokens = []

        for token in whitespace_tokenize(text):

            chars = list(token)

            if len(chars) > self.max_input_chars_per_word:

                output_tokens.append(self.unk_token)

                continue



            is_bad = False

            start = 0

            sub_tokens = []

            while start < len(chars):

                end = len(chars)

                cur_substr = None

                while start < end:

                    substr = "".join(chars[start:end])

                    if start > 0:

                        substr = "##" + substr

                    if substr in self.vocab:

                        cur_substr = substr

                        break

                    end -= 1

                if cur_substr is None:

                    is_bad = True

                    break

                sub_tokens.append(cur_substr)

                start = end



            if is_bad:

                output_tokens.append(self.unk_token)

            else:

                output_tokens.extend(sub_tokens)

        return output_tokens





def _is_whitespace(char):

    """Checks whether `chars` is a whitespace character."""

    # \t, \n, and \r are technically contorl characters but we treat them

    # as whitespace since they are generally considered as such.

    if char == " " or char == "\t" or char == "\n" or char == "\r":

        return True

    cat = unicodedata.category(char)

    if cat == "Zs":

        return True

    return False





def _is_control(char):

    """Checks whether `chars` is a control character."""

    # These are technically control characters but we count them as whitespace

    # characters.

    if char == "\t" or char == "\n" or char == "\r":

        return False

    cat = unicodedata.category(char)

    if cat.startswith("C"):

        return True

    return False





def _is_punctuation(char):

    """Checks whether `chars` is a punctuation character."""

    cp = ord(char)

    # We treat all non-letter/number ASCII as punctuation.

    # Characters such as "^", "$", and "`" are not in the Unicode

    # Punctuation class but we treat them as punctuation anyways, for

    # consistency.

    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or

            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):

        return True

    cat = unicodedata.category(char)

    if cat.startswith("P"):

        return True

    return False

identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 

                    'christian', 'jewish','muslim', 'black',

                    'white', 'psychiatric_or_mental_illness']

TOXICITY_COLUMN = 'target'
BERT_MODEL_PATH = "../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/"

init_checkpoint_pt = os.path.join(BERT_MODEL_PATH, "pytorch_model.bin")

bert_config_file = os.path.join(BERT_MODEL_PATH, "bert_config.json")

vocab_file = os.path.join(BERT_MODEL_PATH, "vocab.txt")

do_lower_case = True

tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

device = torch.device('cuda')

model_save_path = '../input/trained-models/model_3_new.cpt'

max_seq_len = 500

WORK_DIR = "../working/"

convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(

    BERT_MODEL_PATH + 'bert_model.ckpt',

BERT_MODEL_PATH + 'bert_config.json',

WORK_DIR + 'pytorch_model.bin')



shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + 'bert_config.json')
model = BertModel.from_pretrained('../working/').to('cuda')

model.eval()
### fetching bert embeddings for the list

def Bert_embedder(sentences):

    tokenized_list = []

    indexed_list = []

    segments = []

    tokens_tensors = []

    segments_tensors = []

    for sents in sentences:

            tokens = tokenizer.tokenize(sents)

            if len(tokens) > max_seq_len:

                tokens_up = tokens[0:max_seq_len]

            else:

                

                tokens_up = tokens

            tokens_up = ["[CLS]"] + tokens_up + ["[SEP]"]

                

            tokenized_list.append(tokens_up)



    

    ### Padding the tokens

    pad_token = 1

    X_lengths = [len(sentence) for sentence in tokenized_list]

    padded_tokenized_list = [[]]

    longest_sent = max(X_lengths)

    batch_size = len(tokenized_list)

    indexed_list = np.zeros((batch_size, longest_sent)) * pad_token

    

        

        

    for i, tokenized_text in enumerate(tokenized_list):

        ids = tokenizer.convert_tokens_to_ids(tokenized_text)

        for j, toks in enumerate(ids):

            indexed_list[i, j] = toks

    

    

    for indexed_tokens in indexed_list:

        segments.append([0]*longest_sent)

    

    for indexed_tokens in indexed_list:

        tokens_tensors.append(torch.tensor([indexed_tokens]).to('cuda').long())

    for segment in segments :   

        segments_tensors.append(torch.tensor([segment]).to('cuda').long())

    

    embeddings = torch.Tensor().to('cuda')

    for i, (tokens_tensor, segments_tensor) in enumerate(zip(tokens_tensors,segments_tensors )): 

        with torch.no_grad():

            encoded_layers, _ = model(tokens_tensor, segments_tensor)

            embeddings = torch.cat((embeddings, encoded_layers[-1]), 0)

            

    return embeddings, X_lengths
class Toxic_classifier(torch.nn.Module):

    def __init__(self, input_dim, out_dim):

        super(Toxic_classifier, self).__init__()

        #self.lstm = torch.nn.LSTM(input_dim, out_dim, bidirectional=True, batch_first=True)

        self.lstm = torch.nn.LSTM(input_dim, out_dim, bidirectional=True, batch_first=True, dropout=0.5)

        self.Weights = torch.nn.Parameter(torch.Tensor(2*out_dim))

        self.Weights.data.uniform_(-0.1, 0.1)

        self.soft = torch.nn.Softmax()

        self.fc_layers = torch.nn.Linear(512, 128, bias=True)

        #self.linear = torch.nn.Linear(128, 64, bias=False)

        self.linear_layers = torch.nn.Linear(128, 2, bias=True)

        

               

    def forward(self, inputs, max_seq_len):

        lstm_out, _= self.lstm(inputs)

        prod = torch.einsum('ijk,k->ij', (lstm_out, self.Weights))  

        soft_out = self.soft(prod)

            #print(soft_out)

        attn_state_prod = torch.einsum('ij,ijl->ijl', (soft_out, lstm_out))

        sum_state = torch.sum(attn_state_prod, dim=1)

        fc_state = torch.relu(self.fc_layers(sum_state))

        #temp_state = torch.relu(self.linear(fc_state))

           

            #print(fc_state[action].size())

        final_out = self.linear_layers(fc_state)

        return final_out



        

            

            
Toxic_model = torch.load(model_save_path)

Toxic_model.eval()
### Create dataset and dataloader for loading batches 

class Dataset(data.Dataset):

    

    def __init__(self, data):

        self.data = data

       



    def __len__(self):

        return len(self.data)



    def __getitem__(self, index):

        record = self.data.iloc[index]



        return record.to_dict()
val_dataset = Dataset(test)

dataset_loader_test = data.DataLoader(val_dataset,batch_size=256, shuffle=False)
test['prediction'] = 0

validate_latest = pd.DataFrame()

submission = pd.DataFrame()
for batch in tqdm(dataset_loader_test):

    #print(torch.softmax(Toxic_model(batch['comment_text'], 400), 1))

    validate_latest = pd.DataFrame()

    embeddings, lengths =  Bert_embedder(batch['comment_text'])

    y_out = Toxic_model(embeddings, max_seq_len)

    #print(y_out[:,1].detach().cpu().numpy())

    y_out = torch.softmax(y_out, 1)

    batch_ids = batch['id'].detach().cpu().numpy().tolist()

    output = y_out[:,1].detach().cpu().numpy().tolist()

    validate_latest['id'] = batch_ids

    validate_latest['prediction'] = output

    submission = submission.append(validate_latest)
submission.to_csv('submission.csv', index=None)