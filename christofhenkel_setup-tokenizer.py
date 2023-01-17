!pip install fairseq fastBPE
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from types import SimpleNamespace

from fairseq.data.encoders.fastbpe import fastBPE

from fairseq.data import Dictionary



class BERTweetTokenizer():

    

    def __init__(self,pretrained_path = 'pretrained_models/BERTweet_base_transformers/'):

        



        self.bpe = fastBPE(SimpleNamespace(bpe_codes= pretrained_path + "bpe.codes"))

        self.vocab = Dictionary()

        self.vocab.add_from_file(pretrained_path + "dict.txt")

        self.cls_token_id = 0

        self.pad_token_id = 1

        self.sep_token_id = 2

        self.pad_token = '<pad>'

        self.cls_token = '<s>'

        self.sep_token = '</s>'

        

    def bpe_encode(self,text):

        return self.bpe.encode(text)

    

    def encode(self,text,add_special_tokens=False):

        subwords = self.bpe.encode(text)

        input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()

        return input_ids

    

    def tokenize(self,text):

        return self.bpe_encode(text).split()

    

    def convert_tokens_to_ids(self,tokens):

        input_ids = self.vocab.encode_line(' '.join(tokens), append_eos=False, add_if_not_exist=False).long().tolist()

        return input_ids

    

    def decode_tokens(self, tokens):

        decoded = ' '.join(tokens).replace('@@ ','').strip()

        return decoded
tokenizer = BERTweetTokenizer('/kaggle/input/bertweet-base-transformers/')
tokenizer.encode('hello world')