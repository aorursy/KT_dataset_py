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
import pandas as pd

import torch
URL= 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv'

data = pd.read_csv(URL)

train_data1 = data[data['SPLIT'] == 'train']

train_data_smiles2 = (train_data1["SMILES"].squeeze()).astype(str).tolist()

#train_data_smiles2

train_data = train_data_smiles2

chars = set()

for string in train_data:

    chars.update(string)



all_sys = sorted(list(chars)) + ['<bos>', '<eos>', '<pad>', '<unk>']



#chars = set()

#for string in train_data:

#    chars.update(string)

#all_sys = sorted(list(chars)) + ['<bos>', '<eos>', '<pad>', '<unk>']

vocab = all_sys

c2i = {c:i for i,c in enumerate(all_sys)}

i2c = {i: c for i, c in enumerate(all_sys)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vector = torch.eye(len(c2i))
def char2id(char):

    if char not in c2i:

        return c2i['<unk>']

    else:

        return c2i[char]





def id2char(id):

    if id not in i2c:

        return i2c[32]

    else:

        return i2c[id]



def string2ids(string,add_bos=False, add_eos=False):

    ids = [char2id(c) for c in string]

    if add_bos:

        ids = [c2i['<bos>']] + ids

    if add_eos:

        ids = ids + [c2i['<eos>']]

    return ids
def ids2string(ids, rem_bos=True, rem_eos=True):

    if len(ids) == 0:

        return ''

    if rem_bos and ids[0] == c2i['<bos>']:

        ids = ids[1:]

    if rem_eos and ids[-1] == c2i['<eos>']:

        ids = ids[:-1]

    string = ''.join([id2char(id) for id in ids])

    return string

def string2tensor(string, device='model'):

    ids = string2ids(string, add_bos=True, add_eos=True)

    tensor = torch.tensor(ids, dtype=torch.long,device=device if device == 'model' else device)

    return tensor

tensor = [string2tensor(string, device=device) for string in train_data]

vector = torch.eye(len(c2i))