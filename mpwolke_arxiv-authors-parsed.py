#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTvQTNjonNkaJjF74TzdfrkGVBp26ZvE9NmU1_SMoPCMvV-DxaW&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import json

file_path = '/kaggle/input/arxiv/authors-parsed.json'

with open(file_path) as json_file:

     json_file = json.load(json_file)

json_file
len(json_file)
# The unique characters in the file

vocab = sorted(set(json_file))

print(vocab)

len(vocab)
import tensorflow as tf
tf.__version__
char_to_ind = {u:i for i, u in enumerate(vocab)}

ind_to_char = np.array(vocab)

encoded_jason_file = np.array([char_to_ind[c] for c in json_file])

seq_len = 250

total_num_seq = len(json_file)//(seq_len+1)

total_num_seq
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ1gNwnp5hoB1jej945TKd5MYyNq-AvM73QH-L4hV6q3KVsKyvF&usqp=CAU',width=400,height=400)