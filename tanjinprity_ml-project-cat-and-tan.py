# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#!pip install tensorflow_datasets





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# import tensorflow as tf

# import tensorflow_datasets as tfds



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import re # regular expressions

from nltk.corpus import stopwords
data_train_file = "../input/random-acts-of-pizza/train.json"

data_test_file = "../input/random-acts-of-pizza/test.json"



df_train = pd.read_json(data_train_file)

df_test = pd.read_json(data_test_file)



# Number of rows of train and test datasets

train_rows = df_train.shape[0]

test_rows = df_test.shape[0]

pizza_distributed = 0

for i in range(df_train['requester_received_pizza'].count()):

    if df_train['requester_received_pizza'][i] == True:

        pizza_distributed += 1



baseline_val = pizza_distributed/ train_rows



print(baseline_val)
def log_loss(Y_true, Y_pred):

  """Returns the binary log loss for a list of labels and predictions.

  

  Args:

    Y_true: A list of (true) labels (0 or 1)

    Y_pred: A list of corresponding predicted probabilities



  Returns:

    Binary log loss

  """

  return -(Y_true * (np.log(Y_pred)) + (1 - Y_true) * (np.log(1 - Y_pred))).mean()

np_train_val = df_train['requester_received_pizza'].astype(int).to_numpy()



baseline_val_train = np.full((train_rows),baseline_val)



train_log_loss = log_loss(np_train_val, baseline_val_train)



print(train_log_loss)
df_submission = df_test[['request_id']].copy()



baseline_val_test = np.full((test_rows),baseline_val)

df_submission['requester_received_pizza'] = baseline_val_test



df_submission.to_csv('/kaggle/working/submission1.csv', index=False)
data_train_file = "../input/random-acts-of-pizza/train.json"

data_test_file = "../input/random-acts-of-pizza/test.json"



df_train = pd.read_json(data_train_file)

df_test = pd.read_json(data_test_file)



# Number of rows of train and test datasets

train_rows = df_train.shape[0]

test_rows = df_test.shape[0]
def review_wordlist(review, remove_stopwords=False):

    # 2. Removing non-letter.

    review_text = re.sub("[^a-zA-Z]"," ",review_text)

    # 3. Converting to lower case and splitting

    words = review_text.lower().split()

    # 4. Optionally remove stopwords

    if remove_stopwords:

        stops = set(stopwords.words("english"))     

        words = [w for w in words if not w in stops]

    

    return(words)
# word2vec expects a list of lists.

# Using punkt tokenizer for better splitting of a paragraph into sentences.



import nltk.data

#nltk.download('popular')



tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')