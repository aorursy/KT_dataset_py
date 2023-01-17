import json

import pickle

import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

import numpy as np

import pandas as pd

from tqdm import tqdm_notebook as tqdm

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras.preprocessing.sequence import pad_sequences



PATH = '../input/google-quest-challenge/'

df_train = pd.read_csv(PATH+'train.csv')

df_test = pd.read_csv(PATH+'test.csv')

df_train
le = LabelEncoder()

categoria = df_test.category

train_categoria = le.fit_transform(categoria)

train_categoria = tf.keras.utils.to_categorical(train_categoria, num_classes=5)

x = df_test.columns[[1,2,5]]

x = df_test[x]

model = load_model('/kaggle/input/kernel454bd65e1b/modelo.h5')

model.summary()

x = df_test.columns[[1,2,5]]

x = df_test[x]

with open('/kaggle/input/kernel454bd65e1b/tokenizer.pickle', 'rb') as f:

    tokenizer = pickle.load(f)

test_title = tokenizer.texts_to_sequences(x.question_title)

test_question = tokenizer.texts_to_sequences(x.question_body)

test_answer = tokenizer.texts_to_sequences(x.answer)

test_title = pad_sequences(test_title)

test_question = pad_sequences(test_question)

test_answer = pad_sequences(test_answer)



test_target = model.predict([test_title,test_question,test_answer, train_categoria], batch_size=128)

test_target = np.concatenate((test_target[0],test_target[1]),axis=1)

[df_test.qa_id,test_target.tolist()]

np.array(df_test.qa_id).tolist()

test_target.tolist()

[np.array(df_test.qa_id).tolist(), test_target.tolist()]

np.array(df_test.qa_id).tolist()

sample_submission = []



for i in range(len(np.array(df_test.qa_id).tolist())):

    n = test_target.tolist()[i]



    n.insert(0,np.array(df_test.qa_id).tolist()[i])

    sample_submission.append(n)

    

submission = pd.DataFrame(sample_submission)



x 

yy = df_train.columns[11:]

headers = list(yy)

headers.insert(0,"qa_id")

submission.columns = headers

submission.to_csv('submission.csv', index=False)

submission.head()