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

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
train_df.head()
train_df.info()
train_df['sentiment'].value_counts().plot(kind='bar')
test_df['sentiment'].value_counts().plot(kind='bar')
train_df.count()
train_df.dropna(inplace=True)
train_df.count()
def build_train_data(df):

    train_data = []

    for index in df.index:

        context = df.loc[index,'text']

        idx = df.loc[index,'textID']

        que = 'what is '+df.loc[index,'sentiment']

        ans = df.loc[index,'selected_text']

        start_pos = df.loc[index,'text'].find(df.loc[index,'selected_text'])

        

        qas = []

        answers = []

        ans_dict = {}

        qa_dict = {}

        ans_dict = {'text': ans,'answer_start':start_pos}

        answers.append(ans_dict)

        qa_dict = {'id':idx,'question': que,'is_impossible':False,'answers':answers}

        qas.append(qa_dict)

        cont_dict ={'context':context,'qas':qas}

        train_data.append(cont_dict)

    return train_data
build_train_data(train_df)[0]
test_df.head()
def build_test_data(df):

    test_data = []

    for index in df.index:

        context = df.loc[index,'text']

        idx = df.loc[index,'textID']

        que = 'what is '+df.loc[index,'sentiment']



        

        qas = []

        answers = []

        ans_dict = {}

        qa_dict = {}

        ans_dict = {'text': '__NONE__','answer_start':1000000}

        answers.append(ans_dict)

        qa_dict = {'id':idx,'question': que,'is_impossible':False,'answers':answers}

        qas.append(qa_dict)

        cont_dict ={'context':context,'qas':qas}

        test_data.append(cont_dict)

    return test_data
build_test_data(test_df)[0]
distil_accept_train = build_train_data(train_df)
distil_accept_train[0:1]
distil_accept_test = build_test_data(test_df)
distil_accept_test[0:1]
!pip install '/kaggle/input/simple-transformers-pypi/seqeval-0.0.12-py3-none-any.whl' -q
!pip install '/kaggle/input/simple-transformers-pypi/simpletransformers-0.22.1-py3-none-any.whl' -q
from simpletransformers.question_answering import QuestionAnsweringModel
model_path = '/kaggle/input/transformers-pretrained-distilbert/distilbert-base-uncased-distilled-squad'
%%time



model = QuestionAnsweringModel('distilbert',model_path,args={'reprocess_input_data': True,

                                     'overwrite_output_dir': True,                    

                                     'learning_rate': 5e-5,

                                     'num_train_epochs': 3,

                                     'max_seq_length': 192,

                                     'doc_stride': 64,

                                     'fp16': False,                      

                                    },use_cuda=True)
%%time

model.train_model(distil_accept_train)
pred =  model.predict(distil_accept_test)
pred[0:2]
sub_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
submit_df = pd.DataFrame.from_dict(pred)

sub_df['selected_text'] = submit_df['answer']

sub_df.to_csv('submission.csv',index=False)

print('successfully submit')
import shutil

# import os
#os.remove('train.json')
#shutil.rmtree('cache_dir/')