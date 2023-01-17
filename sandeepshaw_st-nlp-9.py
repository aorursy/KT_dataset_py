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
a=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

b=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

c=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
# Add the following

#'https://www.kaggle.com/jonathanbesomi/simple-transformers-pypi'

#'https://www.kaggle.com/jonathanbesomi/transformers-pretrained-distilbert'
!pip install '/kaggle/input/simple-transformers-pypi/seqeval-0.0.12-py3-none-any.whl' -q

!pip install '/kaggle/input/simple-transformers-pypi/simpletransformers-0.22.1-py3-none-any.whl' -q
a.head()

a['text'][a['text'].isnull()]='0'

a['selected_text'][a['selected_text'].isnull()]='0'

a.isnull().sum()
def f(x):

    return(x[1].find(x[2]))

    

a['str_len']=a.apply(f, axis=1) #Imp



def f1(x):

    return({'context':x[1], 'qas':[{'question':x[3],'id':x[0],'is_impossible':False,'answers':[{'answer_start':x[4],'text':x[2]}]}]})



a['dict']=a.apply(f1, axis=1)



a.loc[0]



train=a['dict']

train[:3]



outer_list=[]

len(train)



train[0]



for i in range(len(train)):

    outer_list.append(train[i])



outer_list[:3]



train=outer_list



train[:3]
from simpletransformers.question_answering import QuestionAnsweringModel
#MODEL='/kaggle/input/transformers-pretrained-distilbert/distilbert-base-uncased-distilled-squad/'

#model=QuestionAnsweringModel('distilbert', MODEL, use_cuda=True)



MODEL = '/kaggle/input/transformers-pretrained-distilbert/distilbert-base-uncased-distilled-squad/'



model = QuestionAnsweringModel('distilbert',  

                               MODEL,

                               args={'reprocess_input_data': True,

                                     'overwrite_output_dir': True,

                                     'learning_rate': 5e-5,

                                     'num_train_epochs': 2,

                                     'max_seq_length': 192,

                                     'doc_stride': 64,

                                     'fp16': False

                                    }, 

                               use_cuda=True

                              )
train
import os

import json

import numpy as np



os.makedirs('data', exist_ok = True)



with open('data/train.json', 'w') as f:

    json.dump(train, f)

    f.close()
type(train)
model.train_model('data/train.json')
#for test

b.columns

b.isnull().sum()



def ft(x):

    return({'context':x[1], 'qas':[{'question':x[2],'id':x[0],'is_impossible':False,'answers':[{'answer_start':1000000,'text':'__None__'}]}]})



b['dict']=b.apply(ft, axis=1)



test=b['dict']

test[:3]



outer_test=[]

len(test)



test[0]



for i in range(len(test)):

    outer_test.append(test[i])



outer_test[:3]
test=outer_test
with open('data/test.json', 'w') as f:

    json.dump(test, f)

    f.close()
pred_df = model.predict(test)

pred_df = pd.DataFrame.from_dict(pred_df)
pred_df.head()
c.head()
c['selected_text']=pred_df['answer']
print(pred_df.shape)

print(c.shape)
c.to_csv("submission.csv", index=False)