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
import json

from sklearn.model_selection import train_test_split



train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv').fillna('')

test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sub_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')



train_df=train_df[train_df['sentiment'] != 'neutral']





train = np.array(train_df)

test = np.array(test_df)

#valdf=np.array(val1)



!mkdir -p data



use_cuda = True

def find_all(input_str, search_str):

    l1 = []

    length = len(input_str)

    index = 0

    

    while index < length:

        i = input_str.find(search_str, index)

        if i == -1:

            return l1

        l1.append(i)

        index = i + 1

    return l1

def do_qa_train(train):



    output = {}

    output['version'] = 'v1.0'

    output['data'] = []

    paragraphs = []

    for line in train:

        context = line[1]



        qas = []

        question = line[-1]

        qid = line[0]

        answers = []

        answer = line[2]

        if type(answer) != str or type(context) != str or type(question) != str:

            print(context, type(context))

            print(answer, type(answer))

            print(question, type(question))

            continue

        answer_starts = find_all(context, answer)

        for answer_start in answer_starts:

            answers.append({'answer_start': answer_start, 'text': answer.lower()})

            break

        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})



        paragraphs.append({'context': context.lower(), 'qas': qas})

        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

        

    return paragraphs

















 

qa_train = do_qa_train(train)

with open('data/train.json', 'w') as outfile:

    json.dump(qa_train, outfile)

output = {}

output['version'] = 'v1.0'

output['data'] = []



#qa_val = do_qa_train(valdf)

#with open('data/val.json', 'w') as outfile:

#    json.dump(qa_val, outfile)

#output = {}

#output['version'] = 'v1.0'

#output['data'] = []

def do_qa_test(test):

    paragraphs = []

    for line in test:

        context = line[1]

        qas = []

        question = line[-1]

        qid = line[0]

        if type(context) != str or type(question) != str:

            print(context, type(context))

            print(answer, type(answer))

            print(question, type(question))

            continue

        answers = []

        answers.append({'answer_start': 1000000, 'text': '__None__'})

        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})



        paragraphs.append({'context': context.lower(), 'qas': qas})

        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

    return paragraphs



qa_test = do_qa_test(test)

#val=do_qa_test(valdf)

print('done')

with open('data/test.json', 'w') as outfile:

    json.dump(qa_test, outfile)

!pip install '/kaggle/input/simple-transformers-pypi/seqeval-0.0.12-py3-none-any.whl' -q

!pip install '/kaggle/input/simple-transformers-pypi/simpletransformers-0.22.1-py3-none-any.whl' -q
from simpletransformers.question_answering import QuestionAnsweringModel



MODEL_PATH = '/kaggle/input/bert-pretrained-transformers/'



model = QuestionAnsweringModel('bert', 

                               MODEL_PATH, 

                               args={'reprocess_input_data': True,

                                     'overwrite_output_dir': True,

                                     'learning_rate': 3e-5,

                                     'num_train_epochs': 2,

                                     'max_seq_length': 96,

                                     'doc_stride': 64,

                                     'fp16': False

                                     

                                    },

                              use_cuda=use_cuda)



model.train_model('data/train.json')

predictions = model.predict(qa_test)



predictions_df = pd.DataFrame.from_dict(predictions)



sub_df['selected_text'] = predictions_df['answer']





print("File submitted successfully.")
sub_df['selected_text'][test_df['sentiment']=='neutral'] = test_df['text'][test_df['sentiment']=='neutral']



sub_df['selected_text'] = sub_df['selected_text'].map(lambda x: x.lstrip(' '))

sub_df.to_csv('submission.csv', index=False)