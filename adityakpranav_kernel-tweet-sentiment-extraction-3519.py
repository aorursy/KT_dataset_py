# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

paths = []

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        paths.append(os.path.join(dirname, filename))

        

def read_train():

    train=pd.read_csv("../input/tweet-sentiment-extraction/train.csv")

    train['text']=train['text'].astype(str)

    train['selected_text']=train['selected_text'].astype(str)

    return train



def read_test():

    test=pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

    test['text']=test['text'].astype(str)

    return test



def read_submission():

    test=pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

    return test      

!mkdir -p data

!pip install '/kaggle/input/simple-transformers-pypi/seqeval-0.0.12-py3-none-any.whl' -q

!pip install '/kaggle/input/simple-transformers-pypi/simpletransformers-0.22.1-py3-none-any.whl' -q

print("installation done")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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



from simpletransformers.question_answering import QuestionAnsweringModel

from copy import deepcopy

import json



use_cuda = False

train_df = read_train()

test_df = read_test()

submission_df_distil = read_submission()





train = np.array(train_df)

test = np.array(test_df)

qa_train = do_qa_train(train)





with open('data/train.json', 'w') as outfile:

    json.dump(qa_train, outfile)

output = {}

output['version'] = 'v1.0'

output['data'] = []



qa_test = do_qa_test(test)



with open('data/test.json', 'w') as outfile:

    json.dump(qa_test, outfile)

def post_process(submission_df, test_df):



    index_to_selected_text = {}

    for i, row in test_df.iterrows():

        _id = row[0]

        text = row[1]

        sentiment = row[2]

        if len(text.split(" ")) <= 3 or sentiment == "neutral":

            index_to_selected_text[_id] = text

    

    submission_rows = submission_df.to_dict("records")

    new_rows = []

    for row in submission_rows:

        _id = row['textID']

        if _id in index_to_selected_text:

            new_row = deepcopy(row)

            new_row['selected_text'] = index_to_selected_text[_id]

        else:

            new_row = row

        

        new_rows.append(new_row)



    return pd.DataFrame(new_rows)

    

MODEL_PATH = '/kaggle/input/transformers-pretrained-distilbert/distilbert-base-uncased-distilled-squad/'

model = QuestionAnsweringModel('distilbert', 

                               MODEL_PATH, 

                              args={"reprocess_input_data": True,

                               "overwrite_output_dir": True,

                               "learning_rate": 8e-05,

                               "num_train_epochs": 3,

                               "max_seq_length": 192,

                               "weight_decay": 0.001,

                               "doc_stride": 64,

                               "save_model_every_epoch": False,

                               "fp16": False,

                               "do_lower_case": True,

                                 'max_query_length': 8,

                               'max_answer_length': 150

                                    },

                              use_cuda=True)



model.train_model('data/train.json')

predictions = model.predict(qa_test)



predictions_df = pd.DataFrame.from_dict(predictions)



submission_df_distil['selected_text'] = predictions_df['answer']



submission_df_distil = post_process(submission_df_distil, test_df)



submission_df_distil.to_csv('data/submission.csv', index=False)



submission_df_distil.to_csv('submission.csv', index=False)


