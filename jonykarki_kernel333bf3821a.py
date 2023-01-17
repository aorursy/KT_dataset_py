#Load the data

import numpy as np

import pandas as pd

import json



BASE_URL = "/kaggle/input/tweet-sentiment-extraction/"



train_df = pd.read_csv(BASE_URL+"train.csv")

test_df = pd.read_csv(BASE_URL+"test.csv")

sub_df = pd.read_csv(BASE_URL+"sample_submission.csv")
sub_df.head()
def qa_format_train(df):

    return [

            {

                'context': str(row[1]),

                'qas' : [{

                    'id': str(row[0]),

                    'is_impossible': False,

                    'question': str(row[3]),

                    'answers': [{

                        'text': str(row[2]),

                        'answer_start': str(row[1]).find(str(row[2]))

                    }]

                }],

            }

            for row in np.array(df)

    ]



def qa_format_test(df):

    return [

            {

                'context': row[1],

                'qas': [{

                    'question': row[2],

                    'id': row[0],

                    'is_impossible': False,

                    'answers': [{

                        'answer_start': 1000000,

                        'text': '__None__'

                    }]

                }]

            }

            for row in np.array(df)

    ]



qa_train = qa_format_train(train_df)

qa_test = qa_format_test(test_df)
qa_test
!pip install '/kaggle/input/simple-transformers-pypi/seqeval-0.0.12-py3-none-any.whl' -q

!pip install '/kaggle/input/simple-transformers-pypi/simpletransformers-0.22.1-py3-none-any.whl' -q
%%time



from simpletransformers.question_answering import QuestionAnsweringModel



model = QuestionAnsweringModel('distilbert', 

                               '/kaggle/input/transformers-pretrained-distilbert/distilbert-base-uncased-distilled-squad/', 

                               args={'reprocess_input_data': True,

                                     'overwrite_output_dir': True,

                                     'learning_rate': 5e-5,

                                     'num_train_epochs': 4,

                                     'max_seq_length': 200,

                                     'doc_stride': 64,

                                     'fp16': False,

                                    },

                              use_cuda=True)

model.train_model(qa_train)
%%time



preds = model.predict(qa_test)

predic_df = pd.DataFrame.from_dict(preds)

sub_df['selected_text'] = predic_df['answer']

sub_df.to_csv("submission.csv", sep=',', index=False)
sub_df.head()