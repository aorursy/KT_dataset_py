!pip install '/kaggle/input/simple-transformers-pypi/seqeval-0.0.12-py3-none-any.whl' -q

!pip install '/kaggle/input/simple-transformers-pypi/simpletransformers-0.22.1-py3-none-any.whl' -q
import json



train_data = list()



import pandas as pd



train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')



for id, row in train.iterrows():

    template = {

        'context': "",

        'qas': [

            {

                'id': "",

                'is_impossible': False,

                'question': "",

                'answers': [

                    {

                        'text': "",

                        'answer_start':''

                    }

                ]

            }

        ]

    }

    template['qas'][0]['id'] = row['textID']

    

    context = str(row['text'])

    question = row['sentiment']

    answer = str(row['selected_text'])

    

    template['context'] = context.lower()

    template['qas'][0]['question'] = question.lower()

    template['qas'][0]['answers'][0]['text'] = answer.lower()

    try:

        template['qas'][0]['answers'][0]['answer_start'] = context.index(answer)

    except AttributeError:

        print(id, row['text'], row['selected_text'])



    train_data.append(template)



with open('train_processed.json', 'w') as f:

    json.dump(train_data, f)
import json



test_data = list()



import pandas as pd



test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')



for id, row in test.iterrows():

    template = {

        'context': "",

        'qas': [

            {

                'id': "",

                'is_impossible': False,

                'question': "",

                'answers': [

                    {

                        'text': "",

                        'answer_start': ''

                    }

                ]

            }

        ]

    }



    template['context'] = str(row['text']).lower()

    template['qas'][0]['id'] = row['textID']

    template['qas'][0]['question'] = str(row['sentiment']).lower()

    test_data.append(template)



with open('test_processed.json', 'w') as f:

    json.dump(test_data, f)
from simpletransformers.question_answering import QuestionAnsweringModel

import json

import os

import logging





logging.basicConfig(level=logging.INFO)

transformers_logger = logging.getLogger("transformers")

transformers_logger.setLevel(logging.WARNING)



with open('train_processed.json', 'r') as f:

    train_data = json.load(f)



train_args = {

    'reprocess_input_data': True,

    'use_multiprocessing': False,

#     'use_early_stopping': True,

#     'early_stopping_patience': 7,

#     'weight_decay': 0.000001,

    'do_lower_case': True,

    "wandb_project": False,

    'learning_rate': 5e-5,

    'num_train_epochs': 3,

    'max_seq_length': 192,

    'doc_stride': 64,

    'overwrite_output_dir': True,

    'reprocess_input_data': False,

#     'train_batch_size': 8,

#     'gradient_accumulation_steps': 1,

    'save_steps': 0,

    'fp16': False,

    'save_eval_checkpoints': False,

    'save_model_every_epoch': False

}



arch = 'albert'

m = '/kaggle/input/pretrained-albert-pytorch/albert-large-v1'



# arch = 'distilbert'

# m = '/kaggle/input/transformers-pretrained-distilbert/distilbert-base-uncased-distilled-squad/'



# m = 'outputs/best_model'

# m = 'outputs/checkpoint-1644-epoch-2'



# Create the QuestionAnsweringModel

model = QuestionAnsweringModel(arch, m,

                               args=train_args,

                               use_cuda=True

                               )



# Train the model with JSON file

# model.train_model()



model.train_model(train_data)





with open('test_processed.json', 'r') as f:

    test_data = json.load(f)



import pandas as pd

pred = model.predict(test_data)



final_output = list()



for p in pred:

    idText = p['id']

    answer = p['answer']

    out = {'textID': idText, 'selected_text': answer}

    final_output.append(out)

    

out_df = pd.DataFrame(final_output)

out_df.to_csv('submission.csv', index=False)