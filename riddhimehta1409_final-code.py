import pandas as pd
import numpy as np
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
train.head()
test.head()
sample.head()
train.shape, test.shape, sample.shape
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text
train = np.array(train)
test = np.array(test)
train[:3]
test[:3]
def search(input_string, search_string):
    length = len(input_string)
    start_index = []
    length = len(input_string)
    index = 0
    while index < length:
        i = input_string.find(search_string, index)
        if i == -1:
            return start_index
        start_index.append(i)
        index = i + 1
    return start_index
## For example, we have:

search("hello I am having a good day today, what about you?", "good day today")

# This will return the value of the character index where the first letter of the search_string starts.
# In this case, it is the 20th position.
def convert_train_to_json(train_set): 
    
    outer_list = []
    
    for row in train_set:
        qid = row[0]        # As explained previously, this is the textID column value
        context = row[1]    # As explained previously, this is the text column value
        answer = row[2]     # As explained previously, this is the selected_text column value
        question = row[-1]  # As explained previously, this is the sentiment column value

                             # Here, we consider the sentiment value as "question" because given
                             # the sentiment we, then predict what should be the selected_text.
        inner_list = []              
        answers = []
        
        # We need to run the following IF command because if there are non string values then the code 
        # will throw an error and this is what we have to prevent. 
        # Hence, as soon as the error comes, we ask the code to CONTINUE.
        
        if type(context) != str or type(answer) != str or type(question) != str: 
            continue
        answer_starts = search(context, answer)
        for answer_start in answer_starts:
            answers.append({'answer_start': answer_start, 'text': answer.lower()})
            break
        inner_list.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})

        outer_list.append({'context': context.lower(), 'qas': inner_list})
        
    return outer_list
train = convert_train_to_json(train)
len(train)
train[:3]
def convert_test_to_json(test_set):
    
    outer_list = []
    
    for row in test_set:
        
        qid = row[0]
        context = row[1]
        question = row[-1]
        inner_list = []
                
        if type(context) != str or type(question) != str:
            continue
            
        answers = []
        answers.append({'answer_start': 1000000, 'text': '__None__'}) # Random initialisation of values
        inner_list.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
        outer_list.append({'context': context.lower(), 'qas': inner_list})
    return outer_list
test = convert_test_to_json(test)
len(test)
test[:3]
import os
import json

os.makedirs('data', exist_ok = True)

with open('data/train.json', 'w') as f:
    json.dump(train, f)
    f.close()
    
with open('data/test.json', 'w') as f:
    json.dump(test, f)
    f.close()
!pip install '/kaggle/input/simple-transformers-pypi/seqeval-0.0.12-py3-none-any.whl' -q
!pip install '/kaggle/input/simple-transformers-pypi/simpletransformers-0.22.1-py3-none-any.whl' -q
from simpletransformers.question_answering import QuestionAnsweringModel
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
model.train_model('data/train.json')
pred_df = model.predict(test)
pred_df = pd.DataFrame.from_dict(pred_df)
pred_df.head()
sample["selected_text"] = pred_df["answer"]
sample.to_csv('submission.csv', index=False)
print("Everything is successful! Good Luck for the score!")