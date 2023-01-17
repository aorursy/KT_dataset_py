# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install simpletransformers==0.40.0
!pip install transformers
!pip install tensorflow==2.1.0
!pip install tokenizers==0.7.0
!export CUDA_HOME=/usr/local/cuda-10.1
!git clone https://github.com/NVIDIA/apex
%cd apex
!pip install -v --no-cache-dir ./
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
import pandas as pd

df = pd.read_csv("/kaggle/input/test-data-for-fake-jobs/train_data.csv")
df.head()
df.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns
print(df["Фейк"].value_counts())
sns.barplot(df["Фейк"].unique(), df["Фейк"].value_counts())
df.fillna(' ',inplace=True)
df["text"] = df['Название'] + ' ' + df['Место'] + ' ' + df['Отдел'] + ' ' + df['Описание компании'] + ' ' + df['Описание вакансии'] + ' ' + df['Требования'] + ' ' + df['Соцпакет'] + ' ' + df['Тип занятости'] + ' ' + df['Образование'] + ' ' + df['Индустрия'] + ' ' + df['Позиция']
del df['Название']
del df['Место']
del df['Отдел']
del df['Описание компании']
del df['Описание вакансии']
del df['Требования']
del df['Соцпакет']
del df['Тип занятости']
del df['Опыт']
del df['Образование']
del df['Индустрия']
del df['Позиция']
del df['Дистанционно']
del df['Зарплата']
del df['Вопросы']
import spacy, re
#Data Cleanup

df['text']=df['text'].str.replace('\n','')
df['text']=df['text'].str.replace('\r','')
df['text']=df['text'].str.replace('\t','')
  
#This removes unwanted texts
df['text'] = df['text'].apply(lambda x: re.sub(r'[0-9]','',x))
df['text'] = df['text'].apply(lambda x: re.sub(r'[/(){}\[\]\|@,;.:-]',' ',x))
  
#Converting all upper case to lower case
df['text']= df['text'].apply(lambda s:s.lower() if type(s) == str else s)
  

#Remove un necessary white space
df['text']=df['text'].str.replace('  ',' ')

#Remove Stop words
nlp=spacy.load("en_core_web_sm")
df['text'] =df['text'].apply(lambda x: ' '.join([word for word in x.split() if nlp.vocab[word].is_stop==False ]))
# from nltk.corpus import stopwords
# import string
# c = set(stopwords.words('english'))
# punctuation = list(string.punctuation)
# stop.update(punctuation)
def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from tqdm import tqdm
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    final_text = []
    for i in tqdm(text.split()):
        if i.strip().lower() not in stop:
            pos = pos_tag([i.strip()])
            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))
            final_text.append(word.lower())
    return " ".join(final_text)  
# df.text = df.text.apply(lemmatize_words)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df["Фейк"], test_size = 0.20, stratify=df["Фейк"], random_state=777)

train_df = pd.DataFrame({0: df['text'], 1: df["Фейк"]})
test_df = pd.DataFrame({0: X_test, 1: y_test})
from simpletransformers.classification import ClassificationModel, ClassificationArgs
model_args = ClassificationArgs()
model_args.use_early_stopping = True
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "mcc"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 1
model_args.evaluate_during_training_steps = 1000
model_args.overwrite_output_dir = True
model_args.train_batch_size = 64
model_args.save_steps = 10000
model_args.save_model_every_epoch = False
model_args.num_train_epochs = 10

# model = ClassificationModel('albert', 'albert-base-v2', num_labels=2, args={'overwrite_output_dir': True, "train_batch_size": 64, "save_steps": 10000, "save_model_every_epoch":False,
#                                                                           'num_train_epochs': 5}, use_cuda=True)
# model = ClassificationModel('bert', 'bert-base-uncased', num_labels=2, args=model_args, use_cuda=True)

model = ClassificationModel('bert', 'bert-base-uncased', num_labels=2, args={'overwrite_output_dir': True, "train_batch_size": 64, "save_steps": 10000, "save_model_every_epoch":False,
                                                                           'num_train_epochs': 5}, use_cuda=True)
model.train_model(train_df)
# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(test_df)
import numpy as np
preds = [np.argmax(tuple(m)) for m in model_outputs]
from sklearn.metrics import f1_score

print(f1_score(test_df[1], preds, average='micro'))
print(f1_score(test_df[1], preds, average='macro'))
print(f1_score(test_df[1], preds))
from sklearn.metrics import classification_report

print(classification_report(test_df[1], preds))
submission = pd.read_csv('/kaggle/input/test-data-for-fake-jobs/test_data.csv')
submission.head()
submission.fillna(" ",inplace = True)
submission["text"] = submission['Название'] + ' ' + submission['Место'] + ' ' + submission['Отдел'] + ' ' + submission['Описание компании'] + ' ' + submission['Описание вакансии'] + ' ' + submission['Требования'] + ' ' + submission['Соцпакет'] + ' ' + submission['Тип занятости'] + ' ' + submission['Образование'] + ' ' + submission['Индустрия'] + ' ' + submission['Позиция']
del submission['Название']
del submission['Место']
del submission['Отдел']
del submission['Описание компании']
del submission['Описание вакансии']
del submission['Требования']
del submission['Соцпакет']
del submission['Тип занятости']
del submission['Опыт']
del submission['Образование']
del submission['Индустрия']
del submission['Позиция']
del submission['Дистанционно']
del submission['Зарплата']
del submission['Вопросы']
submission.head()
ids = submission["Id"].copy()
submission = submission.drop(columns='Id')
submission.head()
submission['text']=submission['text'].str.replace('\n','')
submission['text']=submission['text'].str.replace('\r','')
submission['text']=submission['text'].str.replace('\t','')

#This removes unwanted texts
submission['text'] = submission['text'].apply(lambda x: re.sub(r'[0-9]','',x))
submission['text'] = submission['text'].apply(lambda x: re.sub(r'[/(){}\[\]\|@,;.:-]',' ',x))

#Converting all upper case to lower case
submission['text']= submission['text'].apply(lambda s:s.lower() if type(s) == str else s)


#Remove un necessary white space
submission['text']=submission['text'].str.replace('  ',' ')

#Remove Stop words
nlp=spacy.load("en_core_web_sm")
submission['text'] =submission['text'].apply(lambda x: ' '.join([word for word in x.split() if nlp.vocab[word].is_stop==False ]))
# submission.text = submission.text.apply(lemmatize_words)
submission['text']
predictions, raw_outputs = model.predict(submission.text)
predictions
result = pd.DataFrame()
result['Id'] = ids.values
result['Фейк'] = predictions
result = result.set_index('Id')
result
import os
os.chdir(r'/kaggle/working')

result.to_csv(r'lim_bert_submission_best5.csv', )
from IPython.display import FileLink
FileLink(r'lim_bert_submission_best5.csv')




