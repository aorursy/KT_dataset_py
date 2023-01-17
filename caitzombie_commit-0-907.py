! pip install pymorphy2

! pip install pymystem3

! pip install nltk

! pip install --upgrade pip

! pip install razdel

! pip install fasttext
import pandas as pd

import re

import pymorphy2

from nltk.tokenize import word_tokenize

from scipy import sparse

import numpy as np

from pymystem3 import Mystem

from pymorphy2 import MorphAnalyzer

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from razdel import tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report, precision_recall_fscore_support

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from sklearn.linear_model import LogisticRegression

import tqdm

from xgboost import XGBClassifier

from string import punctuation

from tqdm._tqdm_notebook import tqdm_notebook

tqdm_notebook.pandas()



stopwords = stopwords.words('russian')

pymorphy2_analyzer = MorphAnalyzer()

mystem_analyzer = Mystem()



def train_classif(train_vectors, test_vectors, y_train, y_test, random_state=42):

    

    classifiers = {

    #"XGBoost": OneVsRestClassifier(XGBClassifier(seed=42)),

    #"LogisticRegression": OneVsRestClassifier(LogisticRegression(random_state=42)),    

    "Linear SVM": OneVsRestClassifier(LinearSVC(random_state=random_state)),

    #"One vs. one linear svm": OneVsOneClassifier(LinearSVC(random_state=1)),

      }



    for name, clf in classifiers.items():

        print(name)

        clf.fit(train_vectors, y_train)

        print(classification_report(y_test, clf.predict(test_vectors), digits=4))
def delete_punctuation(s):

    return re.sub(r'[^A-Za-zА-Яа-я0-9=_]+', ' ', s)



def tokenizer(s):

    return ' '.join([ _.text for _ in list(tokenize(s)) if _.text!=''])

    

def find_numbers(s):

    return re.sub(r'\d{5,}', 'номер', s)



def find_key_words(s, regexp):

    return len(regexp.findall(s))

                  

def normalize(s):

    #return [pymorphy2_analyzer.normal_forms(word)[0] if word not in '=-_' else word  for word in s]

    return mystem_analyzer.lemmatize(s)[:-1]



def identity_tokenizer(text):

    return text
train=pd.read_csv('../input/tinkoff/train.csv', index_col='id')

test=pd.read_csv('../input/tinkoff/test.csv', index_col='id')
train.text=train.text.str.lower()

test.text=test.text.str.lower()

#train.text=train.text.progress_apply(lambda x: re.sub(r'https?:\/\/.*[\r\n]*', 'http', x, flags=re.MULTILINE))

#train.text=train.text.progress_apply(delete_punctuation)

#train.text=train.text.progress_apply(tokenizer)
train.text=train.text.progress_apply(normalize)

test.text=test.text.progress_apply(normalize)
vectorizer=TfidfVectorizer(lowercase=False, tokenizer=identity_tokenizer)

vectorizer.fit(train.text.append(test.text))

train_vectors=vectorizer.transform(train.text)

X_train, X_test, y_train, y_test = train_test_split(train_vectors, train.label, test_size=0.2, random_state=42)
test_vectors=vectorizer.transform(test.text)
train_classif(X_train, X_test, y_train, y_test, 1)
train.text=train.text.apply(lambda x: ''.join(x))
python='def|python|питон|lambda|django|джанго|pip|venv|линукс|crm|for|while|import|numpy|traceback|call|range|elif|github|requests'

ml='pandas|xgboost|train|test|kaggle|kmeans|кластер|классиф|fit|predict|tfidf|embedding|интеллект|мл|каггл|cабмит|submit|public|private|паблик|приват|буткемп|ods|ml|score|скор|tensorflow|датасет|тензор'

meets=' дев| парень| м | ж | муж| жен| лет|смс|sms|номер|познаком'

english=r'[a-z]'

russian=r'[а-я]'

vowels=r'[аяуеёоиюэыaeiou]'

consonants=r'[бвгджзйклмнпрстфхцчшщbcdfghjklmnpqrstvwxyz]'



python_regexp=re.compile(python)

ml_regexp=re.compile(ml)

meets_regexp=re.compile(meets)

english_regexp=re.compile(english)

russian_regexp=re.compile(russian)

vowels_regexp=re.compile(vowels)

consonants_regexp=re.compile(consonants)



train['python']=train.text.progress_apply(lambda x: find_key_words(x, python_regexp))

train['ml']=train.text.progress_apply(lambda x: find_key_words(x, ml_regexp))

train['meets']=train.text.progress_apply(lambda x: find_key_words(x, meets_regexp))

train['english']=train.text.progress_apply(lambda x: find_key_words(x, english_regexp))

train['russian']=train.text.progress_apply(lambda x: find_key_words(x, russian_regexp))

train['vowels']=train.text.progress_apply(lambda x: find_key_words(x, vowels_regexp))

train['consonants']=train.text.progress_apply(lambda x: find_key_words(x, consonants_regexp))



train['len_text']=train.text.str.len()

train['mean_word_length']=train.text.apply(lambda x: np.mean([len(word) for word in x.split()]))

train['max_word_length']=train.text.apply(lambda x: np.max([len(word) for word in x.split()] if x!='' else 0 ))
test.text=test.text.apply(lambda x: ''.join(x))
test['ml']=test.text.progress_apply(lambda x: find_key_words(x, ml_regexp))

test['meets']=test.text.progress_apply(lambda x: find_key_words(x, meets_regexp))

test['english']=test.text.progress_apply(lambda x: find_key_words(x, english_regexp))

test['len_text']=test.text.str.len()
train_features=train[[#'python',

                      'ml',

                      'meets',

                      'english',

#                       'russian',

#        'vowels',

#                       'consonants',

                      'len_text',

#                       'mean_word_length',

#        'max_word_length'

                     ]]
from sklearn.preprocessing import Normalizer

normal=Normalizer()

#train_features=pd.concat([train_features,  pd.DataFrame(ft_predict)], axis=1)

train_features_norm=normal.fit_transform(train_features)
test_features=test[[#'python',

                      'ml',

                      'meets',

                      'english',

#                       'russian',

#        'vowels',

#                       'consonants',

                      'len_text',

#                       'mean_word_length',

#        'max_word_length'

]]
test_features_norm=normal.transform(test_features)
train_vectors_features=sparse.hstack([train_vectors, train_features_norm])
X_train, X_test, y_train, y_test = train_test_split(train_vectors_features, train.label, test_size=0.2, random_state=42)
train_classif(X_train, X_test, y_train, y_test, 1)
import fastText
train_path='../input/train-test-ft/train_norm.txt'

test_path='../input/train-test-ft/test_norm.txt'

train_path='../input/ft-train-2/train_norm.txt'

ft=fastText.train_supervised(train_path)

ft.save_model("ft.bin")
print(*ft.test(test_path))
train.text=train.text.apply(lambda x: ''.join(x))
test.text=test.text.apply(lambda x: ''.join(x))
import re

def find_numbers(s):

    return re.sub(r'\d{5,}', 'номер', ' '.join(s.replace('-', '').replace(' - ', '').split(' ')))
train.text=train.text.apply(lambda x: find_numbers(x))
test.text=test.text.apply(lambda x: find_numbers(x))
ft_vectors= np.array([ft.get_sentence_vector(text) for text in train.text])
test_ft_vectors=np.array([ft.get_sentence_vector(text) for text in test.text])
ft_predict=np.array([int(ft.predict(text)[0][-1][-1]) for text in train.text])

test_ft_predict=np.array([int(ft.predict(text)[0][-1][-1]) for text in test.text])
ft_vectors.shape
all_vectors=sparse.hstack([train_vectors, train_features_norm, ft_vectors])

all_vectors
X_train, X_test, y_train, y_test = train_test_split(all_vectors, train.label, test_size=0.2, random_state=42)
train_classif(X_train, X_test, y_train, y_test, 42)
clf=OneVsRestClassifier(LinearSVC(random_state=1))

clf.fit(all_vectors, train.label)
all_test_vectors=sparse.hstack([test_vectors, test_features_norm, test_ft_vectors])

y_pred=clf.predict(all_test_vectors)
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(pd.DataFrame({'label':y_pred}, index=test.index))