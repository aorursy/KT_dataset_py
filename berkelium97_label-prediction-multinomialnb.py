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
import nltk

import pandas as pd

import numpy as np
train = pd.read_csv("/kaggle/input/emotions-dataset-for-nlp/train.txt", delimiter=';', header=None, names=['sentence','label'])

test = pd.read_csv("/kaggle/input/emotions-dataset-for-nlp/test.txt", delimiter=';', header=None, names=['sentence','label'])

val = pd.read_csv("/kaggle/input/emotions-dataset-for-nlp/val.txt", delimiter=';', header=None, names=['sentence','label'])
df_data = pd.concat([train, test,val])



df_data
df_data.to_csv (r'exportdata.txt', index=False)

dt_data =  pd.read_csv("exportdata.txt")



dt_data
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import RegexpTokenizer



token = RegexpTokenizer(r'[a-zA-Z0-9]+')

cv = CountVectorizer(stop_words='english', ngram_range=(1,1), tokenizer = token.tokenize)

text = cv.fit_transform(dt_data['sentence'])

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(text,dt_data['label'], test_size=0.30, random_state=5)
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()

mnb.fit(X_train, y_train)
predicted = mnb.predict(X_test)
from sklearn import metrics

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix,classification_report
acc_score = metrics.accuracy_score(predicted,y_test)

prec_score = precision_score(y_test,predicted, average='macro')

recall = recall_score(y_test, predicted,average='macro')

f1 = f1_score(y_test,predicted,average='macro')

matrix = confusion_matrix(y_test,predicted)
print(str('Accuracy: '+'{:04.2f}'.format(acc_score*100))+'%')

print(str('Precision: '+'{:04.2f}'.format(prec_score*100))+'%')

print(str('Recall: '+'{:04.2f}'.format(recall*100))+'%')

print('F1 Score: ',f1)

print(matrix)
#Dummy data (test)

test_data = ['i feel sick','i am ecstatic my model works', 'i feel shitty', 'i feel lost', 'im petrified', 'i am worried']



test_result = mnb.predict(cv.transform(test_data))



print(test_result)