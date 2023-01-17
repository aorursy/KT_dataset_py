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
ed_file = '/kaggle/input/english-dutch-text-classification-nlp-beginner/train_new.txt'

ed_label = []

ed_text = []



def read_file(ed_file):

    file = open(ed_file,"r")

    read_data=file.readlines()

    columns=read_data[0].split("|")

    columns[-1] = columns[-1].strip()  

    rows = read_data[:]

    rows = [string.rstrip('\n') for string in rows]

    print(rows[12])

    for i in range(len(rows)):

        columns = rows[i].split("|")

        columns[-1] = columns[-1].strip()  

        ed_label.append(columns[0])

        ed_text.append(columns[1])

    file.close()
read_file(ed_file)

print(ed_text[12])
# create a dataframe



ed_data = pd.DataFrame(list(zip(ed_text, ed_label)), columns =['text', 'label'])

ed_data.head()
from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
# split the dataset into training and testing data



X_train, X_test, y_train, y_test = train_test_split(ed_data['text'], ed_data['label'], train_size=0.2, random_state=1)
# initiate vectorizer



vectorizer = CountVectorizer()

vector = vectorizer.fit_transform(X_train)
# build the model



clf = MultinomialNB()

clf.fit(vector, y_train)
# make prediction



vector_test = vectorizer.transform(X_test)

y_pred = clf.predict(vector_test)
from sklearn import metrics

acc_score = metrics.accuracy_score(y_test, y_pred)

print('Total accuracy classification score: {}'.format(acc_score))
# compare actual and predicted label

def pred(msg):

    msg = vectorizer.transform([msg])

    prediction = clf.predict(msg)

    return prediction[0]

for i in range(240,260,4):

    print(ed_data['text'].iloc[i][:50], "...")

    print("Actual label: ", ed_data['label'][i])

    print("predicted label: ", pred(ed_data['text'][i]))