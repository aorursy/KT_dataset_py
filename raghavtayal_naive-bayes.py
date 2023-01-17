import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split as tts

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix

import string

import os

print(os.listdir("../input"))
#Readind data from csv file with latin1 data encoding

df = pd.read_csv('../input/spamDataset.csv',encoding = 'latin1')
df.info()
features = ['v1','v2']

df = df[features]
df['v1'] = df['v1'].replace({'ham':0, 'spam':1})

df = df.rename(columns = {'v1':'class', 'v2':'message'})
features = CountVectorizer(stop_words='english')

x_df = features.fit_transform(df['message']).toarray()

x_df = pd.DataFrame(data = x_df)

y_df = df['class']

x_df.head()

y_df.head()
X_train, X_test, Y_train, Y_test = tts(x_df, y_df, test_size = 0.3)

result = pd.concat([X_train, Y_train], axis=1, join_axes=[Y_train.index])

result.head()
totalCount = result.groupby('class').count()

result = result.groupby('class').sum()

result = pd.DataFrame(data = result)

result.head()

total = result.sum(axis=1)

total_spam = total.iloc[1]

total_ham = total.iloc[0]

total = total_ham + total_spam

print(total)
prob_total_ham = totalCount.iloc[0][0]/(totalCount.iloc[0][0]+totalCount.iloc[1][0])

prob_total_spam = totalCount.iloc[1][0]/totalCount.iloc[0][0]+totalCount.iloc[1][0]
result = np.array(result)

result1 = [x/total_ham for x in result[0] ]

result2 = [x/total_spam for x in result[1] ]
def NaiveBayes(x):

    prob_ham = 1

    prob_spam = 1

    for i in enumerate(x):

        if(i[1] != 0 ):

            if(result1[i[0]] == 0 and result2[i[0]] == 0):

                continue

            prob_ham *= result1[i[0]]

            prob_spam *= result2[i[0]]

            

    prob_ham *= prob_total_ham

    prob_spam *= prob_total_spam

    if(prob_ham >= prob_spam) :

        return 0

    else:

        return 1
Y_predicted = [NaiveBayes(x) for x in X_test.values]
result = confusion_matrix(Y_test, Y_predicted)

accuracy = (result[0][0] + result[1][1]) / (result[0][0] + result[0][1] + result[1][0] + result[1][1])

print(accuracy)