import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.naive_bayes import MultinomialNB
df = pd.read_csv('/kaggle/input/sms-spam/spamraw.csv')

df
df.describe()
spam_message  = df[df['type'] == 'spam']

spam_message.head()
spam_message.describe()
ham_message = df[df['type'] == 'ham']

ham_message.head()
ham_message.describe()
sns.countplot(data=df, x=df['type']).set_title("Amount Of Spam and Ham Message")

plt.show()
data_train, data_test, labels_train, labels_test = train_test_split(df.text, df.type, test_size=0.2, random_state=0)

print("data_train, labels_train : ",data_train.shape, labels_train.shape)

print("data_test, labels_test: ",data_test.shape, labels_test.shape)
vectorizer = CountVectorizer()

#fit & transform

# fit: build dict (i.e. word->wordID)  

# transform: convert document (i.e. each line in the file) to word vector 

data_train_count = vectorizer.fit_transform(data_train)

data_test_count  = vectorizer.transform(data_test)
data_train_count
data_test_count
data_train
data_test
clf = MultinomialNB()

clf.fit(data_train_count, labels_train)

predictions = clf.predict(data_test_count)

predictions
# for i in range(len(predictions)):

#     print(predictions[i], data_test_count[i])
print ("accuracy_score : ", accuracy_score(labels_test, predictions))
print ("confusion_matrix : \n", confusion_matrix(labels_test, predictions))
print (classification_report(labels_test, predictions))