import pandas as pd

import numpy as np
df = pd.read_csv('../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv', sep=',')
df.head()
len(df)
df.isnull().sum()
df['Category'].unique()
df['Category'].value_counts()
print('Percent Ham messages : {}'.format(round(len(df[df['Category']=='ham'])/len(df)*100,2)))

print('Percent Spam messages : {}'.format(round(len(df[df['Category']=='spam'])/len(df)*100,2)))
length = []

for i in df['Message']:

    length.append(len(i))

df['Length'] = length
df.head()
df['Length'].describe()
import matplotlib.pyplot as plt

%matplotlib inline



plt.xscale('log')

bins = 1.15**(np.arange(0,50))

plt.hist(df[df['Category']=='ham']['Length'],bins=bins,alpha=0.8)

plt.hist(df[df['Category']=='spam']['Length'],bins=bins,alpha=0.8)

plt.legend(('ham','spam'))

plt.show()
from sklearn.model_selection import train_test_split



x = df['Message']  

y = df['Category']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('tfidf', TfidfVectorizer()),

                     ('clf', LinearSVC()),

])



text_clf.fit(x_train, y_train)  
pred= text_clf.predict(x_test)
from sklearn import metrics

print(metrics.confusion_matrix(y_test,pred))
print(metrics.classification_report(y_test,pred))
print(metrics.accuracy_score(y_test,pred))