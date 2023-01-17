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
#read data

data = pd.read_csv('/kaggle/input/stocknews/Combined_News_DJIA.csv')

data.head()
#step1: data exploration

#1- data quantity: dataset<100000 

data.shape
#2-unstructured data --> text

data.columns
#3-statistics of quantitative data

# we have mean =53% 

data.describe()
#5-binary classification : Label (1,0)

#"1" when DJIA Adj Close value rose or stayed as the same;

#"0" when DJIA Adj Close value decreased.



data['Label'].value_counts().plot.bar()



#conclusion--> text classification: we use first Naive Bayes classification
#step2: text classification with Naive Bayes

#split our dataset for ML task

train = data[data['Date'] <='2014-12-31']

test = data[data['Date'] >='2015-01-02']



print (train.shape[0],test.shape[0])
#use a global context for all tops



train['Global_context']=train.iloc[:,2:].apply(lambda r: ''.join(str(r.values)), axis=1)

test ['Global_context']=test.iloc[:,2:].apply(lambda r: ''.join(str(r.values)), axis=1)

train=train.drop([ 'Date', 'Top1', 'Top2', 'Top3', 'Top4', 'Top5', 'Top6', 'Top7',

       'Top8', 'Top9', 'Top10', 'Top11', 'Top12', 'Top13', 'Top14', 'Top15',

       'Top16', 'Top17', 'Top18', 'Top19', 'Top20', 'Top21', 'Top22', 'Top23',

       'Top24', 'Top25'],axis=1)

train
test=test.drop([ 'Date', 'Top1', 'Top2', 'Top3', 'Top4', 'Top5', 'Top6', 'Top7',

       'Top8', 'Top9', 'Top10', 'Top11', 'Top12', 'Top13', 'Top14', 'Top15',

       'Top16', 'Top17', 'Top18', 'Top19', 'Top20', 'Top21', 'Top22', 'Top23',

       'Top24', 'Top25'],axis=1)

test
X_train=train['Global_context']

X_test=test['Global_context']

y_train=train['Label']

y_test=test['Label']
#Convert Global_context into word count vectors

from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')

X_train_cv = cv.fit_transform(X_train)

X_test_cv = cv.transform(X_test)
#count frequency of words in Global_context

words_freq = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())

words = pd.DataFrame(words_freq.sum()).sort_values(0, ascending=False)
print(words)
#fit a naive_bayes model and make predictions



from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()

naive_bayes.fit(X_train_cv, y_train)

predictions = naive_bayes.predict(X_test_cv)
#Evaluation metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,confusion_matrix,classification_report

#print('Accuracy score: ', accuracy_score(y_test, predictions))

#print('Precision score: ', precision_score(y_test, predictions))

#print('Recall score: ', recall_score(y_test, predictions))

#print('F-score: ', f1_score(y_test, predictions))

print ('\n confusion matrix:\n',confusion_matrix(y_test, predictions))

print ('\n clasification report:\n', classification_report(y_test,predictions))
#testing our predictions

testing_predictions = []

for i in range(len(X_test)):

    if predictions[i] == 1:

        testing_predictions.append('1')

    else:

        testing_predictions.append('0')

results = pd.DataFrame({'Labels': list(y_test), 'prediction': testing_predictions, 'tops':list(X_test)})

results.replace(to_replace=0, value=0, inplace=True)

results.replace(to_replace=1, value=1, inplace=True)
results 
# we should choose an another ML model