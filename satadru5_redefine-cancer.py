# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings



# Any results you write to the current directory are saved as output.
training_variants_df = pd.read_csv("../input/training_variants")

training_text_df = pd.read_csv("../input/training_text",sep="\|\|", 

engine='python', header=None, skiprows=1, names=["ID","Text"])
#pd.set_option('display.max_columns', None)

#pd.set_option('display.max_colwidth', -1)

training_text_df.head(2)
training_variants_df.head(5)
training_merge_df = training_variants_df.merge(training_text_df,left_on="ID",right_on="ID")
training_merge_df.head(2)
testing_variants_df = pd.read_csv("../input/test_variants")

testing_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

testing_merge_df = testing_variants_df.merge(testing_text_df,left_on="ID",right_on="ID")
sns.countplot(training_merge_df['Class'])
import missingno as msno

%matplotlib inline

msno.bar(training_merge_df)
training_merge_df.isnull().sum()
msno.bar(testing_merge_df)
from sklearn.model_selection import train_test_split



train ,test = train_test_split(training_merge_df,test_size=0.2,random_state=100) 

np.random.seed(0)

pd.set_option('display.max_colwidth', 50)

train.head(2)
X_train = train['Text'].values

X_test = test['Text'].values

y_train = train['Class'].values

y_test = test['Class'].values
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

vect = CountVectorizer()

vect.fit(X_train)
X_train_df = vect.transform(X_train)

X_test_df = vect.transform(X_test)
prediction = dict()

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train_df,y_train)

prediction["Multinomial"] = model.predict(X_test_df)

accuracy_score(y_test,prediction["Multinomial"])
#prediction = dict()

#from sklearn.naive_bayes import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train_df,y_train)

prediction["random_forest"] = model.predict(X_test_df)

accuracy_score(y_test,prediction["random_forest"])
X_test_final = testing_merge_df['Text'].values
X_final = vect.transform(X_test_final)
predicted_class=model.predict(X_final)
testing_merge_df['predicted_class'] = predicted_class
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()

model.fit(X_train_df,y_train)

prediction["adaboost"] = model.predict(X_test_df)

accuracy_score(y_test,prediction["adaboost"])
#from sklearn.neural_network import MLPClassifier

#clf = MLPClassifier(hidden_layer_sizes=(500,500))

#clf.fit(X_train_df,y_train)

#prediction["NN"] = clf.predict(X_test_df)

#accuracy_score(y_test,prediction["NN"])
X_test_final = testing_merge_df['Text'].values

vect.fit(X_test_final)

X_final = vect.transform(X_test_final)

predicted_class = clf.predict(X_final)
testing_merge_df['predicted_class'] = predicted_class
testing_merge_df.head()
onehot = pd.get_dummies(testing_merge_df['predicted_class'])

testing_merge_df = testing_merge_df.join(onehot)
text_clf = Pipeline([('vect', CountVectorizer()),

                     ('tfidf', TfidfTransformer()),

                     ('clf', MultinomialNB())

])

text_clf = text_clf.fit(X_train,y_train)
prediction["Multinomial"] = text_clf.predict(X_test)

accuracy_score(y_test,prediction["Multinomial"])

#y_test_predicted = text_clf.predict(X_test)
text_clf = Pipeline([('vect', CountVectorizer()),

                     ('tfidf', TfidfTransformer()),

                     ('clf', svm.LinearSVC())

])

text_clf = text_clf.fit(X_train,y_train)