# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_json('../input/train.json', orient='columns')
df.head()
df['IngredientString'] = df['ingredients'].astype('str')
df['IngredientString'] = df['IngredientString'].str.strip('[').str.strip(']').str.replace(',',' ').str.replace(' ','').str.replace('\'',' ')
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#count_vectorizer = CountVectorizer(binary='true')
tfidf = TfidfVectorizer(binary=True)
train_documents = [line.lower() for line in df['IngredientString']]
train_documents = tfidf.fit_transform(train_documents)
train_labels = [line for line in df['cuisine']]
train_documents
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(train_labels)
y
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_documents, y, test_size = 0.05, random_state = 0)
X_train.shape
print(pd.DataFrame(train_labels).describe())
y_train
y_train.shape
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
classifier = SVC(C=100, 
kernel='rbf', # kernel type, rbf working fine here
 degree=3, # default value
 gamma=1, # kernel coefficient
 coef0=1, # change to 1 from default value of 0.0
 shrinking=True, # using shrinking heuristics
 tol=0.001, # stopping criterion tolerance 
 probability=False, # no need to enable probability estimates
 cache_size=200, # 200 MB cache size
 class_weight=None, # all classes are treated equally 
 verbose=True, # print the logs 
 max_iter=-1, # no limit, let it run
 decision_function_shape=None, # will use one vs rest explicitly 
 random_state=0)
model = OneVsRestClassifier(classifier)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB().fit(X_train,y_train)

#y_pred = classifier.predict(count_vectorizer.transform(["romainelettuce  blackolives  grapetomatoes  garlic  pepper  purpleonion  seasoning  garbanzobeans  fetacheesecrumbles"]))
label = labelencoder_y.inverse_transform(y_pred)
label
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
import seaborn as sn
plt.figure(figsize = (20,7))
sn.heatmap(cm, annot=True)
df_test = pd.read_json('../input/test.json', orient='columns')
df_test['IngredientString'] = df_test['ingredients'].astype('str')
df_test['IngredientString'] = df_test['IngredientString'].str.strip('[').str.strip(']').str.replace(',',' ').str.replace(' ','').str.replace('\'',' ') 

test_documents = [line.lower() for line in df_test['IngredientString']]
test_documents = tfidf.transform(test_documents)
test_documents
y_test_submission = model.predict(test_documents)
y_test_submission
test_labels = labelencoder_y.inverse_transform(y_test_submission)
test_labels
sub = pd.read_csv('../input/sample_submission.csv')
sub['cuisine'] = ''
sub['cuisine'] = test_labels[sub.index.values]
sub.head()
sub.to_csv('submission_svc.csv',index = False)