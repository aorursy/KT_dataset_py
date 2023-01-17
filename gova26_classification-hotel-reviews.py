# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing dataset:

a=pd.read_csv("../input/reviews.csv")

print(a.head())
#filling null values numeric:

med=a["Review"].median()

med

b=a["Review"]

b.fillna(value=med,inplace=True)

a["Review"].head()

print(a.head())
#filling categorical values:

g=a["Recommends"]

count_true=0

count_false=0

for i in range(len(g)):

    if(g[i]==True):

        count_true=count_true+1

    elif(g[i]==False):

        count_false=count_false+1

        

print(count_true,count_false)

#true occurs most of the times hence we fill the same value as true:

n="True"

a["Recommends"]=a["Recommends"].fillna(value="True")

print(a.head())

print(a.isnull().values.any())
#converting categorical variable recommends:

c=a

c["Recommends"]=pd.get_dummies(data=a["Recommends"])

print(c.head())
#building training and testing set:

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(c["Review Text"],c["Recommends"],test_size=0.3,random_state=12)
#building a vector model:

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

train_vectors = vectorizer.fit_transform(x_train)

test_vectors = vectorizer.transform(x_test)

print(train_vectors.shape, test_vectors.shape)
#logistic regression model:

from sklearn.linear_model import LogisticRegression

lm=LogisticRegression()

model=lm.fit(train_vectors,y_train)

pred=model.predict(test_vectors)

from sklearn.metrics import classification_report

print(classification_report(y_true=y_test,y_pred=pred))

from sklearn.metrics import accuracy_score

accuracy_score(y_true=y_test,y_pred=pred)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_true=y_test,y_pred=pred)

accuracy_score(y_test,pred)

pre=model.predict(train_vectors)

accuracy_score(y_train,pre)
#svm model:

from sklearn.svm import LinearSVC

sv=LinearSVC()

svv=sv.fit(train_vectors,y_train)

predd=svv.predict(test_vectors)

predd

print(classification_report(y_true=y_test,y_pred=predd))

confusion_matrix(y_test,predd)

accuracy_score(y_test,predd)