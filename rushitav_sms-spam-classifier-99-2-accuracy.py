# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

%matplotlib notebook

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

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
data=pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')

data=data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

data['v1']=np.where(data['v1']=='spam',1,0)

data
data.isnull().sum()
plt.figure()

sns.countplot(data['v1'])

plt.show()
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(data['v2'],data['v1'],random_state=0)



from sklearn.feature_extraction.text import CountVectorizer



cv=CountVectorizer()

X_train_cv=cv.fit_transform(X_train)

X_test_cv=cv.transform(X_test)


from sklearn.naive_bayes import MultinomialNB



clf_multi=MultinomialNB(alpha=0.1)

clf_multi.fit(X_train_cv,y_train)

predicted=clf_multi.predict(X_test_cv)

score=accuracy_score(y_test,predicted)

score

from sklearn.linear_model import LogisticRegression



clf=LogisticRegression()

clf.fit(X_train_cv,y_train)

predicted=clf.predict(X_test_cv)

score=accuracy_score(y_test,predicted)

score
from sklearn.svm import SVC



clf=SVC(kernel='linear')

clf.fit(X_train_cv,y_train)

predicted=clf.predict(X_test_cv)

score=accuracy_score(y_test,predicted)

score
from sklearn.ensemble import RandomForestClassifier



clf=RandomForestClassifier(random_state=0)

clf.fit(X_train_cv,y_train)

predicted=clf.predict(X_test_cv)

score=accuracy_score(y_test,predicted)

score