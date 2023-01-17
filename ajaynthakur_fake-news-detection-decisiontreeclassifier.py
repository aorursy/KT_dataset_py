# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Read the data From Dataset
fake_news = pd.read_csv("/kaggle/input/fake-news-detection/data.csv")
fake_news.head()
#remove the URL column
fake_news=  fake_news.drop(['URLs'], axis=1)
fake_news = fake_news.dropna()
fake_news.head()
#Take only 3500 News Headlines
fake_news=fake_news[0:3500]
X = fake_news.iloc[:,:-1].values
y = fake_news.iloc[:,-1].values
X[0]
y[0]
#Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
News_body = cv.fit_transform(X[:,1]).todense()

News_body
cv_head = CountVectorizer(max_features=5000)
News_head = cv_head.fit_transform(X[:,0]).todense()
News_head
X_mat = np.hstack(( News_head, News_body))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_mat,y, test_size=0.2, random_state=0)
#Here we are using DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train)
y_pred=dtc.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
#Calculate the Accuracy
Accuracy =(98+87)/(98+87+8+7)
Accuracy
