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
data = pd.read_csv("/kaggle/input/performance-prediction/summary.csv")

data.head()
#check for missing values

data.isna().sum()
#fill in the missing values with mode

data["3PointPercent"].fillna(data["3PointPercent"].mean(),inplace=True)

data.isna().sum()
#perform EDA to check linear relationships 

import matplotlib.pyplot as plt

import seaborn as sns

X=data.iloc[:,:20]

y=data["Target"]

#check for multi-collinearity

corr=X.corr()

corr.style.background_gradient(cmap='coolwarm')
from sklearn.decomposition import PCA

pca = PCA().fit(X.iloc[:,1:])

plt.grid(True)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');
import sklearn.metrics

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,shuffle=True)
pca = PCA(n_components=5)

X_train_pca=X_train.drop('Name',axis=1)

X_test_pca=X_test.drop('Name',axis=1)

train_pca = pca.fit_transform(X_train_pca)

test_pca = pca.fit_transform(X_test_pca)

train_pca.shape,test_pca.shape
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score,classification_report

lr = LogisticRegression(max_iter=100,random_state=10)

lr.fit(train_pca,y_train)

lr_predictions = lr.predict(test_pca)

score = accuracy_score(y_test,lr_predictions) 

print("This is the accuracy of our Logistic Regression Model is",round(score*100))

rep = classification_report(y_test,lr_predictions)

print("This is the Classification-Report : \n", rep)
#Support-Vector Machine

svc = SVC(random_state=10,C=2.1)

svc.fit(train_pca,y_train)

svc_predictions = svc.predict(test_pca)

score = accuracy_score(y_test,svc_predictions) 

print("This is the accuracy of our SVC Model is :",round(score*100))

rep = classification_report(y_test,svc_predictions)

print("This is the Classification-Report : \n", rep)
nb = GaussianNB()

nb.fit(train_pca,y_train)

nb_predictions = nb.predict(test_pca)

score = accuracy_score(y_test,nb_predictions) 

print("This is the accuracy of our Naive Bayes Model is:",round(score*100))

rep = classification_report(y_test,nb_predictions)

print("This is the Classification-Report : \n", rep)
ld = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')

ld.fit(train_pca,y_train)

ld_predictions = ld.predict(test_pca)

score = accuracy_score(y_test,ld_predictions) 

print("This is the accuracy of our LDA Model is:",round(score*100))

rep = classification_report(y_test,ld_predictions)

print("This is the Classification-Report : \n", rep)
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'Names':X_test['Name'],'Target':svc_predictions})



#Visualize the first 5 rows

submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'BasketBall Player Prediction.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)