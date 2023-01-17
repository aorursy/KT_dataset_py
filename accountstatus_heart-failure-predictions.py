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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegressionCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC
data=pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data.head()
corr=data.corr()

sns.heatmap(corr)
# Target variable = Death Event
# We have 12 features in total let's find out the 8 best features for our model
X=data.drop(columns=['DEATH_EVENT'])

y=data['DEATH_EVENT']
# Code to select the 8 best features for the dataset with the help of chi2

bestfeatures = SelectKBest(score_func=chi2, k=8)

fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(8,'Score')) 
data.drop(columns=['diabetes','sex','smoking','anaemia'],axis=1,inplace=True)
X=data.drop(columns=['DEATH_EVENT']).values
y=data['DEATH_EVENT'].values
scaler = StandardScaler()
X=scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
lg=LogisticRegression()
lg.fit(X_train,y_train)
y_hat=lg.predict(X_test)
print('The accuracy score of logistic regression is : ',accuracy_score(y_hat,y_test))
# Let's do 10 k folds validation set and check the accuracy score

for i in range(2,11):

    cv=LogisticRegressionCV(cv=i,random_state=True,scoring='accuracy').fit(X,y)

    print(cv.score(X,y))
# we see that the score is higher when the k folds value is equal to 2,3,4....

# we will take as 2 in that condition
cv=LogisticRegressionCV(cv=2,random_state=True,scoring='accuracy').fit(X_train,y_train)

y_hat_cv=cv.predict(X_test)

print('The accuracy score with the CV is :',accuracy_score(y_hat_cv,y_test))
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_hat_dt=dt.predict(X_test)
print('The accuracy score of the decision tree classifier is: ',accuracy_score(y_hat_dt,y_test))
# Decision Tree classifier less score than the Logistic Regression
# let's take 10 nearest neighbors and have a look at the score
wss=[]

for i in range(1,11):

    kn=KNeighborsClassifier(n_neighbors=i)

    kn.fit(X_train,y_train)

    y_hat_kn=kn.predict(X_test)

    wss.append(accuracy_score(y_hat_kn,y_test))

    
plt.plot([int(x) for x in range(1,11)],wss)
# We can see the highest accuracy at n=7
kn=KNeighborsClassifier(n_neighbors=7)

kn.fit(X_train,y_train)

y_hat_kn=kn.predict(X_test)

print('The accuracy of the K-Nearest Model is : ',accuracy_score(y_hat_kn,y_test))
sv=LinearSVC(max_iter=10000)

sv.fit(X_train,y_train)

y_hat_sv=sv.predict(X_test)

print('The accuracy score using the SVM model is : ',accuracy_score(y_hat_sv,y_test))
# Well the model has the same accuracy as the Logistic regression model