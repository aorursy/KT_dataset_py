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
import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 

# For feature Selection

from sklearn.feature_selection import f_regression

from sklearn.feature_selection import SelectKBest

# For Machine Learning

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import  LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
#import Data

df=pd.read_csv('/kaggle/input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv')

#see first 5 rows

df.head()

#see numbers of rows and columns

df.shape

#see data types of columns

df.info()
df.drop(columns=['objid','specobjid'],inplace=True)
df.head()


sns.countplot(df['camcol'])
#see which is the most class 

sns.countplot(df['class'])
# see if there are missing values

df.isna().sum()
#change  category data into nummerical 

def change_category (cat):

    if cat=='STAR':

        return 0

    elif cat == 'GALAXY':

        return 1 

    else :

        return 2

    
df['ClassCat']=df['class'].apply(change_category)
df.head()
sns.pairplot(df[['u','g','r','i']])
df.drop(columns='class',inplace=True)
X=df.drop(columns='ClassCat')
y=df['ClassCat']


best_feature = SelectKBest(score_func=f_regression,k='all')

fit = best_feature.fit(X,y)
score = pd.DataFrame(fit.scores_)

columns = pd.DataFrame(X.columns)

featureScores = pd.concat([columns,score],axis=1)

featureScores.columns = ['Feature','Score']

featureScores = featureScores.sort_values(by='Score',ascending=False).reset_index(drop=True)



featureScores
X= X[featureScores.Feature[:8].values]

X=StandardScaler().fit_transform(X)
X_train ,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


logreg= LogisticRegression()

logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

log_reg_acc=logreg.score(X_test,y_test)

print(' Logisitic Regression score {}'.format(log_reg_acc))

print(' Root Mean Squared Error  {}'.format(np.sqrt(mean_squared_error(y_test,y_pred))))







knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

y_pred1=knn.predict(X_test)

knn_acc=knn.score(X_test,y_test)



print(' KNN score {}'.format(knn_acc))

print(' Root Mean Squared Error  {}'.format(np.sqrt(mean_squared_error(y_test,y_pred1))))







dt=DecisionTreeClassifier(max_leaf_nodes=20,random_state=0)

dt.fit(X_train,y_train)

y_pred2=dt.predict(X_test)

dt_score=dt.score(X_test,y_test)



print(' Decision Tree   score {}'.format(dt_score))

print(' Root Mean Squared Error  {}'.format(np.sqrt(mean_squared_error(y_test,y_pred2))))



rf=RandomForestClassifier(n_estimators=120)

rf.fit(X_train,y_train)

y_pred3=rf.predict(X_test)

rf_acc=rf.score(X_test,y_test)

print(' Random Forest score {}'.format(rf_acc))

print(' Root Mean Squared Error  {}'.format(np.sqrt(mean_squared_error(y_test,y_pred3))))


linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

y_pred4 = linear_svc.predict(X_test)

linear_svc_acc=linear_svc.score(X_test,y_test)

print(' Random Forest score {}'.format(linear_svc_acc))

print(' Root Mean Squared Error  {}'.format(np.sqrt(mean_squared_error(y_test,y_pred4))))

models = pd.DataFrame({

    'Model': [ 'KNN', 'Logistic Regression', 

              'Random Forest' ,'Linear SVC', 

              'Decision Tree'],

    'Score': [knn_acc, log_reg_acc, rf_acc, 

              linear_svc_acc ,dt_score]})

models.sort_values(by='Score', ascending=False)