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

import matplotlib.pyplot as plt     #visualisation

import seaborn as sns               #visualisation



sns.set(color_codes=True)
# Loading the data into the data frame 

df = pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')

df.head(5)
#Find any null values are present

df.isnull().mean()*100
#Correlation

c = df.corr()

c



#Heatmap

sns.heatmap(c, cmap='BrBG', annot=True)
#Total Deaths and Survivers

sns.countplot(df['Survived'])
# Male and Female death count

sns.countplot(df['Sex'],hue=df['Survived'])
# Passenger and Crew death count

sns.countplot(df['Category'],hue=df['Survived'])
# Male passengers & crew vs Female passengers & crew

sns.countplot(df['Sex'],hue=df['Category'])
# According to Country wise

country = sns.countplot(df['Country'],hue=df['Survived'])

country.set_xticklabels(country.get_xticklabels(), rotation=45)
# Dropping unneccessary columns

df= df.drop(['PassengerId','Country','Firstname','Lastname'],axis=1)

df.head()
# Converting Catergorial variables into numerical and adding dummies

cat_var=['Sex','Category']



for var in cat_var:

    dummy=pd.get_dummies(df[var],prefix=var,drop_first=True)

    

    df=pd.concat([df,dummy],axis=1)

    df=df.drop([var],axis=1)

df.head()
# Train and Test Split

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

np.random.seed()
#Train and Test set

df_train,df_test = train_test_split(df,train_size=0.7,test_size=0.3,random_state=100)

y_train=df_train.pop('Survived')

X_train=df_train



y_test=df_test.pop('Survived')

X_test=df_test
# Random Forest

from sklearn.ensemble import RandomForestClassifier



# Create instance

rf_classifier = RandomForestClassifier(criterion='entropy',n_jobs=-1,n_estimators=10,

                                       random_state=None,max_depth=5)



# Fitting the model

rf_classifier.fit(X_train,y_train)
# Predicting the values

y_pred_train = rf_classifier.predict(X_train)

y_pred_test = rf_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score



# Confusion matrix Train

cm_train = confusion_matrix(y_train, y_pred_train)

print(cm_train)
# Confusion matrix Test

cm_test = confusion_matrix(y_test, y_pred_test)

print(cm_test)
# Accuracy train

accuracy_train = accuracy_score(y_train, y_pred_train)

accuracy_train
# Accuracy Test

accuracy_test = accuracy_score(y_test, y_pred_test)

accuracy_test
# Grid Search -- Cross-Validation 

from sklearn.model_selection import GridSearchCV



rf_classifier = RandomForestClassifier(criterion='entropy',n_jobs=-1,n_estimators=10,

                                       random_state=None,max_depth=5)



#Parameters Grid

params_grid = {'min_samples_split':[2,3,4]}
# Creating instance

grid_classifier = GridSearchCV(rf_classifier,params_grid,scoring='accuracy',cv=5)



# Fitting the model

grid_classifier.fit(X_train,y_train)

#best parameters

grid_classifier.best_params_
#Cross validation -- best estimator



cvrf_classifier = grid_classifier.best_estimator_
# Predict

y_pred_train = cvrf_classifier.predict(X_train)

y_pred_test = cvrf_classifier.predict(X_test)
# Confusion Matrix

cm_train = confusion_matrix(y_train, y_pred_train)

print(cm_train)



cm_test = confusion_matrix(y_test, y_pred_test)

print(cm_test)
# Accuracy

accuracy_train = accuracy_score(y_train, y_pred_train)

print(accuracy_train)



accuracy_test = accuracy_score(y_test, y_pred_test)

print(accuracy_test)