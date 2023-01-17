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



%matplotlib inline

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

import seaborn as sns

import xgboost as xgb

from sklearn.model_selection import KFold

from IPython.display import HTML, display

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
import pandas_profiling

report = pandas_profiling.ProfileReport(train)



display(report)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train["Cabin"].isna().sum()/train["Cabin"].shape[0]
train["Age"].isna().sum()/train["Age"].shape[0]
train['Cabin'] = train['Cabin'].fillna('Missing')
train['cabin']=train['Cabin'].str.replace('\d+', '')

train.boxplot(column='Age')
train['Age']  = train['Age'].fillna(train['Age'].mean())
train['survivedcat']=train['Survived'].astype('category')
a = pd.crosstab(train['survivedcat'],  

                            train['cabin'], 

                                margins = False) 

from scipy.stats import chi2_contingency 



# defining the table 



stat, p, dof, expected = chi2_contingency(a) 



# interpret p-value 

alpha = 0.05

print("p value is " + str(p)) 

if p <= alpha: 

	print('Dependent (reject H0)') 

else: 

	print('Independent (H0 holds true)') 

b= pd.crosstab(train['survivedcat'],  

                            train['Embarked'], 

                                margins = False) 

from scipy.stats import chi2_contingency 



# defining the table 



stat, p, dof, expected = chi2_contingency(b) 



# interpret p-value 

alpha = 0.05

print("p value is " + str(p)) 

if p <= alpha: 

	print('Dependent (reject H0)') 

else: 

	print('Independent (H0 holds true)') 
c = pd.crosstab(train['survivedcat'],  

                            train['Ticket'], 

                                margins = False) 

from scipy.stats import chi2_contingency 



# defining the table 



stat, p, dof, expected = chi2_contingency(c) 



# interpret p-value 

alpha = 0.05

print("p value is " + str(p)) 

if p <= alpha: 

	print('Dependent (reject H0)') 

else: 

	print('Independent (H0 holds true)') 
d = pd.crosstab(train['survivedcat'],  

                            train['Sex'], 

                                margins = False) 

from scipy.stats import chi2_contingency 



# defining the table 



stat, p, dof, expected = chi2_contingency(d) 



# interpret p-value 

alpha = 0.05

print("p value is " + str(p)) 

if p <= alpha: 

	print('Dependent (reject H0)') 

else: 

	print('Independent (H0 holds true)') 
e = pd.crosstab(train['survivedcat'],  

                            train['Pclass'], 

                                margins = False) 

from scipy.stats import chi2_contingency 



# defining the table 



stat, p, dof, expected = chi2_contingency(e) 



# interpret p-value 

alpha = 0.05

print("p value is " + str(p)) 

if p <= alpha: 

	print('Dependent (reject H0)') 

else: 

	print('Independent (H0 holds true)')
sns.catplot(x="Sex", y="Age", hue="Pclass",

            col="Survived", aspect=.7,

            kind="swarm", data=train)
sns.catplot(y="cabin", hue="Pclass", kind="count",

            palette="pastel", edgecolor=".6",

            data=train)

sns.catplot(y="Sex", hue="Pclass", kind="count",

            palette="pastel", edgecolor=".6",

            data=train)
train.head()
train['Male'] = train['Sex'].map( {'male':1, 'female':0} )

train[['Sex', 'Male']]
train = train.drop(['Cabin','Embarked','Name','Ticket','cabin','survivedcat'], axis = 1)
train=train.drop('Sex', axis=1)
train.dtypes
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop(columns=["Survived"]), train["Survived"], random_state = 42)  
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error, confusion_matrix, accuracy_score, classification_report



features = ["Pclass", "Male", "SibSp", "Parch","Fare", "Age"]



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f''' MAE : {mean_absolute_error(y_test, predictions)}

 Confusion Matrix: 

 {confusion_matrix(y_test, predictions)}

Accuracy:  {accuracy_score(y_test, predictions)}

Classification report:{ classification_report(y_test, predictions)} ''')
test['Age']  = test['Age'].fillna(test['Age'].mean())
test['Male'] = test['Sex'].map( {'male':1, 'female':0} )

test[['Sex', 'Male']]
test = test.drop(['Cabin','Embarked','Name','Ticket','Sex'], axis = 1)
test.isnull().sum()
test.Fare = test.Fare.fillna(0)
submit = model.predict(test)
test.head()
from xgboost import XGBRegressor

my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.05)



# Fit the model

my_model_2.fit(X_train, y_train)



# Get predictions

predictions_2 = my_model_2.predict(X_test)



# Calculate MAE

mae_2 = mean_absolute_error(predictions_2, y_test)

print("Mean Absolute Error:" , mae_2)



# make predictions which we will submit. 

test_preds = my_model_2.predict(test)
from sklearn import linear_model

from sklearn import preprocessing

# Initialize logistic regression model

log_model = linear_model.LogisticRegression(solver = 'lbfgs')



# Train the model

# Fit the model

log_model.fit(X_train, y_train)



# Get predictions

predictions_2 = log_model.predict(X_test)



# Calculate MAE

mae_2 = mean_absolute_error(predictions_2, y_test)

print("Mean Absolute Error:" , mae_2)

# Save test predictions to file

output = pd.DataFrame({'PassengerId': test.PassengerId,

                       'Survived': submit})

output.to_csv('submission.csv', index=False)