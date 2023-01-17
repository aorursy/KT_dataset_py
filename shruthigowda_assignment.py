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
import numpy as np
## read csv file

import pandas as pd

df= pd.read_csv("../input/New data1.csv")
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
## Number of rows and columns of data

##first 5 rows of data

df.shape

df.head()
## Basic statistical details of data

df.describe()
## column names

df.columns
## Number of distinct observations

df.nunique()
## find missing values in each column

df.isna().sum()
## replace Nan by mean value

tr=df.fillna(df.mean())

tr
tr.isna().sum()
## following columns have no correlation so i removed

tr.drop(['OTHER'],axis=1,inplace=True)

tr.drop(['PREDICTIVE RISK FACTORS'],axis=1,inplace=True)

tr.drop(['Being bullied'],axis=1,inplace=True)

tr.drop(['Being exploited'],axis=1,inplace=True)
tr.head()
## correlation matrix

corr = tr.corr()

corr.style.background_gradient(cmap='coolwarm')
tr.head()

tr.columns
tr.shape
print(tr['Worries about falling or feels unsteady when standing or walking'].value_counts())
feature_columns =['Any fall in past year','Any fall in past years','big risk of being underweight','Bullying of others','Cruelty to animals','Damage to property','Emotional abuse','Exploitation','Exposure to violence','Fire setting','Health risk due to low weight','Homelessness','Hyperactive','Lack of self care','Mental retardation','Neglect','Non-compliance','Physical abuse','Physical harm','Recent significant loss','Reckless/impulsive behaviour (inappropriate to stage of development)','RISK FROM OTHERS','Risk of Suicide','Running away from home at night','School exclusion/non-attendance','Self harm or threats of self harm','Sexual abuse','Sexual harm','Substance misuse','Threatening behaviour','Worries about falling or feels unsteady when standing or walking']

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV

import seaborn as sns
X = tr[feature_columns].values

y = tr['Physical disability'].values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
import math

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_squared_error 

rmse_val = [] #to store rmse values for different k

for K in range(1,20):

    K = K+1

    model = KNeighborsClassifier(n_neighbors = K)



    model.fit(X_train, y_train)  #fit the model

    pred=model.predict(X_test) #make prediction on test set

    error = math.sqrt(mean_squared_error(y_test,pred)) #calculate rmse

    rmse_val.append(error) #store rmse values

print(min(rmse_val))

print(rmse_val)

import matplotlib.pyplot as plt

plt.plot(rmse_val)

print(rmse_val.index(min(rmse_val)))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score



# Instantiate learning model (k = 5)

classifier = KNeighborsClassifier(n_neighbors=5)



# Fitting the model

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

cm

accuracy = accuracy_score(y_test, y_pred)*100

print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')