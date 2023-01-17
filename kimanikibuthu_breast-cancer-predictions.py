# Importing the necessary libraries

import numpy as np

import pandas as pd

import pandas_profiling as pp

import matplotlib.pyplot as plt

import seaborn as sns







# Model development libraries

from sklearn.linear_model import BayesianRidge

from fancyimpute import IterativeImputer as MICE

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



%matplotlib inline
breast = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
breast.head()
breast.describe(include='all')
# Check for missing values



breast.isnull().sum()
# From this we can see that it is an imbalanced dataset, however not so much



breast['diagnosis'].value_counts(normalize= True)*100
#pp.ProfileReport(breast)
breast.drop('Unnamed: 32',axis = 1,  inplace = True)
breast.head()
breast['diagnosis'] = breast['diagnosis'].replace('M', 1)

breast['diagnosis'] = breast['diagnosis'].replace('B', 0)
breast.head()
## check the correlation between the variables in order to know which ones to get rid off. Get rid off the ones with less correlation



x = breast.iloc[:,:]

y = breast.iloc[:, 0]

corrmat = breast.corr()

top_features = corrmat.index

plt.figure(figsize=(20,10))

matrix =sns.heatmap(breast[top_features].corr(),annot=True,cmap="RdYlGn")
# Dropping several columns



breast.drop(['area_mean','area_se', 'perimeter_worst', 'concavity_mean','id','radius_worst' ,'area_worst','concave points_worst', 'radius_mean', 'radius_se', 'perimeter_worst','area_mean','texture_mean'], axis = 1 , inplace = True)
#pp.ProfileReport(breast)
# Histogram

breast.hist(figsize = (20,20))

plt.show()
x = breast.drop('diagnosis', axis = 1)
y = breast['diagnosis']
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size = 0.25 , random_state = 7)
# Logistic regression

lr = LogisticRegression()



lr.fit(x_train,y_train)



lr_pred = lr.predict(x_test)



print(classification_report(y_test, lr_pred))



# Random Forest



rf = RandomForestClassifier(n_estimators = 100)



rf.fit(x_train,y_train)



rf_pred = rf.predict(x_test)



print(classification_report(y_test, rf_pred))