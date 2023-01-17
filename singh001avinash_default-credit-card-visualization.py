# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
import pandas_profiling as pds
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Load  and reading Credit Default datasets
cred_df = pd.read_csv("../input/attachment_default.csv")

cred_df.head()
detail_report= pds.ProfileReport(cred_df)
detail_report.to_file("default_card.html")

cred_df.info()
sns.heatmap(cred_df.corr(),annot=True,cmap= "YlGnBu")
# Relation between balance and default
sns.boxplot(x='default', y='balance', data=cred_df,palette="Set2")
#sns.catplot(x='balance', y='income',col='student',data=cred_df, kind= "box")
plt.show()
# Relation between income and default
sns.boxplot(x='default', y='income', data=cred_df)
plt.show()
# Relation between balance and income and whether they have defaulted or not 

sns.lmplot(x='balance', y='income', hue = 'default', data=cred_df, col='student',aspect=1.5, fit_reg = False)

sns.catplot(x='default', y='income', data=cred_df,hue='default',col='student', kind='boxen')


#plt.figure(figsize=(6,8))
#g=sns.FacetGrid(cred_df, row='balance',col='income')
#g=g.map(plt.scatter,"default")

plt.show()
# Relation between Student and default value representation

pd.crosstab(cred_df['default'], cred_df['student'], rownames=['Default'], colnames=['Student'])
# Convert Categorical to Numerical for default column

default_dummies = pd.get_dummies(cred_df.default, prefix='default', drop_first= True)
cred_df = pd.concat([cred_df, default_dummies], axis=1)

cred_df.head()
# Convert Categorical to Numerical for student column

student_dummies = pd.get_dummies(cred_df.student, prefix='student', drop_first= True)
cred_df = pd.concat([cred_df, student_dummies], axis=1)
cred_df.head()
# Removing repeat columns
cred_df.drop(['default', 'student'], axis=1, inplace=True)
# Try simple linear regression on the data between balance and default

sns.lmplot(x='balance', y='default_Yes', data=cred_df, aspect=1.5, fit_reg = True)
# Building Linear Regression Model and determining the coefficients
from sklearn.linear_model import LinearRegression

x= cred_df[['balance']]
y= cred_df[['default_Yes']]

linreg= LinearRegression()
linreg.fit(x,y)
print(linreg.coef_)
print(linreg.intercept_)
# Building the Logistic Regression Model

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1e42)                            # Set Large C value for low regularization
logreg.fit(x, y)

print(logreg.coef_)                                            # Coefficients for Logistic Regression
print(logreg.intercept_)

y_pred = logreg.predict_proba(x) 
plt.scatter(x.values, y_pred[:,0])                             # Visualization
plt.scatter(x.values, y)
plt.show()
cred_df.head()
# splitting the features and labels

X= cred_df.drop('default_Yes', axis=1)
y = cred_df['default_Yes']

# splitting the data into train and test with 70:30 ratio
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train,y_test= train_test_split(X,y, random_state= 123,test_size=0.30)

# calling logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression(C=.1)
# fitting the model with x and y attributes of train data
# in this it is goin to learn the pattern
logreg.fit(x_train,y_train)
# now applying our learnt model on test and also on train data
y_pred_test = logreg.predict(x_test)
y_pred_train = logreg.predict(x_train)
# comparing the metrics of predicted lebel and real label of test data
metrics.accuracy_score(y_test, y_pred_test)
# comparing the metrics of predicted lebel and real label of test data
metrics.accuracy_score(y_train, y_pred_train)
# creating a confusion matrix to understand the classification
conf = metrics.confusion_matrix(y_test, y_pred_test)
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(conf,cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,
            fmt='d')
