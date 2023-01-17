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
# Imorting libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Importing and combining datasets
df1 = pd.read_csv("../input/student-mat.csv")
df2 = pd.read_csv("../input/student-por.csv")
df3 = pd.concat([df1,df2])
df3.head()
# Data Preprocessing and Exploratory analysis
df3=df3.drop_duplicates(["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])
df3.columns
df3.describe()
df3.corr()
df3.info()
#Drop the columns which is not essentials for grade prediction
df3 = df3.drop(['famsize', 'Pstatus', 'Fjob', 'Mjob'])
df3 = df3.drop(['reason','traveltime', 'studytime', 'failures'])
df3 = df3.drop(['schoolsup','famsup', 'paid', 'nursery', 'internet', 'freetime'])
df3 = df3.drop(['higher', 'health'])
df3.columns
#Some visualizations
plt.pie(df3['sex'].value_counts().tolist(), 
        labels=['Female', 'Male'], colors=['red', 'green'], 
        autopct='%1.1f%%', startangle=90)
axis = plt.axis('equal')
plt.pie(df3['guardian'].value_counts().tolist(), 
        labels=['mother', 'father', 'other'], colors=['red', 'green','blue'], 
        autopct='%1.1f%%', startangle=90)
axis = plt.axis('equal')
plt.subplot(2, 1, 1)
plt.hist(df3['Walc'], bins=10)
plt.xlabel('Bottle')
plt.title('Walc')
plt.subplot(2, 1, 2)
plt.hist(df3['Dalc'], bins=10)
plt.xlabel('Bottle')
plt.title('Dalc')
plt.show()
fig, ax = plt.subplots(figsize=(5, 4))
sns.distplot(df3['age'],  
             hist_kws={"alpha": 1, "color": "blue"}, 
             kde=False, bins=8)
ax= ax.set(ylabel="Count", xlabel="Age")
#Given the high correlation between different grades, drop G1 & G2
df3 = df3.drop(['G1', 'G2'])
#combine weekdays alcohol consumption with weekend alcohol consumption
df3['Dalc'] = df3['Dalc'] + df3['Walc']
#combine mother's education with father's education & call it parent's education
df3['Pedu'] = df3['Medu'] + df3['Fedu']
# combine goout and absences
df3['goout'] = df3['goout'] + df3['absences']
df3 = df3.drop(['Walc','Medu','Fedu','absences'])
df3.columns
#Getting dummies
df3 = pd.get_dummies(df3, drop_first=True)
df3.info()

# define target variable and training and test sets
X = df3.drop("G3",axis=1)
Y = df3["G3"]
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
# Building Optimal Model using Backward Elimination
import statsmodels.formula.api as sm
X_opt = X
regressor_OLS = sm.OLS(endog =Y, exog = X_opt).fit()
regressor_OLS.summary()
#Backward Eliminiation Process
#Drop the variable which is not significant(p>0.05)
X_opt = X.drop(['goout','activities_yes', 'address_U', 'school_MS', 'sex_M', 'guardian_mother'], axis=1)
regressor_OLS = sm.OLS(endog =Y, exog = X_opt).fit()
regressor_OLS.summary()

