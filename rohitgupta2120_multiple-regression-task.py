#Libraries used in the kernel



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # graphs potting 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, classification_report

from statsmodels.api import OLS



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataframe = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv", index_col="sl_no")

dataframe.head()
#Make copies of dataframe

data_reg = dataframe.copy()

data_class = dataframe.copy()
sns.kdeplot(dataframe.mba_p[ dataframe.gender=="M"])

sns.kdeplot(dataframe.mba_p[ dataframe.gender=="F"])

plt.legend(["Male", "Female"])

plt.xlabel("mba percentage")

plt.show()
matrix = dataframe.corr()

plt.figure(figsize=(8,6))

#plot heat map

g=sns.heatmap(matrix,annot=True,cmap="YlGn_r")
plt.figure(figsize=(12,8))

sns.regplot(x="ssc_p",y="mba_p",data=dataframe)

sns.regplot(x="hsc_p",y="mba_p",data=dataframe)

plt.legend(["ssc percentage", "hsc percentage"])

plt.ylabel("mba percentage")

plt.show()
# Seperating independent and dependent variables

#dependent variables ssc_p, hsc_p

X = data_class.iloc[:,[1,3]].values

y = data_class.iloc[:,-3].values.reshape(-1,1)
#splitting into training and test set

#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)
#Multiple linear regression

#import library

#from sklearn.linear_model import LinearRegression

regressor = LinearRegression()



#train the model

regressor.fit(X_train, y_train)



#predict the test set(mba_p)

y_pred_m = regressor.predict(X_test)
#from sklearn.metrics import r2_score, classification_report

print("R2 score: " + str(r2_score(y_test, y_pred_m)))
print(regressor.coef_)

print(regressor.intercept_)
#from statsmodels.api import OLS

summ=OLS(y_train,X_train).fit()

summ.summary()
# Seperating independent and dependent variables

#dependent variables ssc_p, degree_p

X = data_class.iloc[:,[1,6]].values

y = data_class.iloc[:,-3].values.reshape(-1,1)



#splitting into training and test set

#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)



#Multiple linear regression

#from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



#predict the dependent variable

y_pred_m = regressor.predict(X_test)



#from sklearn.metrics import r2_score, classification_report

print("R2 score: " + str(r2_score(y_test, y_pred_m)))

print("regression coeff: " + str(regressor.coef_))

print("regression intercept: " + str(regressor.intercept_))

print("mba_p = 0.12 x ssc_p + 0.22 x degree_p + 39.66")
# Seperating independent and dependent variables

X = data_class.iloc[:,[3,6]].values

y = data_class.iloc[:,-3].values.reshape(-1,1)



#splitting into training and test set

#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)



#Multiple linear regression

#from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred_m = regressor.predict(X_test)



#from sklearn.metrics import r2_score, classification_report

print("R2 score:" + str(r2_score(y_test, y_pred_m)))

print("regression coeff:" + str(regressor.coef_))

print("regression intercept:" + str(regressor.intercept_))

print("mba_p = " + str(regressor.coef_[0][0]) + " x hsc_p + " + str(regressor.coef_[0][1]) + " x degree_p + " + str(regressor.intercept_[0]))
# Seperating independent and dependent variables

X = data_class.iloc[:,[1,3,6]].values

y = data_class.iloc[:,-3].values.reshape(-1,1)



#splitting into training and test set

#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=1001)



#Multiple linear regression

#from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
#Summary of the model

#from statsmodels.api import OLS

summ=OLS(y_train,X_train).fit()

summ.summary()
# Seperating independent and dependent variables

X = data_class.iloc[:,[3,6]].values

y = data_class.iloc[:,-3].values.reshape(-1,1)



#splitting into training and test set

#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=1001)



#Multiple linear regression

#from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
#predict the values

y_pred_m = regressor.predict(X_test)
#Summary of the model

#from statsmodels.api import OLS

summ=OLS(y_train,X_train).fit()

summ.summary()
#from sklearn.metrics import r2_score, classification_report

#R2 score

print("R2 score:" + str(r2_score(y_test, y_pred_m)))



#model p values

print("regression coeff:" + str(regressor.coef_))

print("regression intercept:" + str(regressor.intercept_))

print("mba_p = " + str(regressor.coef_[0][0]) + " x hsc_p + " + str(regressor.coef_[0][1]) + " x degree_p + " + str(regressor.intercept_[0]))
np.set_printoptions(precision=2)

dff = pd.DataFrame(list(zip(y_test, y_pred_m.round(2))),columns=("Target","Predicted"))

dff.head(8)