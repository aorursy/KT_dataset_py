# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression  #Import Linear regression model

from sklearn.model_selection import train_test_split  #To split the dataset into Train and test randomly

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/insurance.csv")
df.columns
df.head()
df.describe().T
df.isna().sum()
from scipy.stats import kurtosis, skew, stats
print("Summary Statistics of Medical Costs")

print(df['expenses'].describe())

print("skew:  {}".format(skew(df['expenses'])))

print("kurtosis:  {}".format(kurtosis(df['expenses'])))

print("missing charges values: {}".format(df['expenses'].isnull().sum()))

print("missing smoker values: {}".format(df['smoker'].isnull().sum()))
import matplotlib.pyplot as plt

import seaborn as sns
f, axes = plt.subplots(1, 2)

sns.kdeplot(df['expenses'], ax=axes[0])

sns.boxplot(df['expenses'], ax=axes[1])

plt.show()
#prepare our 2 groups to test

#smoker = df[df['smoker']==1]

#non_smoker = df[df['smoker']==0]

ax = sns.swarmplot(x='smoker',y='expenses',data=df)

ax.set_title("Smoker vs Expenses")

plt.xlabel("Smoker (Yes - 1, No - 0)")

plt.ylabel("Expenses")

plt.show(ax)
#plt.title('Distribution of Medical Costs for Smokers Vs Non-Smokers')

#ax = sns.kdeplot(smoker['expenses'], bw=10000, label='smoker')

#ax = sns.kdeplot(non_smoker['expenses'], bw=10000, label='non-smoker')

#plt.show()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

df.iloc[:,4] = labelencoder.fit_transform(df.iloc[:,4])
df.head()
df.corr()
x = df[['age','bmi','smoker']]

y = df['expenses']

#train_test_split() to split the dataset into train and test set at random.

#test size data set should be 30% data

X_train,X_test,Y_train, Y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#Creating an linear regression model object

model = LinearRegression()

model.fit(X_train, Y_train) 
print("Intercept value:", model.intercept_)

print("Coefficient values:", model.coef_)
coef_df = pd.DataFrame(list(zip(X_train.columns,model.coef_)), columns = ['Features','Predicted Coeff'])

coef_df

Y_train_predict = model.predict(X_train)

Y_test_predict = model.predict(X_test)
ax = sns.scatterplot(Y_train,Y_train_predict)

ax.set_title("Actual Expenses vs Predicted Expenses")

plt.xlabel("Actual Expenses")

plt.ylabel("Predicted Expenses")

plt.show(ax)
print("MAE")

print("train : ",mean_absolute_error(Y_train,Y_train_predict))

print("test : ",mean_absolute_error(Y_test,Y_test_predict))
print("MSE")

print("train : ",mean_squared_error(Y_train,Y_train_predict))

print("test : ",mean_squared_error(Y_test,Y_test_predict))
print("Rsquare")

print("train : ",r2_score(Y_train,Y_train_predict))

print("test : ",r2_score(Y_test,Y_test_predict))
smoker_model = LinearRegression()

smoker_model.fit(X_train[['smoker']], Y_train)

print("intercept:",smoker_model.intercept_, "coeff:", smoker_model.coef_)



#print("Train - Mean squared error:", np.mean((Y_train - model.predict(X_train)) ** 2))

smoker_df = pd.DataFrame(list(zip(Y_train, smoker_model.predict(X_train[['smoker']]))), columns = ['Actual Expenses','Predicted Expenses'])

smoker_df.head()

#X_train['smoker'].shape
print("MSE:",np.sqrt(mean_squared_error(Y_train, Y_train_predict)))

print("MSE only for Smoker:", np.sqrt(mean_squared_error(Y_train,smoker_model.predict(X_train[['smoker']]))))
#R-Squared value for Train data set

print("R-squared value:",round(r2_score(Y_train, Y_train_predict),3))

print("R-squared value only for smoker:", round(r2_score(Y_train,smoker_model.predict(X_train[['smoker']]))),3)
#Mean absolute error for Train data set

print("Mean absolute error:",mean_absolute_error(Y_train, Y_train_predict))

print("Mean absolute Error only for Smoker:", mean_absolute_error(Y_train,smoker_model.predict(X_train[['smoker']])))