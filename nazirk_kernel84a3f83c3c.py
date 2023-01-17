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
import pandas as pd
df = pd.read_csv("../input/insurance.csv")
df.describe()
df.shape
df.isna().sum()
df.duplicated().sum()
df.drop_duplicates(inplace = True)
df.duplicated().sum()
from sklearn.preprocessing import LabelEncoder

labelencode = LabelEncoder()

df.iloc[:,4]= labelencode.fit_transform(df.iloc[:,4])
df.head()
df.describe()
import matplotlib.pyplot as plt

import seaborn as sns

age_outlier = sns.boxplot(df['age'])

age_outlier.set_title('Dispersion of Age')

plt.show(age_outlier)
bmi_outlier = sns.boxplot(df['bmi'])

bmi_outlier.set_title('Dispersion of BMI')

plt.show(age_outlier)
exp_outlier = sns.boxplot(df['expenses'])

exp_outlier.set_title("Dispersion of Expenses")

plt.show(exp_outlier)
age_bmi_scatter = sns.scatterplot(x='age', y='bmi', data = df)

age_bmi_scatter.set_title('Age and BMI')

plt.show(age_bmi_scatter)
plt.figure(figsize=(15,10))

ax = sns.scatterplot(x='age',y='expenses',hue = 'bmi',size = 'bmi', data=df)

ax = ax.set_title("Age vs Expenses by BMI")

plt.xlabel("Age")

plt.ylabel("Expenses")

plt.show(ax)
plt.figure(figsize=(10,7))

ax = sns.scatterplot(x='age',y='expenses', hue='sex',style = 'sex',data=df)

ax.set_title("Age vs Expenses by Sex")

plt.show(ax)
df.corr()
#Swarm plot shows how smoker feature is influencing the expeneses compare with smoker and non-smoker

ax = sns.swarmplot(x='smoker',y='expenses',data=df)

ax.set_title("Smoker vs Expenses")

plt.xlabel("Smoker (Yes - 1, No - 0)")

plt.ylabel("Expenses")

plt.show(ax)
from sklearn.linear_model import LinearRegression  #Import Linear regression model

from sklearn.model_selection import train_test_split  #To split the dataset into Train and test randomly

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
#These three features have relationship with expenses.

x = df[['age','bmi','smoker']]

y = df['expenses']

#train_test_split() to split the dataset into train and test set at random.

#test size data set should be 30% data

X_train,X_test,Y_train, Y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#Creating an linear regression model object

model = LinearRegression()

#Training the model using training data set

model.fit(X_train, Y_train) 

#X_train_predict = model.predict(X_train)

#X_test_predict = model.predict(X_test)
print("Intercept value:", model.intercept_)

print("Coefficient values:", model.coef_)
# Need to understand this syntax

coef_df = pd.DataFrame(list(zip(X_train.columns,model.coef_)), columns = ['Features','Predicted Coeff'])

coef_df

#print("Features train data:\n",X_train.smoker)
#Predicting the Y value from the train set and test set.

Y_train_predict = model.predict(X_train)

Y_train_predict[0:5]



Y_test_predict = model.predict(X_test)
#Plot to see the actual expenses and predicted expenses from Train data set

ax = sns.scatterplot(Y_train,Y_train_predict)

ax.set_title("Actual Expenses vs Predicted Expenses")

plt.xlabel("Actual Expenses")

plt.ylabel("Predicted Expenses")

plt.show(ax)
#Train and predict the Y_train for the feature 'smoker'

smoker_model = LinearRegression()

smoker_model.fit(X_train[['smoker']], Y_train)

print("intercept:",smoker_model.intercept_, "coeff:", smoker_model.coef_)



#print("Train - Mean squared error:", np.mean((Y_train - model.predict(X_train)) ** 2))

smoker_df = pd.DataFrame(list(zip(Y_train, smoker_model.predict(X_train[['smoker']]))), columns = ['Actual Expenses','Predicted Expenses'])

smoker_df.head()

#X_train['smoker'].shape
import numpy as np
#MSE for Train data set

print("MSE:",np.sqrt(mean_squared_error(Y_train, Y_train_predict)))

print("MSE only for Smoker:", np.sqrt(mean_squared_error(Y_train,smoker_model.predict(X_train[['smoker']]))))
#R-Squared value for Train data set

print("R-squared value:",round(r2_score(Y_train, Y_train_predict),3))

print("R-squared value only for smoker:", round(r2_score(Y_train,smoker_model.predict(X_train[['smoker']]))),3)
#Mean absolute error for Train data set

print("Mean absolute error:",mean_absolute_error(Y_train, Y_train_predict))

print("Mean absolute Error only for Smoker:", mean_absolute_error(Y_train,smoker_model.predict(X_train[['smoker']])))