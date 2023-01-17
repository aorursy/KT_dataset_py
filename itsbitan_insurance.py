# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualision

import seaborn as sns # data visualision

import scipy as sp # statistical calcution



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Import the dataset

df = pd.read_csv("/kaggle/input/insurance/insurance.csv")

df.head()
#Lets look general information of our datset

df.info()
#Now look if there are any missing value or not

df.isnull().sum()
#Now we look the statistical inferance of datset

df.describe()
# At first we check our target variable

sns.distplot(df['charges'])

print('Skewness: %f', df['charges'].skew())

print("Kurtosis: %f" % df['charges'].kurt())
#Log Transformation

df['charges_log'] = np.log(df['charges'])

sns.distplot(df['charges_log'], color = 'blue')

print('Skewness: %f', df['charges_log'].skew())

print("Kurtosis: %f" % df['charges_log'].kurt())
#Now lets drop the charges coloumn

df.drop('charges', axis = 1, inplace = True)
#Lets look the distribution of bmi

sns.distplot(df['bmi'], color = 'purple')

plt.title('Distribution of BMI')

print('Skewness: %f', df['bmi'].skew())

print("Kurtosis: %f" % df['bmi'].kurt())
#Lets look the distribution of age

sns.distplot(df['age'], color = 'deeppink')

plt.title('Distribution of Age')

print('Skewness: %f', df['age'].skew())

print("Kurtosis: %f" % df['age'].kurt())
#Now lets count the categorical variables

#Count of Sex

sns.countplot(df['sex'])

plt.title('Count of Sex')
#Count of Smoker

sns.countplot(df['smoker'])

plt.title('Count of Smoker')

# Encoding the Categorical data

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

df['sex'] = labelencoder.fit_transform(df['sex'])#Female=0, Male=1

df['smoker'] = labelencoder.fit_transform(df['smoker'])#Smoker=1, Non-smoker=0

df['region'] = labelencoder.fit_transform(df['region'])

#Lets see the Distribution of charges_log of smoker and non-smoker

f= plt.figure(figsize=(12,5))



ax=f.add_subplot(121)

sns.distplot(df[(df.smoker == 1)]["charges_log"],color='c',ax=ax)

ax.set_title('Distribution of charges for smokers')



ax=f.add_subplot(122)

sns.distplot(df[(df.smoker == 0)]['charges_log'],color='b',ax=ax)

ax.set_title('Distribution of charges for non-smokers')
#Lets see the Distribution of charges_log of male and female

f= plt.figure(figsize=(12,5))



ax=f.add_subplot(121)

sns.distplot(df[(df.sex == 1)]["charges_log"],color='pink',ax=ax)

ax.set_title('Distribution of charges for male')



ax=f.add_subplot(122)

sns.distplot(df[(df.smoker == 0)]['charges_log'],color='purple',ax=ax)

ax.set_title('Distribution of charges for female')

#Correlation and Heat map

corr = df.corr()

colormap = sns.diverging_palette(220, 10, as_cmap = True)

plt.figure(figsize = (8,8))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            annot=True,fmt='.2f',linewidths=0.30,

            cmap = colormap, linecolor='white')

plt.title('Correlation of df Features', y = 1.05, size=15)
#Lets look the correlation score

print (corr['charges_log'].sort_values(ascending=False), '\n')
df.drop('region', axis = 1, inplace = True)
#BMI with Charges_log

sns.jointplot(x = df['bmi'], y = df['charges_log'], kind ='scatter',color = 'green')
#Age and Charges_log

sns.jointplot(x = df['age'], y = df['charges_log'], kind ='scatter',color = 'red')
#At first we take our matrices of features

x = df.iloc[:,:-1].values

y = df.iloc[:,5].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state = 0)
#Fitting Simple Linear Regression to the tranning set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)
#Lets check the accuracy of trainning set

accuracy_train = regressor.score(x_train, y_train)

print(accuracy_train)
#Predicting test set result

y_pred = regressor.predict(x_test)
#Lets check the accuracy of test set

accuracy_test = regressor.score(x_test, y_test)

print(accuracy_test)


#Now Check the error for regression

from sklearn import metrics

print('MAE :'," ", metrics.mean_absolute_error(y_test,y_pred))

print('MSE :'," ", metrics.mean_squared_error(y_test,y_pred))

print('RMAE :'," ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

#Visualising the Acutal and predicted Result

plt.plot(y_test, color = 'deeppink', label = 'Actual')

plt.plot(y_pred, color = 'blue', label = 'Predicted')

plt.grid(alpha = 0.3)

plt.xlabel('Number of Candidate')

plt.ylabel('charges_log')

plt.title('Actual vs Predicted')

plt.legend()

plt.show()
#Fitting SVR to dataset

from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(x_train, y_train)
#Lets check the accuracy of trainning set

accuracy_train = regressor.score(x_train, y_train)

print(accuracy_train)
#Predicting test set result

y_pred = regressor.predict(x_test)
#Lets check the accuracy of test set

accuracy_test = regressor.score(x_test, y_test)

print(accuracy_test)
#Now Check the error for regression

from sklearn import metrics

print('MAE :'," ", metrics.mean_absolute_error(y_test,y_pred))

print('MSE :'," ", metrics.mean_squared_error(y_test,y_pred))

print('RMAE :'," ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

#Visualising the Acutal and predicted Result

plt.plot(y_test, color = 'deeppink', label = 'Actual')

plt.plot(y_pred, color = 'blue', label = 'Predicted')

plt.grid(alpha = 0.3)

plt.xlabel('Number of Candidate')

plt.ylabel('charges_log')

plt.title('Actual vs Predicted')

plt.legend()

plt.show()
#Now fitting the Random forest regression to the traning set

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(x_train, y_train)

#Lets check the accuracy of trainning set

accuracy_train = regressor.score(x_train, y_train)

print(accuracy_train)
#Predicting test set result

y_pred = regressor.predict(x_test)
#Lets check the accuracy of test set

accuracy_test = regressor.score(x_test, y_test)

print(accuracy_test)
#Now Check the error for regression

from sklearn import metrics

print('MAE :'," ", metrics.mean_absolute_error(y_test,y_pred))

print('MSE :'," ", metrics.mean_squared_error(y_test,y_pred))

print('RMAE :'," ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#Visualising the Acutal and predicted Result

plt.plot(y_test, color = 'deeppink', label = 'Actual')

plt.plot(y_pred, color = 'blue', label = 'Predicted')

plt.grid(alpha = 0.3)

plt.xlabel('Number of Candidate')

plt.ylabel('charges_log')

plt.title('Actual vs Predicted')

plt.legend()

plt.show()
y_pred = np.expm1(y_pred)