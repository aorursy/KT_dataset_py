# Importing required packages

from sklearn.preprocessing import LabelEncoder

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from scipy import stats

import statsmodels.api as sm

from statsmodels.formula.api import ols
def overview():

    '''

    Read a comma-separated values (csv) file into DataFrame.

    Print 5 rows of data

    Print number of rows and columns

    Print datatype for each column

    Print number of NULL/NaN values for each column

    Print summary data

    

    Return:

    data, rtype: DataFrame

    '''

    data = pd.read_csv('../input/insurance/insurance.csv')

    print("The first 5 rows if data are:\n", data.head())

    print("\n")

    print("The (Row,Column) is:\n", data.shape)

    print("\n")

    print("Data type of each column:\n", data.dtypes)

    print("\n")

    print("The number of null values in each column are:\n", data.isnull().sum())

    print("\n")

    print("Numeric summary:\n", data.describe())

    return data



df = overview()
# Sorting out numerical and categorical variables 

def categorical_variable(df):

    return list(df.select_dtypes(include = ['category', 'object']))



def numerical_variable(df):

    return list(df.select_dtypes(exclude = ['category', 'object']))

g = sns.catplot(x="smoker", y="charges",col_wrap=3, col="sex",data= df, kind="box",height=5, aspect=0.8);
g = sns.catplot(x="smoker", y="age",col_wrap=3, col="sex",data= df, kind="box",height=5, aspect=0.8);
# Changing categorical values to numeric

df2 = df.copy()

#sex

le = LabelEncoder()

le.fit(df2.sex.drop_duplicates()) 

df2.sex = le.transform(df2.sex)

# smoker or not

le.fit(df2.smoker.drop_duplicates()) 

df2.smoker = le.transform(df2.smoker)

#region

le.fit(df2.region.drop_duplicates()) 

df2.region = le.transform(df2.region)
print(df2.corr()['charges'])



lm = ols('sex ~ charges', data = df2).fit()

table = sm.stats.anova_lm(lm)

print("P-value for 1-way ANOVA test between sex and charges is: ",table.loc['charges','PR(>F)'])

lm1 = ols('smoker ~ charges', data = df2).fit()

table1 = sm.stats.anova_lm(lm1)

print("P-value for 1-way ANOVA test between smoker and charges is: ",table1.loc['charges','PR(>F)'])

lm2 = ols('region ~ charges', data = df2).fit()

table2 = sm.stats.anova_lm(lm2)

print("P-value for 1-way ANOVA test between region and charges is: ",table2.loc['charges','PR(>F)'])
# Creating training and testing dataset

y = df2['charges']

X = df2.drop(['charges'], axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42)



lr = LinearRegression().fit(X_train,y_train)

y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)



print(lr.score(X_test,y_test))
# Creating training and testing dataset

y = df2['charges']

X = df2.drop(['charges', 'region'], axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42)



lr = LinearRegression().fit(X_train,y_train)

y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)



print(lr.score(X_test,y_test))
# Creating training and testing dataset

y = df2['charges']

X = df2.drop(['charges', 'region'], axis = 1)



poly_reg  = PolynomialFeatures(degree=2)

X_poly = poly_reg.fit_transform(X)



X_train,X_test,y_train,y_test = train_test_split(X_poly,y,test_size=0.20, random_state = 42)



lin_reg = LinearRegression()

lin_reg  = lin_reg.fit(X_train,y_train)



print(lin_reg.score(X_test,y_test))