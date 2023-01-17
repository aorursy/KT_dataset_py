# Last amended: 15th August, 2020



# Objective:

#           i) Learn Pipelining

#          ii) Explore RandomForest



# 1.1 Call data manipulation libraries

import pandas as pd

import numpy as np
#1.2 Call other libraries

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



from sklearn.ensemble import RandomForestClassifier as rf

import os
#1.3 To display all coumns

pd.options.display.max_columns = 100
#1.4 Display outputs of all commands from a cell--not just of the last command

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
#2.1 Set the path of the data file

path = "/kaggle/input"

os.chdir(path)

os.listdir(path)

#2.2 Read the file and peek through the various characteristics of the data

df=pd.read_csv("../input/caravan-insurance-challenge/caravan-insurance-challenge.csv")

df.head()

df.describe()

df.info()

df.dtypes

df.dtypes.value_counts()

len(df)



len(df.ORIGIN.unique())                   

df.ORIGIN.value_counts()
#2.3 Drop ORIGIN Column which consists of Test and Train values

df.pop("ORIGIN")
#2.4 Copy the target Column to a new variable and drop from the dataframe

y = df.pop("CARAVAN")

type(y)

y
# 3.1 Check if a categorical variable exist. 

# Any variable with levels less than 7 would be a categorical feature

min(df.nunique())

max(df.nunique())



df_cat = df.nunique() < 7

len(df_cat[df_cat])
# 3.2 To check whether Standard Deviation of any column is Zero

s = []

for i in df.columns:

    s.append(df[i].std())

type(s)  

s

0 in s

# 3.3 Collect numerical columns

num_columns = list(df_cat[df_cat == False].index)



len(num_columns)
# 3.4 Collect categorical columns

cat_columns = list(df_cat[df_cat].index)



len(cat_columns)
# 4.1 Create a Class for using RandomScaler for numeric columns and 

# OneHotEncoder for Categorical Columns



ct = ColumnTransformer(

                        [

                            ('abc', RobustScaler(), num_columns),

                            ('cde', OneHotEncoder(handle_unknown='ignore'), cat_columns),

                            

                        ], remainder = 'passthrough')

# 4.2 Fit and Transform the dataframe

ct.fit_transform(df)
# 4.3 Train and Test the data based on test size of 25%

X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

X_train.shape

X_test.shape
# 4.4 Check the size of train data

len(X_train.columns)

len(y_train)
# 4.5 Create pipeline with class ct and class RandomForest

pipe1 = Pipeline([

                ('ct', ct),

                ('rf', rf())

    ])

# 4.6 Fit the train data using the pipe and check the accuracy of prediction

pipe1.fit(X_train, y_train)

y_pred = pipe1.predict(X_test)

np.round(np.sum(y_pred == y_test)/len(y_test) * 100, 2)

# 5.1 Split the data based on the column ORIGIN

df2 = pd.read_csv("../input/caravan-insurance-challenge/caravan-insurance-challenge.csv")

df2

X2_train = df2[df2['ORIGIN'] == 'train']

X2_train.pop('ORIGIN')

y2_train = X2_train.pop('CARAVAN')

y2_train



X2_test = df2[df2['ORIGIN'] == 'test']

X2_test.pop('ORIGIN')

y2_test = X2_test.pop('CARAVAN')

y2_test



X2_train.shape

X2_test.shape
# 5.2 Fit the train data using the pipe and check the accuracy of prediction

pipe1.fit(X2_train, y2_train)

y2_pred = pipe1.predict(X2_test)

np.round(np.sum(y2_pred == y2_test)/len(y2_test) * 100, 2)
# The accuracy of prediction based on random split of data as well as 

# data split based on the column ORIGIN is around 93%