#Best Score Submission
#Importing libraries

import pandas as pd

import seaborn as sns

from xgboost import XGBClassifier
#reading required csv files into respective dataframes

df1 = pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")

df2 = pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")
#Quering information of dataframe of train dataset

df1.info()
#Quering information of dataframe of train dataset

df1.describe()
#Quering information of dataframe of train dataset

df1.head()
#Counting the number of NaN Values in columns

missing_count = df1.isnull().sum()

missing_count
#Finding the datatypes of columns in the dataframe

df_dtype_nunique = pd.concat([df1.dtypes, df1.nunique()], axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
#plotting the correlation matrix

corr = df1.corr()

sns.heatmap(corr)

corr
xgb = XGBClassifier()
#Seperating out features and dependent variable of the dataframe

x = df1.iloc[:, 1:-1]

y = df1.iloc[:, -1:].values

x
#Running the classification algorithm on the train dataset

xgb.fit(x, y.ravel())

print(xgb)
#Seeing the Test Data set

df2.head()
#Counting number of NaN Values

missing_count = df2.isnull().sum()

missing_count
df2.info()
#Seperating out id and feature columns of the dataframe, and discarding the id column

xtest = df2.iloc[:, 1:]

xtest
#Predicting classes by running test dataset on our model

yPrediction = xgb.predict(xtest)

yPrediction
id_column = df2.iloc[:, 0:1].values;

id_array = []

for x_temp in id_column:

    for id_value in x_temp:

        id_array.append(id_value)

result = pd.DataFrame({'id': id_array, 'class': yPrediction})

result
#Writing our final predictions to a .csv file

result.to_csv("result1.csv", index=False)
# Second Submission
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from xgboost import XGBClassifier
#reading required csv files into respective dataframes

df1 = pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")

df2 = pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")
#Quering information of dataframe of train dataset

df1.info()
#Quering information of dataframe of train dataset

df1.describe()
#Quering information of dataframe of train dataset

df1.head()
#Counting the number of NaN Values in columns

missing_count = df1.isnull().sum()

missing_count
#Finding the datatypes of columns in the dataframe

df_dtype_nunique = pd.concat([df1.dtypes, df1.nunique()], axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
#plot heat map

plt.subplots(figsize=(20,9))

g=sns.heatmap(df1.corr(),annot=True,cmap="RdYlGn")
#Removing features that have little correlation with the class

df1 = df1.drop(columns=["chem_2", "chem_3", "chem_7"])

df1
xgb = XGBClassifier()
#Seperating out features and dependent variable of the dataframe

x = df1.iloc[:, 1:-1]

y = df1.iloc[:, -1:].values

x
#Running the classification algorithm on the train dataset

xgb.fit(x, y.ravel())
#Seeing the Test Data set

df2.head()
#Counting number of NaN Values

missing_count = df2.isnull().sum()

missing_count
#Quering information of dataframe of test dataset

df2.info()
#Seperating out id and feature columns of the dataframe, and discarding columns that are not needed

df2= df2.drop(columns=["chem_2", "chem_3", "chem_7"])

xtest = df2.iloc[:, 1:]

xtest
#Predicting classes by running test dataset on our model

yPrediction = xgb.predict(xtest)

yPrediction
id_column = df2.iloc[:, 0:1].values;

id_array = []

for x_temp in id_column:

    for id_value in x_temp:

        id_array.append(id_value)

result = pd.DataFrame({'id': id_array, 'class': yPrediction})

result
#Writing our final predictions to a .csv file

result.to_csv("result9.csv", index=False)