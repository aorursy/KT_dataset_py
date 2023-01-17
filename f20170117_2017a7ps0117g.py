# This is the best submission

#Importing the required libraries

import pandas as pd

import numpy as np

from sklearn.ensemble import ExtraTreesRegressor

import seaborn as sns
#reading required csv files into respective dataframes

df1 = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")

df2 = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")
#Quering information of dataframe of train dataset

df1.info()
#Quering information of dataframe of train dataset

df1.describe()
#Quering information of dataframe of train dataset

df1.head()
#Counting the number of NaN Values in columns

missing_count = df1.isnull().sum()

missing_count[missing_count > 0]
#Replacing NaN values in the features with their mean values

df1["feature3"].fillna(value=df1["feature3"].mean(), inplace=True)

df1["feature4"].fillna(value=df1["feature4"].mean(), inplace=True)

df1["feature5"].fillna(value=df1["feature5"].mean(), inplace=True)

df1["feature8"].fillna(value=df1["feature8"].mean(), inplace=True)

df1["feature9"].fillna(value=df1["feature9"].mean(), inplace=True)

df1["feature10"].fillna(value=df1["feature10"].mean(), inplace=True)

df1["feature11"].fillna(value=df1["feature11"].mean(), inplace=True)
#Repacing the data "new" of column "type" as 0

df1.replace("new", 0, inplace = True)

df1.head(5)
#Repacing the data "old" of column "type" as 1

df1.replace("old", 1, inplace = True)

df1.head(5)
#Finding the datatypes of columns in the dataframe

df_dtype_nunique = pd.concat([df1.dtypes, df1.nunique()], axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
#Changing the data type of column "type" into an integer type

df1["type"]= df1["type"].astype("int64")
#plotting the correlation matrix

corr = df1.corr()

sns.heatmap(corr)

corr
rf = ExtraTreesRegressor(n_estimators=10000, random_state = 27)
#Seperating out features and dependent variable of the dataframe

x = df1.iloc[:, 1:-1]

y = df1.iloc[:, -1:].values
#Running the Regression algorithm on the train dataset

rf.fit(x, y.ravel())
#Seeing the Test Data set

df2.head()
#Repacing the data "old" of column "type" as 1

#Repacing the data "new" of column "type" as 0

#Changing the data type of column "type" into an integer type

df2.replace("old", 1, inplace = True)

df2.replace("new", 0, inplace = True)

df2["type"]= df2["type"].astype("int64")
#Counting number of NaN Values

missing_count = df2.isnull().sum()

missing_count[missing_count > 0]
#Replacing missing features with their means

df2["feature3"].fillna(value=df2["feature3"].mean(), inplace=True)

df2["feature4"].fillna(value=df2["feature4"].mean(), inplace=True)

df2["feature5"].fillna(value=df2["feature5"].mean(), inplace=True)

df2["feature8"].fillna(value=df2["feature8"].mean(), inplace=True)

df2["feature9"].fillna(value=df2["feature9"].mean(), inplace=True)

df2["feature10"].fillna(value=df2["feature10"].mean(), inplace=True)

df2["feature11"].fillna(value=df2["feature11"].mean(), inplace=True)
#Seperating out id and feature columns of the dataframe, and discarding the id column

xtest = df2.iloc[:, 1:].values
#Predicting ratings by running test dataset on our model

yPrediction = rf.predict(xtest)

yPrediction
rating_array = []

for rating_value in yPrediction:

        rating_array.append(round(rating_value))
id_column = df2.iloc[:, 0:1].values;

id_array = []

for x_temp in id_column:

    for id_value in x_temp:

        id_array.append(id_value)

result = pd.DataFrame({'id': id_array, 'rating': rating_array})
#Converting the datatype of column "rating" to integer

result["rating"]= result["rating"].astype("int64")
#Writing our final predictions to a .csv file

result.to_csv("result13.csv", index=False)
from sklearn.linear_model import LinearRegression
#reading required csv files into respective dataframes

df1 = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")

df2 = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")
#Quering information of dataframe of train dataset

df1.info()
#Quering information of dataframe of train dataset

df1.describe()
#Quering information of dataframe of train dataset

df1.head()
#Counting the number of NaN Values in columns

missing_count = df1.isnull().sum()

missing_count[missing_count > 0]
#Replacing NaN values in the features with their mean values

df1["feature3"].fillna(value=df1["feature3"].mean(), inplace=True)

df1["feature4"].fillna(value=df1["feature4"].mean(), inplace=True)

df1["feature5"].fillna(value=df1["feature5"].mean(), inplace=True)

df1["feature8"].fillna(value=df1["feature8"].mean(), inplace=True)

df1["feature9"].fillna(value=df1["feature9"].mean(), inplace=True)

df1["feature10"].fillna(value=df1["feature10"].mean(), inplace=True)

df1["feature11"].fillna(value=df1["feature11"].mean(), inplace=True)
#Repacing the data "new" of column "type" as 0

df1.replace("new", 0, inplace = True)

df1.head(5)
#Repacing the data "old" of column "type" as 1

df1.replace("old", 1, inplace = True)

df1.head(5)
#Finding the datatypes of columns in the dataframe

df_dtype_nunique = pd.concat([df1.dtypes, df1.nunique()], axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
#Changing the data type of column "type" into an integer type

df1["type"]= df1["type"].astype("int64")
#plotting the correlation matrix

corr = df1.corr()

sns.heatmap(corr)

corr
linearRegressor = LinearRegression()
#Seperating out features and dependet variable of the dataframe

x = df1.iloc[:, :-1]

y = df1.iloc[:, -1:].values
#Running the Linear Regression algorithm on the train dataset

linearRegressor.fit(x,y)
#Seeing the Test Data set

df2.head()
#Repacing the data "old" of column "type" as 1

#Repacing the data "new" of column "type" as 0

#Changing the data type of column "type" into an integer type

df2.replace("old", 1, inplace = True)

df2.replace("new", 0, inplace = True)

df2["type"]= df2["type"].astype("int64")
#Counting number of NaN Values

missing_count = df2.isnull().sum()

missing_count[missing_count > 0]
#Replacing missing features with their means

df2["feature3"].fillna(value=df2["feature3"].mean(), inplace=True)

df2["feature4"].fillna(value=df2["feature4"].mean(), inplace=True)

df2["feature5"].fillna(value=df2["feature5"].mean(), inplace=True)

df2["feature8"].fillna(value=df2["feature8"].mean(), inplace=True)

df2["feature9"].fillna(value=df2["feature9"].mean(), inplace=True)

df2["feature10"].fillna(value=df2["feature10"].mean(), inplace=True)

df2["feature11"].fillna(value=df2["feature11"].mean(), inplace=True)
#Seperating out id and feature columns of the dataframe, and discarding the id column

xtest = df2.iloc[:, :].values
#Predicting ratings by running test dataset on our model

yPrediction = linearRegressor.predict(xtest)
rating_array = []

for y_temp in yPrediction:

    for rating_value in y_temp:

        rating_array.append(round(rating_value))
id_column = df2.iloc[:, 0:1].values;

id_array = []

for x_temp in id_column:

    for id_value in x_temp:

        id_array.append(id_value)

result = pd.DataFrame({'id': id_array, 'rating': rating_array})
#Converting the datatype of column "rating" to integer

result["rating"]= result["rating"].astype("int64")
#Writing our final predictions to a .csv file

result.to_csv("final_result.csv", index=False)