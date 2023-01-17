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
df=pd.read_csv("../input/Big_Cities_Health_Data_Inventory.csv")
df.head()
df.info()
df[df.duplicated(keep=False)]
df.duplicated().sum()
df.iloc[12320:12325,:]
df=df.drop_duplicates()
df.iloc[12320:12325,:]
#Since there are so many categorical column analyze what are the coluns are required.
df.columns
df['Indicator Category'].value_counts()
df.Indicator.nunique()
df.Indicator.value_counts()
#Assuming/understanding that based on indicator it is already categorized and given as indocator category. So drop indaicator column
df=df.drop(columns='Indicator')
df.Year.value_counts()
#considering the rows which is not having the -(hyphen)

df = df[~df.Year.str.contains("-")]
df.Year.value_counts()
df.Gender.value_counts()
df['Race/ Ethnicity'].value_counts()
df.Place.value_counts()
df['BCHC Requested Methodology'].value_counts()
df=df.drop(columns='BCHC Requested Methodology')
df.columns
df.Source.value_counts()
df.Notes.value_counts()
df.Methods.value_counts()
#unable find any relevance - so removing the columns

df= df.drop(columns=["Source","Notes","Methods"])



df.columns
df.info()
df.Place
df["city_Info"] = df.Place.apply(lambda x : x[-2:])
df["city_Info1"] = df.Place.apply(lambda x : x[:-4])
df.city_Info1.head()
df["city_Info1"].value_counts()
df["city_Info"].head()
df["city_Info"].value_counts()
df.head()
df=df.drop(columns='Place')
df.head()
df.Value.isna().sum()
df.dropna(inplace=True)
#encoding 

df_column_cat = df.select_dtypes(exclude=np.number).columns
encoded_cat_col = pd.get_dummies(df[df_column_cat])
df_final = pd.concat([df['Value'],encoded_cat_col], axis = 1)
X = df_final.drop(columns=['Value'])

y = df_final['Value']
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

model = LinearRegression()

model.fit(X_train,y_train)
print("*****coefficient valuessss",model.coef_)

print("*****intercept iss",model.intercept_)
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
train_predict = model.predict(X_train)



mae_train = mean_absolute_error(y_train,train_predict)



mse_train = mean_squared_error(y_train,train_predict)



rmse_train = np.sqrt(mse_train)



r2_train = r2_score(y_train,train_predict)



mape_train = mean_absolute_percentage_error(y_train,train_predict)
test_predict = model.predict(X_test)



mae_test = mean_absolute_error(test_predict,y_test)



mse_test = mean_squared_error(test_predict,y_test)



rmse_test = np.sqrt(mean_squared_error(test_predict,y_test))



r2_test = r2_score(y_test,test_predict)



mape_test = mean_absolute_percentage_error(y_test,test_predict)
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

print('TRAIN: Mean Absolute Error(MAE): ',mae_train)

print('TRAIN: Mean Squared Error(MSE):',mse_train)

print('TRAIN: Root Mean Squared Error(RMSE):',rmse_train)

print('TRAIN: R square value:',r2_train)

print('TRAIN: Mean Absolute Percentage Error: ',mape_train)

print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

print('TEST: Mean Absolute Error(MAE): ',mae_test)

print('TEST: Mean Squared Error(MSE):',mse_test)

print('TEST: Root Mean Squared Error(RMSE):',rmse_test)

print('TEST: R square value:',r2_test)

print('TEST: Mean Absolute Percentage Error: ',mape_test)
sns.scatterplot(y_train,train_predict)