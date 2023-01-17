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

df=pd.read_csv("../input/bike_share.csv")

df.head(10)
df.shape

#This contains 10886 rows into 11 columns. Count is the output column
df.info()

#All the columns are numerical and there is no categorical columns
df.duplicated().sum()

#There are 21 duplicates to remove
df.drop_duplicates(inplace=True)
df.duplicated().sum()
df.corr()['count']

#Only temp has correlation 0.4.Can drop others
df.isna().sum()
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error



plt.figure(figsize=(10,5))

ax = sns.heatmap(df.corr(), annot=True)

plt.show(ax)

#This plot is not needed. But for practice i have included.
df.head(25)

#Here casual+registered=count hence we can exclude from input X

#Also can include either temp or atemp
df.drop(columns=['season', 'holiday', 'workingday','weather','humidity',

       'windspeed'],inplace=True)
df.columns

#Predicting temp based on count
ax = sns.scatterplot(x="temp", y="count", data=df)

#Here the count is greater at humidities 20 -50 and low at low humidity and gradually decreases 

# as the humidity increases
df['temp_label'] = df['temp'].apply(lambda x: '0-4' if x < 5 else '5-9' if x < 10 else '10-14' if x < 15 else '15-19' if x < 20 else '20-24' if x < 25 else '25-29' if x < 30 else '30-34' if x < 35 else '35-39' if x < 40 else '40-44' if x < 45 else '45-49')
#One Hot Encoding for bucket columns and concatenate all the dataframes

encode_temp_label = pd.get_dummies(df['temp_label'])



df2 = pd.concat([df,encode_temp_label],axis='columns')                     

df2.head()
df2.drop(columns=['temp_label'],inplace=True)

df2.columns
from sklearn.linear_model import LinearRegression  

from sklearn.model_selection import train_test_split 

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

modelinput = df2.drop(columns=['count'],axis=1)

modeloutput = df2['count']

X_train,X_test,Y_train, Y_test = train_test_split(modelinput,modeloutput,test_size=0.3,random_state=123)

model = LinearRegression()

model.fit(X_train,Y_train)

Y_train_predict = model.predict(X_train)

Y_test_predict = model.predict(X_test)

print("-----------------------All included--------------------------")

print("MSE Train:",mean_squared_error(Y_train, Y_train_predict))

print("MSE Test:",mean_squared_error(Y_test, Y_test_predict))

print("RMSE Train:",np.sqrt(mean_squared_error(Y_train, Y_train_predict)))

print("RMSE Test:",np.sqrt(mean_squared_error(Y_test, Y_test_predict)))

print('MAE Train', mean_absolute_error(Y_train, Y_train_predict))

print('MAE Test', mean_absolute_error(Y_test, Y_test_predict))

print('R2 Train',r2_score(Y_train, Y_train_predict))

print('R2 Test',r2_score(Y_test, Y_test_predict))