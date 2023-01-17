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
dataf=pd.read_csv("../input/bike_share.csv")
dataf.head()
dataf.info()
dataf.describe().T
dataf.duplicated().sum()
dataf.shape
dataf.drop_duplicates(inplace=True)
dataf.duplicated().sum()
dataf.isna().sum()
dataf["temp"].unique()
%matplotlib inline

dataf.season.plot(kind="box")

dataf.windspeed.plot(kind="box")









dataf.casual.plot(kind="box")



dataf.registered.plot(kind="box")

dataf["count"].plot(kind="box")
dataf["count"].value_counts()
dataf.corr()
train_X = dataf.drop(columns=["temp","casual","registered"])
train1_Y = dataf["casual"]

train2_Y = dataf["registered"]

train3_Y = dataf["count"]
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model
a = cross_val_score(model, train_X, train_Y, cv=5, scoring='neg_mean_squared_error')


np.mean(np.sqrt(np.abs(a)))
model.fit(train_X, train1_Y)

        

    #Predict training set:

dtrain_predictions = model.predict(train_X)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train_X, train1_Y, test_size=0.3, random_state=123)
model.fit(X_train,Y_train)
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
print("Train:",mean_absolute_error(Y_train,train_predict))

print("Test:",mean_absolute_error(Y_test,test_predict))
print('r2 train',r2_score(Y_train,train_predict))

print('r2 test',r2_score(Y_test,test_predict))
import seaborn as sns
for i in dataf.columns:

        sns.pairplot(data=dataf,x_vars=i,y_vars='count')