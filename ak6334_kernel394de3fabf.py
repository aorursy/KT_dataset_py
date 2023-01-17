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
bkpath  = r'../input/bike_share.csv'

bk = pd.read_csv(bkpath)

bk.head()
bk.info()
bk.isna().sum()
bk.columns
%matplotlib inline

import seaborn as sns

sns.pairplot(bk)
num_cols = bk.select_dtypes(include=np.number).columns

num_cols
nonnumcols = bk.select_dtypes(exclude=np.number).columns

nonnumcols
bk.head()
bk.corr()
import matplotlib.pyplot as plt  # for visualization

def evaluate_linear_regression_model(insurance_df,independent_variable):

    y=bk[independent_variable]

    x=bk.drop(columns=independent_variable)

    train_X,test_X,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

    

    model = LinearRegression()

    model.fit(train_X,train_y)

    

    print("Intercept : ", model.intercept_)

    print("Slope : ", model.coef_)

    

    #Predicting TEST & TRAIN DATA

    train_predict = model.predict(train_X)

    test_predict = model.predict(test_X)

    

    print("MAE")

    print("Train : ",mean_absolute_error(train_y,train_predict))

    print("Test  : ",mean_absolute_error(test_y,test_predict))

    print("====================================")

    

    print("MSE")

    print("Train : ",mean_squared_error(train_y,train_predict))

    print("Test  : ",mean_squared_error(test_y,test_predict))

    print("====================================")

    

    print("RMSE")

    print("Train : ",np.sqrt(mean_squared_error(train_y,train_predict)))

    print("Test  : ",np.sqrt(mean_squared_error(test_y,test_predict)))

    print("====================================")

    

    print("R2 SCORE")

    print("Train : ",r2_score(train_y,train_predict))

    print("Test  : ",r2_score(test_y,test_predict))  

    print("====================================")

    

    print("MAPE - Mean Absolute Percentage Error")

    train_y, train_predict = np.array(train_y), np.array(train_predict)

    print(np.mean(np.abs((train_y - train_predict) / train_y)) * 100)

    print("Mape - Train:" , np.mean(np.abs((train_y,train_predict))))

    print("Mape - Test:" ,np.mean(np.abs((test_y,test_predict))))

    

    #Plot actual vs predicted value

    plt.figure(figsize=(10,7))

    plt.title("Actual vs. predicted",fontsize=25)

    plt.xlabel("Actual",fontsize=18)

    plt.ylabel("Predicted", fontsize=18)

    plt.scatter(x=test_y,y=test_predict)


# Evaluate Linear Regression Prediction Model

from sklearn.linear_model import LinearRegression  #Import Linear regression model

from sklearn.model_selection import train_test_split  #To split the dataset into Train and test randomly

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score





independent_variable = 'count'

evaluate_linear_regression_model(bk,independent_variable)