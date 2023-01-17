# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  # for visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

insurance_df = pd.read_csv("../input/insurance.csv")



# Any results you write to the current directory are saved as output.
insurance_df.head()
insurance_df.info()
insurance_df.columns
%matplotlib inline

import seaborn as sns

sns.pairplot(insurance_df)
num_cols = insurance_df.select_dtypes(include=np.number).columns

num_cols
non_num_cols = insurance_df.select_dtypes(exclude=np.number).columns

non_num_cols
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



for i in non_num_cols:

    insurance_df[i] = label_encoder.fit_transform(insurance_df[i])
sns.boxplot(data=insurance_df)
insurance_df.head()
#find correlation after encoding

insurance_df.corr()
def evaluate_linear_regression_model(insurance_df,independent_variable):

    y=insurance_df[independent_variable]

    x=insurance_df.drop(columns=independent_variable)

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



independent_variable = 'expenses'

evaluate_linear_regression_model(insurance_df,independent_variable)