# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#read files
df_train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df_test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
#set pandas display
pd.options.display.max_columns=None
pd.options.display.max_rows=None
#make "lack table"
def lack_table(df):
    lack_number=df.isnull().sum()
    lack_ratio=lack_number/len(df)
    lack_table=pd.concat([lack_number, lack_ratio], axis=1)
    lack_table.rename(columns={0: "number", 1: "ratio"}, inplace=True)
    return lack_table
#concatenate train and test data
df_train_test=pd.concat([df_train, df_test])

#check the detail of lacking data
lack_table(df_train_test)
#before compensate lacking data, I check the correlation of each parameter in train data
df_train_corr=df_train.corr()
fig, ax=plt.subplots(figsize=(20, 20))
sns.heatmap(df_train_corr, square=True, ax=ax)
#extract highly correlated parameters
important_index_list=df_train_corr[df_train_corr["SalePrice"]>0.6].index.to_list()
important_index_list
important_df_train_test=df_train_test[["OverallQual", 
                                      "TotalBsmtSF", "1stFlrSF", "GrLivArea",
                                      "GarageCars",
                                      "GarageArea", "SalePrice"]]
#again, check the detailed lacking data
lack_table(important_df_train_test)
#fill lacking data of "important_df_train_test"
important_df_train_test["TotalBsmtSF"].fillna(important_df_train_test["TotalBsmtSF"].mean(), inplace=True)
important_df_train_test["GarageCars"].fillna(important_df_train_test["GarageCars"].mean(), inplace=True)
important_df_train_test["GarageArea"].fillna(important_df_train_test["GarageArea"].mean(), inplace=True)
#make input and output of train data
important_df_train=important_df_train_test.iloc[:1460]
train_output=important_df_train["SalePrice"].values.reshape(-1, 1)
train_input=important_df_train.drop("SalePrice", axis=1)
#use train_test_split module
x_train, x_val, y_train, y_val=train_test_split(train_input, train_output, test_size=0.2)

#fit data to linear regression model
lr=LinearRegression()
lr.fit(x_train, y_train)
#asess the prediction, using r2_score
from sklearn.metrics import r2_score
y_pred=lr.predict(x_val)
r2_score(y_val, y_pred)
#prepare test data and predict
important_df_test=important_df_train_test.iloc[1460:]
input_test=important_df_test.drop("SalePrice", axis=1)
test_pred=lr.predict(input_test)
#arange prediction data to submit
df_test_pred=pd.DataFrame(data=test_pred, columns=["SalePrice"])
df_test_id=pd.DataFrame(data=np.arange(1461, 1461+len(df_test_pred)), columns=["Id"])
submission=pd.concat([df_test_id, df_test_pred], axis=1)
submission.to_csv("submission.csv", index=False)
