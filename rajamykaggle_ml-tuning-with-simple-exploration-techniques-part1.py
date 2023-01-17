# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Lets us import other standard libraries

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import math

import matplotlib.pyplot as plt



%matplotlib inline
auto_mpg=pd.read_csv("/kaggle/input/autompg-dataset/auto-mpg.csv")

auto_mpg.head().T
auto_mpg.dtypes
auto_mpg['horsepower']
auto_mpg['horsepower'].unique()
auto_mpg['horsepower'].replace({'?':auto_mpg['horsepower'].mode()[0]},inplace=True)

auto_mpg['horsepower']=auto_mpg['horsepower'].astype(int)

auto_mpg['horsepower'].value_counts()
auto_mpg.describe().T
cat_cols=['origin','model year','cylinders']

for col in cat_cols:

    print("Value count of the column :",col)

    print(auto_mpg[col].value_counts())
#scatterplot

sns.set()

sns.pairplot(auto_mpg, height = 2.0)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import math



#This method will be invoked to build the model and check the RMSE value

def build_model(X,y):

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=101)

    

    auto_mpg_model=LinearRegression()

    auto_mpg_model.fit(X=X_train,y=y_train)

    

    y_pred_lr=auto_mpg_model.predict(X_test)

    print("Linear Regressor RMSE :",math.sqrt(mean_squared_error(y_test,y_pred_lr)))



    return y_pred_lr, y_test
#car_name column is string. As of now drop it.  We may try exploring this further to find out car manufacturer and then check how mpg varies based on the manufacturer.

drop_cols=['mpg','car name']

target_col='mpg'



#Lets take the backup of the original dataset as we will change this dataset later

auto_mpg_orig=auto_mpg.copy()



print("Let us check for full dataset:")

_,_=build_model(auto_mpg_orig.drop(drop_cols,axis=1),auto_mpg_orig[target_col])
def cat_plot(df,cur_cat_var):

    f, ax = plt.subplots(figsize=(8, 6))

    plt.axhline(df.mpg.mean(),color='r',linestyle='dashed',linewidth=2)

    sns.boxplot(x=cur_cat_var, y="mpg", data=df)
cat_plot(auto_mpg_orig,'origin')
cat_plot(auto_mpg_orig,'cylinders')
cat_plot(auto_mpg_orig,'model year')
#creating sub groups with respect to cylinders and model year to split below and apply average. For origin lets build model for each origin

cat_var={'origin':[[1],[2],[3]],

         'cylinders':[[3,6,8],[5],[4]],

         'model year':[[70,72,73,75],

                       [71,74,76,77,78,79],

                       [80,81,82]]

         }







#Lets loop through all these categorical variables.

for var_name in cat_var.keys():

    #Using below variables to determine the overall RMSE after consolidating all sub groups created.

    y_test_full=pd.Series()

    y_pred_full_lr = []

    for cur_val in cat_var[var_name]:

        auto_mpg=auto_mpg_orig[auto_mpg_orig[var_name].isin(cur_val)]     

        

        X=auto_mpg.drop(drop_cols,axis=1)

        y=auto_mpg['mpg']

    

        y_pred_lr,y_test=build_model(X,y)

    

        y_test_full=y_test_full.append(y_test)

        y_pred_full_lr=np.append(y_pred_full_lr,y_pred_lr)

        print("Linear Regressor RMSE upto ",var_name," : ",cur_val," ==>",math.sqrt(mean_squared_error(y_test_full,y_pred_full_lr)))
