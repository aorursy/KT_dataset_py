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
#import all the required libraries



import numpy as np, pandas as pd

import scipy.stats as stat

import sklearn as sk

import pandas_profiling

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

import statsmodels as sm

import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

House_Prices = pd.read_csv("../input/House_Prices.csv")
#checking the shape of dataset

House_Prices.shape
#checking the data types 

House_Prices.dtypes
#dividing the dataset into continous and variables



House_Prices_cont= House_Prices.select_dtypes(include='int64')

House_Prices_var=House_Prices.select_dtypes(include='object')
#data auditing for the report 



def var_summary(x): 

    return pd.Series([x.count(),x.isnull().sum(),x.sum(),x.mean(),x.median(),x.std(),x.var(),x.std()/x.mean(),x.min(),x.max(),

                         x.quantile(0.01),x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75),

                         x.quantile(0.90),x.quantile(0.99),x.max()],index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'CV','MIN', 'P1' , 'P5' ,'P10' ,'P25' 

                                                                            ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])
cont_summary=House_Prices_cont.apply(var_summary).T

cont_summary
def cat_summary(x):

    return pd.Series([x.count(), x.isnull().sum(), x.value_counts()], 

                  index=['N', 'NMISS', 'ColumnsNames'])
cat_summary = House_Prices_var.apply(cat_summary)

cat_summary
# An utility function to create dummy variable

def create_dummies( df, colname ):

    col_dummies = pd.get_dummies(df[colname], prefix=colname, drop_first=True)

    df = pd.concat([df, col_dummies], axis=1)

    df.drop( colname, axis = 1, inplace = True )

    return df
#for c_feature in categorical_features



for c_feature in ['Brick', 'Neighborhood']:

    House_Prices_var[c_feature] = House_Prices_var[c_feature].astype('category')

    House_Prices_var = create_dummies(House_Prices_var , c_feature )
#join both the tables 



house = pd.concat([House_Prices_cont,House_Prices_var], axis=1)

house.columns
house.head()
#data exploration 

plt.figure(figsize=(10,10))

sns.distplot(house.Price,)
cor = house.corr()

cor
plt.figure(figsize=(10,10))

sns.heatmap(cor)
#here we are dividing the house set into test and train

train, test = train_test_split( house,test_size = 0.3,random_state = 1234 )
# P value is more than 0.05 hence  we reject the value of Neighborhood_North

lm = smf.ols("Price~Bathrooms+Bedrooms+Brick_Yes+Neighborhood_West+Offers+SqFt", train).fit()

lm
lm.summary(())
# Price = 2924.52 + 6982.5 * Bathrooms + 5188.566 * Bedrooms + 1.66e+04*  Brick_Yes + 2.05e+04 * Neighborhood_West -6971.3968 * Offers + 50.77 * SqFt
#predict the price on train data

train["pred_price"] = lm.predict(train)
#predict the price on test data

test["pred_price"] = lm.predict(test)
#Accuracy of the model 
#Train Data

MAPE_train = np.mean(np.abs(train.Price - train.pred_price)/train.Price )

print(MAPE_train)



#Test Data

MAPE_test = np.mean(np.abs(test.Price - test.pred_price)/test.Price )

print(MAPE_test)





print('MAPE of training data: ', MAPE_train,  ' | ', 'MAPE of testing data: ', MAPE_test)
#Train Data

RMSE_train = mean_squared_error(train.Price , train.pred_price)

print(RMSE_train)





#Test Data

RMSE_test = mean_squared_error(test.Price , test.pred_price)

print(RMSE_test)



print('RMSE of training data: ', RMSE_train,  ' | ', 'RMSE of testing data: ', RMSE_test)
#Train Data

Corr_train = stat.stats.pearsonr(train.Price , train.pred_price)

print(Corr_train)





#Test Data

Corr_test = stat.stats.pearsonr(test.Price , test.pred_price)

print(Corr_test)



print('Correlation of training data: ', Corr_train,  ' | ', 'Correlation of testing data: ', Corr_test)

df = pd.DataFrame({'Actual': train.Price, 'Predicted': train.pred_price})

df.head()
df1 = df.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()