# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import ElasticNet

import matplotlib.pyplot as plt

import seaborn as sns

from math import log

from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def miss_values(df):

    for column in df:

        # Test whether column has null value

        if len(df[column].apply(pd.isnull).value_counts()) > 1:

            print(column+" has missing value")

            #if column is numeric, fill null with mean

            if df[column].dtype in ('int64','float64'):

                df[column] = df[column].fillna(df[column].mean())

            else:

                df[column] = df[column].fillna("unknown")
def log_skew(df):

    for column in df:

        if df[column].dtype in ('int64','float64') and column != 'SalePrice':

            old_skew = df[column].skew()

            if abs(df[column].skew()) > 1.0:

                df[column] = df[column].apply(lambda x: log(x+1,2))

                print('the skewness of '+column+" is reduced from "+\

                      str(old_skew) + " to "+str(df[column].skew()))
def factor_encoding(df):

    for column in df:

        if df[column].dtype == 'object':

            df = df.merge(pd.get_dummies(data=df[column],prefix=column),right_index=True,left_index=True)

            del df[column]

    return df
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train['source'] = 1.0

test['source'] = 0.0

alls = pd.concat([train,test],ignore_index=True)

miss_values(alls)

#log_skew(alls)

alls = factor_encoding(alls)
train = alls[alls['source'] == 1.0]

train.drop(['source'],inplace=True,axis=1)

test = alls[alls['source'] == 0.0]

test.drop(['source'],inplace=True,axis=1)
x = train.drop(['SalePrice'],axis=1)

Y = train['SalePrice']

x_test = test.drop(['SalePrice'],axis=1)

x_train, x_vali, Y_train, Y_vali = train_test_split(x, Y, test_size=0.25, random_state=42)
para_grid = {

      'alpha' : [i for i in range(1,5)], #Alpha decides degree of penalization

      'l1_ratio':[i*0.1 for i in range(0,10)] #decides degree of mixing of l1 and l2 penality

}
en = ElasticNet()

clf = GridSearchCV(estimator=en, param_grid=para_grid,scoring='neg_mean_squared_error')

clf.fit(x, Y)
clf.best_params_
test_predicted = clf.predict(x_test)

test['SalePrice'] = test_predicted

result = test[['Id','SalePrice']]

result.to_csv('submission.csv',index=False)