import numpy as np

import pandas as pd

import pandas_profiling as pp

import random



#plotly packages

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tools
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print('There are ',train.shape[0],'rows and ',train.shape[1],'columns in the train dataset')

print('There are ',test.shape[0],'rows and ',test.shape[1],'columns in the test dataset')
train.head(5)
print(list(train.columns))
data = go.Histogram(x=train.SalePrice)

fig = [data]

py.iplot(fig)
data = go.Histogram(x=np.log(train.SalePrice))

fig = [data]

py.iplot(fig)
#Saving the log_price variables

SalesPrice_log = np.log(train.SalePrice)

SalePrice = train.SalePrice

Id = train.Id
#Removing the ID and SalePrice Variable

train = train.drop(['Id','SalePrice'], 1)

test = test.drop(['Id'], 1)

print(train.shape)

print(test.shape)
#Combining the train and testing dataset

train_test = pd.concat([train,test], axis=0, sort=False)

train_test.shape
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

train_num = train.select_dtypes(include=numerics)
columns_to_rem =  train_num.columns.tolist()

train_cat = train[train.columns.difference(columns_to_rem)]
def missing_values(df):

    percent_missing = df.isnull().sum() * 100 / len(df)

    missing_value_df = pd.DataFrame({'column_name': df.columns,

                                     'percent_missing': percent_missing})

    return missing_value_df



missing_values(train_num).sort_values(by=['percent_missing'], ascending = False).head()
#Filling the missing values with mean

train_num = train_num.fillna(train_num.mean())
from scipy.stats import zscore



z = train_num.apply(zscore)

threshold = 3

np.where(z > 10)
Q1 = train_num.quantile(0.25)

Q3 = train_num.quantile(0.75)

IQR = Q3 - Q1
#Identifying rows to remove using the z-score and the interquartile range

print(train_num.shape)

train_num = train_num[(z < 10).all(axis=1)]

print(train_num.shape)
#Identifying rows to remove using the interquartile range

#ts = 20

#train_num = train_num[~((train_num < (Q1 - ts * IQR)) |(train_num > (Q3 + ts * IQR))).any(axis=1)]

#train_num.shape
train_test_num = pd.concat([train_num,test[list(train_num.columns)]])

train_test_num.head()
#Filling the missing values with mean

train_test_num = train_test_num.fillna(train_test_num.mean())
profile = pp.ProfileReport(train_test_num)
profile
from scipy.stats import skew



skew_features = train_test_num.apply(lambda x: skew(x)).sort_values(ascending=False)

skews = pd.DataFrame({'skew':skew_features})

skews.head()
from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



#Setting threshold for skew

high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



#creating a new df for skew

train_test_num_skew = pd.DataFrame()



for i in skew_index:

    train_test_num_skew[i]= boxcox1p(train_test_num[i], boxcox_normmax(train_test_num[i]+1))



#Changing col names of new df

skew_cols = [s + '_skew' for s in list(train_test_num_skew.columns)]



train_test_num_skew.columns = skew_cols

train_test_num_skew.head()
print(train_test_num[skew_index].shape) 

print(train_test_num_skew.shape)



train_test_skew_compare = pd.concat([train_test_num[skew_index],train_test_num_skew], axis = 1)

train_test_skew_compare = train_test_skew_compare.reindex(sorted(train_test_skew_compare.columns), axis=1)

train_test_skew_compare.head()
def pick_color():

    colors = ["blue","black","brown","red","yellow","green","orange","beige","turquoise","pink"]

    random.shuffle(colors)

    return colors[0]



def Hist_plot(data,i):

    trace0 = go.Histogram(

        x= data.iloc[:,i],

        name = str(data.columns[i]),

        nbinsx = 100,

        marker= dict(

            color=pick_color(),

            line = dict(

                color = 'black',

                width = 0.5

              ),

        ),

        opacity = 0.70,

  )

    fig_list = [trace0]

    title = str(data.columns[i])

    return fig_list, title

    

def Plot_grid(data, ii, ncols=2):

    plot_list = list()

    title_list = list()

    

    #Saving all the plots in a list

    for i in range(ii):

        p = Hist_plot(data,i)

        plot_list.append(p[0])

        title_list.append(p[1])

    

    #Creating the grid

    nrows = max(1,ii//ncols)

    i = 0

    fig = tools.make_subplots(rows=nrows, cols=ncols, subplot_titles = title_list)

    for rows in range(1,nrows+1):

        for cols in range(1,ncols+1):

            fig.append_trace(plot_list[i][0], rows, cols)

            i += 1

    fig['layout'].update(height=400*nrows, width=1000)

    return py.iplot(fig)

Plot_grid(train_test_skew_compare,len(train_test_skew_compare.columns),2)
#Combinig the skewed data with the original train test dataset

train_test_num = pd.concat([train_test_num, train_test_num_skew], axis = 1)

train_test_num.shape
#Lets check how the data looks like when BsmtFinSF1 is zero for columns that contain Bsmt

bsmt_cols = [col for col in train_test if 'Bsmt' in col]

train_bsmt = train[bsmt_cols][train['BsmtFinSF1']== 0]

train_bsmt.head(10)

def compare_df(data1, data2, column_name):

    data1 = data1.filter(regex = column_name).iloc[:,0]

    df1 = pd.DataFrame(data1.value_counts(normalize=True) * 100)

    

    data2 = data2.filter(regex = column_name).iloc[:,0]

    df2 = pd.DataFrame(data2.value_counts(normalize=True) * 100)

    

    df3 = pd.merge(df1,df2, how='outer', left_index=True, right_index=True)

    

    return df3



train_bsmt_cat = train_bsmt.select_dtypes(exclude=["number","bool_"])



for col in train_bsmt_cat.columns:

    print(compare_df(train_bsmt,train,col))
train_test_num['BsmtFinSF1'] = np.where(train_test_num['BsmtFinSF1'] <= 0, -1,train_test_num['BsmtFinSF1'])
train_bsmt = train[bsmt_cols][train['BsmtFinSF2']== 0]

for col in train_bsmt_cat.columns:

    print(compare_df(train_bsmt,train,col))
train_test_num['BsmtFinSF2'] = np.where(train_test_num['BsmtFinSF2'] <= 0, -1,train_test_num['BsmtFinSF2'])
missing_values(train_test_num).sort_values(by=['percent_missing'], ascending = False).head()
train_test_cat = pd.concat([train_cat,test[list(train_cat.columns)]])

train_test_cat.head()
missing_values(train_test_cat).sort_values(by=['percent_missing'], ascending = False)
#profile = pp.ProfileReport(train_test_cat)
#profile
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb