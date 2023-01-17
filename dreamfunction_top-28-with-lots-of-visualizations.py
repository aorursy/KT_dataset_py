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
import numpy as np
from scipy import stats
from scipy.stats import norm, skew
import datetime as dt
import math
from math import radians, cos, sin, asin,sqrt
import glob
import os
import pandas as pd
import pandas_profiling
pd.set_option('display.max_columns', None)
# Visualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# NOTE THAT INLINE NEEDS TO BE LAST
%matplotlib inline
# Missing Data Visualization
import missingno as msno
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.shape, test.shape
train.columns
train.head()
#train = train.drop(train[((train['GrLivArea']>4000) & (train['SalePrice']<300000)) | (train['SalePrice']>700000)].index)
train = train.drop(train[((train['GrLivArea']>4000) & (train['SalePrice']<300000))].index)

train.reset_index(drop=True)
data = train.append(test, ignore_index=True)
data.describe()
data.info()
data.isnull().sum()
# Creating a list to see all the columns that have any missing values, assigning to a variable in case I want to use this at some point
list_of_na_columns = data.columns[data.isna().any()].tolist()
list_of_na_columns
#visualization to see the missing values
msno.bar(data)
pandas_profiling.ProfileReport(data)
data.loc[(data.PoolArea!=0) & (data.PoolQC.isnull())]
# Make some graphs to see if I can impute these by that information
fig= plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121)
sns.boxplot(x='PoolQC', y='PoolArea', data=data, ax=ax1)
ax2 = fig.add_subplot(122)
sns.boxplot(x='PoolQC', y='SalePrice', data=data, ax=ax2)

# Changing the specific null values to 'Ex' or 'Fa'
data['PoolQC'].loc[(data['PoolQC'].isnull()) & (data['PoolArea'] < 500) & (data['PoolArea'] !=0)] = 'Ex'
data['PoolQC'].loc[(data['PoolQC'].isnull()) & (data['PoolArea'] > 500)] = 'Fa'
bsmt_list = [col for col in data if 'Bsmt' in col]
bsmt_list
data.loc[(data.TotalBsmtSF != 0) & (data.BsmtCond.isnull())]
data.loc[(data.BsmtCond.isnull()) & ((data.index ==2040 ) | (data.index==2185) | (data.index ==2524))]
# Make some graphs to see if I can impute these by that information
fig= plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(221)
sns.boxplot(x='BsmtCond', y='BsmtFinSF1', data=data, ax=ax1)
ax2 = fig.add_subplot(222)
sns.boxplot(x='OverallCond', y='TotalBsmtSF', data=data, ax=ax2)
ax3 = fig.add_subplot(223)
sns.boxplot(x='BsmtFinType1', y='BsmtFinSF1', data=data, ax=ax3)
ax4 = fig.add_subplot(224)
sns.boxplot(x='BsmtExposure', y='TotalBsmtSF', data=data, ax=ax4)
# Comparing overallcond with bsmtcond
plt.figure(figsize=(16,18))
g = sns.lmplot( x="TotalBsmtSF", y="OverallCond", data=data, fit_reg=False, hue='BsmtCond', scatter_kws={"s": 50},height=10)
g.set(xlim=(700, 2500))
# Confirming the numbers
data.loc[(data.OverallCond ==5) & (data.BsmtCond == 'TA')]
data['BsmtCond'].loc[(data['BsmtCond'].isnull()) & ((data.BsmtFinSF1 ==1044 ) | (data['BsmtFinSF1']==1033) | (data['BsmtFinSF1'] ==755))] = 'TA'
data.loc[(data.TotalBsmtSF != 0) & (data.BsmtExposure.isnull())]
data.loc[(data['BsmtExposure'].isnull()) & ((data.index==948 ) | (data.index==1487) | (data.index ==2348))] 
plt.figure(figsize=(16,18))
g = sns.relplot(x="TotalBsmtSF", y="BsmtCond", hue="BsmtExposure",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=data)
g.set(xlim=(0, 2500))
# change height and use xlim if you need a closer look
g = sns.lmplot( x="TotalBsmtSF", y="OverallCond", data=data, fit_reg=False, hue='BsmtExposure', scatter_kws={"s": 50},height=5)
data['BsmtExposure'].loc[(data['BsmtExposure'].isnull()) & ((data.index==948 ) | (data.index==1487) | (data.index ==2348))] ='Av'
data.loc[(data['BsmtFinType2'].isnull()) & data['BsmtFinType1'].notnull()]
data.loc[(data.BsmtFinType2.isnull()) & (data.index ==332)]
g = sns.relplot(x="TotalBsmtSF", y="BsmtQual", hue="BsmtFinType2",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=data)
g.set(xlim=(0, 2500))
data['BsmtFinType2'].loc[(data.BsmtFinType2.isnull()) & (data.index ==332)] = 'Unf'
data.loc[(data.BsmtFullBath.isnull()) & (data.BsmtHalfBath.isnull())]
data.loc[(data.BsmtQual.isnull()) & (data.BsmtExposure.notnull())]
g = sns.relplot(x="TotalBsmtSF", y="BsmtExposure", hue="BsmtQual",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=data)
g.set(xlim=(100, 700))
data['BsmtQual'].loc[(data.BsmtQual.isnull()) & ((data.index ==2217) | (data.index==2218))] = 'TA'
data.isnull().sum()
data.loc[data.Electrical.isnull()]
# going to impute with most common, also since it was built in 2006, SBrkr is most likely what it is
data['Electrical'].loc[(data.index==1379)] = 'SBrkr'
data.loc[(data['Exterior1st'].isnull()) & (data['Exterior2nd'].isnull())]
#data.loc[(data['Exterior1st'] == data['Exterior2nd'])]
# this shows 2400+ rows where the 1st and 2nd exterior are the same
data.Exterior2nd.value_counts()
# change height and use xlim if you need a closer look
plt.figure(figsize=(16,8))
g = sns.boxplot( x="Exterior2nd", y="YearBuilt", data=data)
data['Exterior1st'].loc[(data.index==2151)] ='VinylSd'
data['Exterior2nd'].loc[(data.index==2151)] ='VinylSd'
data.loc[(data.Functional.isnull())]
# Documentation says assume typical, so we assume typical
data['Functional'].loc[(data.index==2216) | (data.index==2473)] = 'Typ'
garage_list= [col for col in data if 'Garage' in col]
garage_list
garage_data = data[garage_list]
garage_data.isnull().sum()
data.loc[(data.GarageType.notnull()) & (data.GarageCond.isnull())]
# No other information for garage at 2576 so Im going to change detchd to NaN for this row
data['GarageType'].loc[(data.index==2576)] = np.NaN
data.loc[(data.GarageType.notnull()) & (data.GarageCond.isnull())]
#data.loc[(data.GarageYrBlt == data.YearBuilt)]
# This showed 2200 rows where garage year built was equal to yearbuilt of the house so I will impute with that
# impute garageyrblt
data['GarageYrBlt'].loc[(data.index == 2126)] = data['YearBuilt'].loc[(data.index ==2126)]
#Impute garage qual with most common value, TA
data['GarageQual'].loc[(data.index == 2126)] = 'TA'
# impute GarageFinish with most common value, 'Unf'
data['GarageFinish'].loc[(data.index==2126)] = 'Unf'
#Impute GarageCond with most common value, TA
data['GarageCond'].loc[(data.index == 2126)] = 'TA'
data.loc[(data.KitchenQual.isnull())]
data.KitchenQual.value_counts()
# Imputing KitchenQual with most common, TA
data['KitchenQual'].loc[(data.index==1555)] ='TA'
data.isnull().sum()
data.loc[(data.MSZoning.isnull())]
# change height and use xlim if you need a closer look
g = sns.boxplot( x="MSZoning", y="LotArea", data=data)
g.set(ylim=(0,60000))
data['MSZoning'].loc[(data.index==1915)|(data.index==2216)| (data.index==2250)| (data.index==2904)] = 'RL'
data.loc[(data.MasVnrType.isnull()) &(data.MasVnrArea.notnull())]
data.MasVnrType.value_counts()
g = sns.boxplot( x="MasVnrType", y="MasVnrArea", data=data)
# Impute with 'BrkFace'
data['MasVnrType'].loc[(data.index==2610)] = 'BrkFace'
data.loc[(data.SaleType.isnull())]
data.SaleType.value_counts()
# Imputing with most common
data['SaleType'].loc[(data.index==2489)] = 'WD'
data.loc[(data.Utilities.isnull())]
data.Utilities.value_counts()
# They clearly have gas and electricity, no way to tell if they have all utilities but ill impute with allpub
data['Utilities'].loc[(data.index==1915) | (data.index==1945)] ='AllPub'
# checking to make sure we are ready to fill in na values with 0 or none
updated_list_of_na_columns = data.columns[data.isna().any()].tolist()
updated_list_of_na_columns
# First need to pull out SalePrice 
lot_frontage = data['LotFrontage']
del data['LotFrontage']
# Want to drop saleprice first so we can keep those as nans
target= data['SalePrice']
del data['SalePrice']
# List Comprehension to fill in the null values, if column is Object fillna with None, else fillna with 0
# assigning it to a random variable so 'none' doesn't get printed a million times
_ = [data[col].fillna('None', inplace=True) if (data[col].dtype=='O') else data[col].fillna(0, inplace=True) for col in data]
# and add back in so we can explore the data
data=data.join(target)
# Need to do try and except and also multiple kwargs
def plotly_plot(df, colx, coly, chart_type,**kwargs):
    #try:
        #print (go.chart_type)
        trace = chart_type(x=df[colx], y=df[coly], **kwargs)
        plot = [trace]
        layout = go.Layout(
                xaxis=dict(
                    title = colx,
                        titlefont=dict(
                            family='Courier New, monospace',
                                size=18,
                                    color='#000000'
                        )
                ),
                yaxis = dict(
                    title=coly,
                        titlefont=dict(
                            family='Courier New, monospace',
                                size=18,
                                    color='#000000'
                        )
                )
        )
        fig=dict(data=plot,layout=layout)
        return offline.iplot(fig)
    #except:
        #print('Please use (go.) before your chart_type of choice')
    
plotly_plot(data, 'OverallCond', 'SalePrice', go.Box)
plotly_plot(data, 'OverallQual', 'SalePrice', go.Box)
## Checking to see if there is a relationship between bedrooms(above ground) and sale price
plotly_plot(data, 'BedroomAbvGr', 'SalePrice', go.Box)
# what about Total Rooms?
plotly_plot(data, 'TotRmsAbvGrd', 'SalePrice', go.Box)
plotly_plot(data, 'TotRmsAbvGrd', 'GrLivArea', go.Box)
bed_bath_group = data.groupby('BedroomAbvGr', as_index=False)['FullBath'].agg('mean')
plotly_plot(bed_bath_group, 'BedroomAbvGr', 'FullBath',go.Scatter)
# setting up groupbys for the chart
max_bath_group = data.groupby('BedroomAbvGr', as_index=False)['FullBath'].agg('max')
avg_bath_group = data.groupby('BedroomAbvGr', as_index=False)['FullBath'].agg('mean')
min_bath_group = data.groupby('BedroomAbvGr', as_index=False)['FullBath'].agg('min')
# Create and style traces
trace0 = go.Scatter(
    x = max_bath_group['BedroomAbvGr'],
    y = max_bath_group['FullBath'],
    name = 'Max baths',
    line = dict(
        color = ('rgb(76, 153, 0)'),
        width = 4)
)
trace1 = go.Scatter(
    x = avg_bath_group['BedroomAbvGr'],
    y = avg_bath_group['FullBath'],
    name = 'Avg Baths',
    line = dict(
        color = ('rgb(22, 96, 167)'),
        width = 4,)
)
trace2 = go.Scatter(
    x = min_bath_group['BedroomAbvGr'],
    y = min_bath_group['FullBath'],
    name = 'Min baths',
    line = dict(
        color = ('rgb(205, 12, 24)'),
        width = 4,
        dash = 'dash') # dash options include 'dash', 'dot', and 'dashdot'
)

plot = [trace0, trace1, trace2]

# Edit the layout
layout = dict(title = 'Min, Avg, and Max bathrooms per bedrooms',
              xaxis = dict(title = 'Bedrooms(Above Ground)'),
              yaxis = dict(title = 'Baths'),
              )

fig = dict(data=plot, layout=layout)
iplot(fig)
data.loc[(data.FullBath ==0)]
data.loc[(data.FullBath==0) & (data.BsmtFullBath ==0)]
data.loc[(data.BedroomAbvGr==0)]
plotly_plot(data, 'FullBath', 'GrLivArea', go.Box)
plotly_plot(data, 'GrLivArea', 'SalePrice',go.Scatter, mode='markers')
plotly_plot(data, 'GrLivArea', 'LotArea', go.Scatter3d, mode='markers', z=data['SalePrice'])
qual_sf_group = data.groupby('OverallQual', as_index=False )['GrLivArea'].agg('mean')
plotly_plot(qual_sf_group, 'OverallQual', 'GrLivArea', go.Bar)
# Curious if there are any houses where the basement is larger than the 1stFloorSF, that would be weird?
plotly_plot(data, '1stFlrSF', 'TotalBsmtSF', go.Scatter, mode='markers')
# groupby to get average
month_sold_group = data.groupby('MoSold', as_index=False)['SalePrice'].agg('mean')
plotly_plot(month_sold_group, 'MoSold', 'SalePrice', go.Scatter, mode='lines')
# setting up groupbys for the chart
max_sold_group = data.groupby('MoSold', as_index=False)['SalePrice'].agg('max')
avg_sold_group = data.groupby('MoSold', as_index=False)['SalePrice'].agg('mean')
min_sold_group = data.groupby('MoSold', as_index=False)['SalePrice'].agg('min')
# Create and style traces
trace0 = go.Scatter(
    x = max_sold_group['MoSold'],
    y = max_sold_group['SalePrice'],
    name = 'Max Sale Price',
    line = dict(
        color = ('rgb(76, 153, 0)'),
        width = 4)
)
trace1 = go.Scatter(
    x = avg_sold_group['MoSold'],
    y = avg_sold_group['SalePrice'],
    name = 'Avg Sale Price',
    line = dict(
        color = ('rgb(22, 96, 167)'),
        width = 4,)
)
trace2 = go.Scatter(
    x = min_sold_group['MoSold'],
    y = min_sold_group['SalePrice'],
    name = 'Min Sale Price',
    line = dict(
        color = ('rgb(205, 12, 24)'),
        width = 4,
        dash = 'dash') # dash options include 'dash', 'dot', and 'dashdot'
)

plot = [trace0, trace1, trace2]

# Edit the layout
layout = dict(title = 'Min, Avg, and Max Sale Prices per month',
              xaxis = dict(title = 'Month'),
              yaxis = dict(title = 'Sale Price'),
              )

fig = dict(data=plot, layout=layout)
iplot(fig)
plotly_plot(data, 'YrSold', 'SalePrice', go.Box)
plotly_plot(data, 'Neighborhood', 'SalePrice', go.Box)
# creating a copy of my df for this chart
data_bubble = data.copy()
slope = 2.666051223553066e-05
hover_text = []
bubble_size = []
for index, row in data_bubble.iterrows():
    hover_text.append(('Neighborhood: {neighborhood}<br>'+
                      'Bedrooms: {bedrooms}<br>'+
                      'Quality: {quality}<br>'+
                      'Year Built: {yrbuilt}<br>'+
                      'Month Sold: {mosold}').format(neighborhood=row['Neighborhood'],
                                            bedrooms=row['BedroomAbvGr'],
                                            quality=row['OverallQual'],
                                            yrbuilt=row['YearBuilt'],
                                            mosold=row['MoSold']))
    bubble_size.append(math.sqrt(row['OverallQual']*slope))


data_bubble['text'] = hover_text
data_bubble['size'] = bubble_size
sizeref = 5.*max(data_bubble['size'])/(100**2)

trace0 = go.Scatter(
    x=data_bubble['GrLivArea'][data_bubble['YrSold'] == 2006],
    y=data_bubble['SalePrice'][data_bubble['YrSold'] == 2006],
    mode='markers',
    name='2006',
    text=data_bubble['text'][data_bubble['YrSold'] == 2006],
    marker=dict(
        symbol='circle',
        sizemode='area',
        sizeref=sizeref,
        size=data_bubble['size'][data_bubble['YrSold'] == 2006],
        line=dict(
            width=2
        ),
    )
)
trace1 = go.Scatter(
    x=data_bubble['GrLivArea'][data_bubble['YrSold'] == 2007],
    y=data_bubble['SalePrice'][data_bubble['YrSold'] == 2007],
    mode='markers',
    name='2007',
    text=data_bubble['text'][data_bubble['YrSold'] == 2007],
    marker=dict(
        symbol='circle',
        sizemode='area',
        sizeref=sizeref,
        size=data_bubble['size'][data_bubble['YrSold'] == 2007],
        line=dict(
            width=2
        ),
    )
)
trace2 = go.Scatter(
    x=data_bubble['GrLivArea'][data_bubble['YrSold'] == 2008],
    y=data_bubble['SalePrice'][data_bubble['YrSold'] == 2008],
    mode='markers',
    name='2008',
    text=data_bubble['text'][data_bubble['YrSold'] == 2008],
    marker=dict(
        symbol='circle',
        sizemode='area',
        sizeref=sizeref,
        size=data_bubble['size'][data_bubble['YrSold'] == 2008],
        line=dict(
            width=2
        ),
    )
)
trace3 = go.Scatter(
    x=data_bubble['GrLivArea'][data_bubble['YrSold'] == 2009],
    y=data_bubble['SalePrice'][data_bubble['YrSold'] == 2009],
    mode='markers',
    name='2009',
    text=data_bubble['text'][data_bubble['YrSold'] == 2009],
    marker=dict(
        symbol='circle',
        sizemode='area',
        sizeref=sizeref,
        size=data_bubble['size'][data_bubble['YrSold'] == 2009],
        line=dict(
            width=2
        ),
    )
)
trace4 = go.Scatter(
    x=data_bubble['GrLivArea'][data_bubble['YrSold'] == 2010],
    y=data_bubble['SalePrice'][data_bubble['YrSold'] == 2010],
    mode='markers',
    name='2010',
    text=data_bubble['text'][data_bubble['YrSold'] == 2010],
    marker=dict(
        symbol='circle',
        sizemode='area',
        sizeref=sizeref,
        size=data_bubble['size'][data_bubble['YrSold'] == 2010],
        line=dict(
            width=2
        ),
    )
)

plot = [trace0, trace1, trace2, trace3, trace4]
layout = go.Layout(
    title='Comparing Sales Price and Sq. ft of houses sold throughout the years',
    xaxis=dict(
        title='Sq Ft. Living Area',
        gridcolor='rgb(255, 255, 255)',
        range=[100,6000 ],
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    yaxis=dict(
        title='Sale Price',
        gridcolor='rgb(255, 255, 255)',
        range=[0, 900000],
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

fig = go.Figure(data=plot, layout=layout)
iplot(fig)
# if year built and remodadd are the same replace yearremodadd with 0. They should not have a value if theyve never been remodeled
data['YearRemodAdd']= np.where(data.YearRemodAdd == data.YearBuilt, 0, data.YearRemodAdd) 
# Change MoSold from int to a String so you can take dummies(based on chart above)
def turn_obj(cols):
    for col in cols:
        data[col] = data[col].astype(str)
turn_obj(['MoSold', 'YrSold', 'OverallCond', 'MSSubClass', 'GarageCars'])
data['TotalSF'] = data['GrLivArea'] + data['TotalBsmtSF']
data['totalSF_by_LotArea'] = data['TotalSF'] / data['LotArea']
# Need to drop SalePrice because it has NaNs and we dont want it in the algo to impute LotFrontage
# will create target again, just because
target = data['SalePrice']
missing_sales = data[data['SalePrice'].isnull()]
sub_id = missing_sales['Id']
# delete so its not used
#del data['Id']
data.drop(['SalePrice', 'Id'], axis=1,inplace=True)
# Bring in LotFrontage
data=data.join(lot_frontage)
data.groupby('Neighborhood')['LotFrontage'].agg('mean')
# Will first impute missing lotfrontage values with the mean based on the neighborhood
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
# 1 more piece of feature engineering
data['lotarea-frontage'] = data['LotFrontage'] / data['LotArea']
plt.figure(figsize=(16,8))
sns.heatmap(data.corr(), annot=True)
from sklearn import preprocessing
cols_for_label = ['BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Condition1', 
                  'Condition2', 'ExterQual', 'GarageCond', 'GarageQual', 'GarageType', 'KitchenQual', 'LotShape', 
                 'LotConfig', 'MiscFeature', 'PavedDrive', 'Functional', 'Fence', 'Alley', 'YearRemodAdd']
# loop to use labelencoder on the chosen columns
le = preprocessing.LabelEncoder()
for col in cols_for_label:
    le.fit(data[col])
    list(le.classes_)
    data[col] = le.transform(data[col])

target_no_nan = target.dropna()
# this code taken from another user here on kaggle. Great stuff thank you!
def check_skewness(df):
    sns.distplot(df, fit = norm);
    fig =plt.figure(figsize=(16,8))
    res = stats.probplot(df, plot=plt)
    # get fitted parameters used by the function
    (avg, std) = norm.fit(df)
    print ('\n avg = {:.2f} and std = {:.2f}\n' .format(avg, std))
check_skewness(target_no_nan)
target_no_nan = np.log1p(target_no_nan)

check_skewness(target_no_nan)
num_feats = data.dtypes[data.dtypes != 'object'].index
#check skew
skewed_feats = data[num_feats].apply(lambda x:skew(x)).sort_values(ascending=False)
skewness = pd.DataFrame({'sKew':skewed_feats})
#skewness = skewness.drop(['price'])
skewness.head()
# Boxcox fix skew
skewness = skewness[abs(skewness) > .75]
print (skewness.shape[0])
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = .15
for feat in skewed_features:
    data[feat] = boxcox1p(data[feat], lam)
data = data.join(target_no_nan)
def get_dummies(df):
    future_drop = [col for col in df if df[col].dtype == 'O']
    # I know get dummies only takes Objects but if I don't do the list comp inside it gives me a columns overlap error
    df = df.join(pd.get_dummies(df[[col for col in df if df[col].dtype == 'O']], drop_first=True)).drop(future_drop, axis=1) 
    return df
    #df.drop(future_drop, axis=1, inplace=True)
#data = get_dummies(data)
data=get_dummies(data)
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
missing_price = data[data['SalePrice'].isnull()]
filled_price = data[data['SalePrice'].notnull()]
X_train, X_test, y_train, y_test = train_test_split(filled_price.drop('SalePrice', axis=1),filled_price['SalePrice'], test_size=.2, random_state=42)
# StandardScaler was almost identical to robust but gave warnings
# FunctionTransformer helped but not too noticeable
# RobustScaler worked the best
gbr=  make_pipeline(RobustScaler(),GradientBoostingRegressor(n_estimators=800, learning_rate=0.05,
                                  max_depth=4, max_features='log2',
                                  min_samples_leaf=8, min_samples_split=6,
                                  loss='huber', random_state=42))
br = make_pipeline(RobustScaler(),BayesianRidge())
r = make_pipeline(RobustScaler(),Ridge())
xgb = make_pipeline(RobustScaler(),XGBRegressor())
svr =make_pipeline(RobustScaler(),SVR(kernel='linear'))
# Lasso and enet were way off until I messed with alpha,possibly fine tuning will bring a better scores
l = make_pipeline(RobustScaler(),Lasso(alpha=.0005))
enet = make_pipeline(RobustScaler(), ElasticNet(alpha=.001))
# going to put cross_val in the tdmassess
def cv_score(algo):
    rmse= np.sqrt(-cross_val_score(algo, X_train, y_train, scoring='neg_mean_squared_error', cv=5))
    return (rmse.mean())
algorithms = [gbr, br, r, xgb, svr,l, enet]
names = ['Gradient Boosting', 'Bayesian Ridge', 'Ridge', 'XGB', 'SVR', 'Lasso','ElasticNet']
def tDMassess_regression():
    #fit the data
    for i in range(len(algorithms)):
        algorithms[i] = algorithms[i].fit(X_train,y_train)
    cv_rmse =[]
    rmse_train=[]
    rmse_test=[]
    for i in range(len(algorithms)):
        rmse_train.append(mean_squared_error(np.expm1(y_train), np.expm1(algorithms[i].predict(X_train))) **.5)
        rmse_test.append(mean_squared_error(np.expm1(y_test), np.expm1(algorithms[i].predict(X_test)))**.5)
        cv_rmse.append(cv_score(algorithms[i]))
    metrics = pd.DataFrame(columns =['RMSE_train', 'RMSE_test', 'cv_RMSE'], index=names)
    metrics['RMSE_train'] = rmse_train
    metrics['RMSE_test'] = rmse_test
    metrics['cv_RMSE'] = cv_rmse
    return metrics
tDMassess_regression()
final_algs = [gbr, br, l,enet]
def average_of_models():
    final_pred=[]
    for i in range(len(final_algs)):
         final_pred.append(np.expm1(final_algs[i].predict(missing_price.drop('SalePrice', axis=1))))
    return (sum(final_pred)/len(final_algs))
avg_preds = average_of_models()
submission = pd.DataFrame()
submission['Id'] =sub_id
submission['SalePrice'] = avg_preds
submission.to_csv('final_sub_LE_la_enet.csv', index=False)