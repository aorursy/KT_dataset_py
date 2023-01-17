import pandas as pd

import numpy as np

from scipy import stats

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.figure_factory as ff

from plotly import tools

py.init_notebook_mode(connected = False)

import plotly.graph_objs as go
traindf = pd.read_csv('../input/train.csv')

testdf = pd.read_csv('../input/test.csv')

print('The size of traind data is:{}'.format(traindf.shape) ,'\n','The size of test data is:{}'.format(testdf.shape) )

traindf.head()
print('The dataset contains: {}'.format(traindf.shape[0]), 'rows and {} columns.'.format(traindf.shape[1]))

traindf.columns
nulls = traindf.isnull().sum()

nulls[nulls > 1000]
n_unique = []



for column in traindf:

    n_unique.append(len(traindf[column].unique()))



#unique_features_len = pd.DataFrame(n_unique, index = traindf.columns, columns = ['Unique'])



unique_features_len = pd.Series(n_unique , index = traindf.columns)



trace = go.Scatter(

        x = unique_features_len.values,

        y = unique_features_len.index,

        mode = 'markers',

        marker = dict(

                size = '14',

                color = np.random.randn(78))

        )



layout = go.Layout(

    

        title = 'Unique Features',

        xaxis = dict(

                title = 'Count',),

        )

fig = go.Figure(data = [trace], layout = layout)

py.iplot(fig)
missingA = pd.Series(traindf.isnull().sum())

missingA = missingA[missingA > 0]



missingB = pd.Series(testdf.isnull().sum())

missingB = missingB[missingB > 0]



trace = go.Scatter(

        x = missingA.values,

        y = missingA.index,

        mode = 'markers',

        marker = dict(

                size = '14',

                color = np.random.randn(len(missingA))),

        name = 'Train data'

        )



trace1 = go.Scatter(

        x = missingB.values,

        y = missingB.index,

        mode = 'markers',

        marker = dict(

                size = '14',

                color = 'black'),

        opacity = '0.5',

        name = 'Test data'

        )



layout = go.Layout(

        title = 'Missing Feature Values',

        xaxis = dict(

                title = 'Count')

)



fig = go.Figure(data = [trace, trace1], layout = layout)

py.iplot(fig)
def bestfitline(x_param, y_param):

    #Extracting the regression line from seaborn to plot on iplotly.

    rg = sns.regplot(x = x_param, y = y_param, data = traindf)

    #Getting X co-ordinates of that line.

    X = rg.get_lines()[0].get_xdata()

    #Getting Y co-ordinates of that line.

    Y = rg.get_lines()[0].get_ydata()

    #This following line of code gets the shaded region around the regression line to plot.

    P = rg.get_children()[1].get_paths()

    plt.close()

    return X, Y
fig = tools.make_subplots(rows = 2, cols = 2, subplot_titles = ('1. GrLiv Area Vs. Sale Price', '2. Lot Area Vs. Sale Price',

                                                               '3. Total Basement Vs. Sale Price', '4. Garage Area Vs. Sale Price'))

p1 = go.Scatter(

        x = traindf['GrLivArea'],

        y = traindf['SalePrice'],

        mode = 'markers',

        name = 'Scatter',

        showlegend = True

        )



fig.append_trace(p1,1, 1)



X, Y = bestfitline('GrLivArea', 'SalePrice')



p2 = go.Scatter(

        x = X,

        y = Y,

        name = 'fit',

        showlegend = False

        )

fig.append_trace(p2,1,1)



p3 = go.Scatter(

        x = np.linspace(4620, 4620),

        y = np.linspace(0, 800000),

        name = 'Boundary',

        showlegend = False,

        line = dict(

                color = 'black'),

        ) 

fig.append_trace(p3,1,1)



#------------------------------------------------------------#



p4 = go.Scatter(

        x = traindf['LotArea'],

        y = traindf['SalePrice'],

        mode = 'markers',

        name = 'Scatter',

        showlegend = True

        )

fig.append_trace(p4, 1, 2)



X, Y = bestfitline('LotArea', 'SalePrice')



p5 = go.Scatter(

        x = X,

        y = Y,

        name = 'fit',

        showlegend = False

        )



fig.append_trace(p5, 1, 2)



p6 = go.Scatter(

        x = np.linspace(150000,150000),

        y = np.linspace(0, 800000),

        name = 'Boundary',

        line = dict(

                color = 'black'

                ),

        showlegend = False

        )

fig.append_trace(p6, 1 ,2)



#------------------------------------------------------------#



p7 = go.Scatter(

        x = traindf['TotalBsmtSF'],

        y = traindf['SalePrice'],

        mode = 'markers',

        showlegend = True,

        name = 'Scatter',

        )

fig.append_trace(p7, 2, 1)



X, Y = bestfitline('TotalBsmtSF', 'SalePrice')



p8 = go.Scatter(

        x = X,

        y = Y,

        name = 'fit',

        showlegend = False

        )



fig.append_trace(p8, 2, 1)



p9 = go.Scatter(

        x = np.linspace(5000,5000),

        y = np.linspace(0, 800000),

        name = 'Boundary',

        showlegend = False,

        line = dict(

                color = 'black')

        )



fig.append_trace(p9, 2, 1)



#------------------------------------------------------------#



p10 = go.Scatter(

        x = traindf['GarageArea'],

        y = traindf['SalePrice'],

        mode = 'markers',

        showlegend = True,

        name = 'Scatter',)



fig.append_trace(p10, 2, 2)



X, Y = bestfitline('GarageArea', 'SalePrice') 



p11 = go.Scatter(

        x = X,

        y = Y,

        name = 'fit',

        showlegend = False

        )



fig.append_trace(p11,2, 2)



fig['layout'].update(height = 800, width = 800)

py.iplot(fig)
traindf.drop(traindf.loc[(traindf['GrLivArea'] > 4500) & (traindf['SalePrice'] < 300000 )].index, axis = 0, inplace = True)

traindf.drop(traindf.loc[(traindf['LotArea'] > 150000) & (traindf['SalePrice'] < 400000)].index, axis = 0, inplace = True)
temp1 = traindf.groupby(['YearBuilt'])['SalePrice'].sum()*100/traindf['SalePrice'].sum()

temp1 = pd.Series(temp1)



temp2 = traindf.groupby('YrSold')['SalePrice'].sum()*100/traindf['SalePrice'].sum()

temp2 = pd.Series(temp2)



temp3 = traindf.groupby('YearRemodAdd')['SalePrice'].sum()*100/traindf['SalePrice'].sum()

temp3 = pd.Series(temp3)



temp4 = traindf.groupby('GarageYrBlt')['SalePrice'].sum()*100/traindf['SalePrice'].sum()

temp4 = pd.Series(temp4)



trace1 = go.Scatter(

        x = temp1.index,

        y = temp1.values,

        mode = 'lines+markers',

        name = 'Year Built')



trace2 = go.Scatter(

        x = temp2.index,

        y = temp2.values,

        mode = 'markers+lines',

        name = 'Year Sold'

        )

trace3 = go.Scatter(

        x = temp3.index,

        y = temp3.values,

        mode = 'markers+lines',

        name = 'Year of Remodel',

        opacity = 0.5)



trace4 = go.Scatter(

        x = temp4.index,

        y = temp4.values,

        mode = 'markers+lines',

        name = 'Year of Garage Built',

        opacity = 0.5

)



layout = go.Layout(

        title = 'Percentage change',

        xaxis = dict(

                title = 'Year->'),

        yaxis = dict(

                title = 'Percentage(%)'))



data = [trace1, trace2, trace3, trace4]

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
from scipy.stats import kurtosis, skew



trace = go.Histogram(

        x = traindf['SalePrice'],

        histnorm = 'density',

        )



layout = go.Layout(

        title = 'Sale Price Distribution',

        )

data = [trace]

fig = go.Figure(data =data, layout = layout)

py.iplot(fig )
print('The above Histogram has skew: {} and kurtosis: {}'.format(skew(traindf['SalePrice']), kurtosis(traindf['SalePrice'])))
traindf['SalePrice'] = np.log1p(traindf['SalePrice'])

data = go.Histogram(

        x = traindf['SalePrice'],

        histnorm = 'density'

        )

layout = go.Layout(

        title = 'Distribution of Sales Price (log Trasnformed)',

        )



fig = go.Figure(data = [data], layout = layout)

py.iplot(fig )
trace = go.Histogram(

        x = traindf['Neighborhood'],

        marker = dict(

                color = 'red'

                ),

        )

layout = go.Layout(

        title = 'Frequency of Neighbors',

        xaxis = dict(

                title = 'Names',),

        yaxis = dict(

                title = 'Frequency'))



fig = go.Figure(data = [trace], layout = layout)



py.iplot(fig)
cols = traindf.columns

cols = cols[cols.str.contains('Bath')]



sns.pairplot(traindf[cols],)
corr = traindf.corr()



mask = np.zeros_like(corr, dtype = np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap = True)

f, ax = plt.subplots(figsize=(14, 10))

ax = sns.heatmap(corr, mask = mask, cmap = cmap, vmax = .3,square = True,

                 linewidths = .5, cbar_kws = {"shrink": .5}, ax = ax)



ax.axhline(y = 0, color = 'k',linewidth = 10)

ax.axhline(y = corr.shape[1], color = 'k',linewidth = 10)

ax.axvline(x = 0, color = 'k',linewidth = 10)

ax.axvline(x = corr.shape[0], color = 'k',linewidth = 10)

plt.show()

# https://stackoverflow.com/questions/36560829/seaborn-heatmap-with-frames
def fulldf(data1, data2):

    fulldf = pd.concat([data1, data2])

    return fulldf



def sepdf(data):

    data1 = fulldf.iloc[:len(traindf)]

    data2 = fulldf.iloc[len(traindf):]

    return data1, data2
target = traindf['SalePrice']

Id = testdf['Id']

fulldf = fulldf(traindf.drop('SalePrice', axis = 1), testdf)
# Selecting dtypes as object for categorical feature extraction.

features = fulldf.select_dtypes(include = ['object'])

f = pd.DataFrame()

# Loop to append all the categorical features of a column into one column.

for fe in features.columns:

    l = 0 # initialize l = 0 for every iteration.

    l = []

    l.append(fulldf[fe].unique())

    f[fe] = l # Now every single column name has all the features in one row, we don't want that.



# Creating a dict to store all the seperated features into a new row.

features_dict = {}

# iterating over columns.

for column in f.columns:

    new_column = 0 # Initilizing it as 0 for every iterations.

    new_column = pd.Series(f[column][0], name = column,) # storing feature in separate rows.

    keys = 0 

    keys = range(len(new_column)) # iterate over each feature length as not all columns has same number of features.

    for i in keys:

        features_dict[column] = new_column

# note: the Series.unique() function took nans as unique as well.

unique_feature_data = pd.DataFrame(features_dict)

unique_feature_data.fillna(' ', inplace = True)

pd.set_option('max_columns',None)

print('                                    Unique Categorical Features')

unique_feature_data.head()
fulldf[['PoolQC','PoolArea',

        'GarageQual','GarageArea',

        'FireplaceQu','Fireplaces',

        'BsmtQual','TotalBsmtSF']].sort_values(by = ['PoolQC','GarageQual','FireplaceQu','BsmtQual','TotalBsmtSF'], na_position = 'first').head()
Qual = ['PoolQC', 'GarageQual', 'FireplaceQu', 'BsmtQual',]



for Q in Qual:

    fulldf[Q].fillna(0, inplace = True)
missing = fulldf.isnull().sum()

missing = missing[missing > 0]



Smallmissing = missing[missing < 20]



for column in Smallmissing.index:

    fulldf[column].fillna(fulldf[column].mode()[0], inplace = True)
fulldf[['GarageYrBlt', 'GarageType', 'GarageCond', 'GarageFinish', 'GarageArea']].sort_values(by = ['GarageYrBlt', 'GarageType',

                                                                                                    'GarageCond', 'GarageFinish'], na_position = 'first').head(3)
Garage = ['GarageYrBlt', 'GarageType', 'GarageCond', 'GarageFinish']



for G in Garage:

    fulldf[G].fillna(0, inplace = True)
fulldf[['BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtExposure', 'TotalBsmtSF']].sort_values(

by = ['BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtExposure'], na_position = 'first').head(3)
Bsmt = ['BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtExposure',]



for B in Bsmt:

    fulldf[B].fillna('None', inplace = True)
k = ['LotFrontage','MasVnrType', 'MasVnrArea', 'Fence', 'MiscFeature']

for K in k:

    fulldf.loc[:,K].fillna(fulldf[K].mode()[0], inplace = True)
unique_feature_data.loc[:,['Alley', 'Street']].head(3)
fulldf.loc[:,['Alley', 'Street']].sort_values(by = 'Alley', na_position = 'first').head(3)
fulldf.drop(['Id', 'Alley'], axis = 1, inplace = True)
ordinal_fea =  ['BsmtCond', 'BsmtQual', 'ExterCond', 'ExterQual', 'FireplaceQu', 'GarageCond', 'GarageQual', 'HeatingQC','KitchenQual', 'PoolQC']

for i in ordinal_fea:

    fulldf[i] = fulldf[i].replace(['Ex','Fa','Gd','TA','Po','None'],[5, 4, 3, 2, 1, 0])
num_fea = fulldf.select_dtypes(exclude = ['object']).columns

skewed_fea = fulldf[num_fea].apply(lambda x: skew(x))

skewed_col_names = skewed_fea.index

skewed_col_names = skewed_col_names.str.contains('(Area)|(SF)|(Frontage)')



for col_name in skewed_fea[skewed_col_names].index:

    fulldf[col_name] = np.log1p(fulldf[col_name])
fulldf['HouseAge'] = fulldf['YearRemodAdd'] - fulldf['YearBuilt']

fulldf['CompleteHouseArea'] = fulldf['LotArea'] + fulldf['MasVnrArea'] + fulldf['GrLivArea'] + fulldf['GarageArea'] + fulldf['PoolArea'] + fulldf['TotalBsmtSF']

                                                               # GrLivArea = 1stFlrSF + 2ndFlrSF
num_cat_fea = list((((set(num_fea)) - set(ordinal_fea)) - set(skewed_fea[skewed_col_names].index )))

from sklearn.preprocessing import LabelEncoder



Le = LabelEncoder()

for col_name in num_cat_fea:

    fulldf[col_name] = Le.fit_transform(fulldf[col_name])
cat_fea = list(set(unique_feature_data.columns[1:]) - set(num_fea))

print('The shapes of train and test Dataframes before conversion is respectively: ({}), ({})'.format(traindf.shape, testdf.shape))

fulldf = pd.get_dummies(fulldf, columns = cat_fea, drop_first = True)

traindf, testdf = sepdf(fulldf)

print('The shapes of train and test Dataframes After conversion is respectively: ({}), ({})'.format(traindf.shape, testdf.shape))
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.linear_model import Lasso, Ridge, ElasticNet

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.pipeline import make_pipeline

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
scorer = make_scorer(mean_squared_error, greater_is_better = False)

def RMSE_CV(model):

    RMSE = np.sqrt(-cross_val_score(model, traindf, target, scoring = scorer, cv = 5))

    return RMSE
alphas = [10, 15, 20, 30, 32, 35, 50, 75]

ridge_cv = []

for alpha in alphas:

    ridge = make_pipeline(StandardScaler(), RobustScaler(), Ridge(alpha = alpha))

    ridge_cv.append(RMSE_CV(ridge).mean())

ridge_cv = pd.Series(ridge_cv, index = alphas)

print('The minimum error is {:.4f} at {:.4f}'.format(ridge_cv.min(), ridge_cv.idxmin()))
alphas = [0.1, 0.15, 0.2, 0.005, 0.0032, 0.0030, 0.0024,]

lasso_cv = []

for alpha in alphas:

    

    lasso = make_pipeline(StandardScaler(), RobustScaler(), Lasso(alpha = alpha))

    lasso_cv.append(RMSE_CV(lasso).mean())

lasso_cv = pd.Series(lasso_cv, index = alphas)

print('The minimum error is {:.4f} at {:.4f}'.format(lasso_cv.min(), lasso_cv.idxmin()))
alphas = [0.1, 0.15, 0.2, 0.005, 0.0032, 0.0030, 0.0024, 0.0028]

ENet_cv = []

for alpha in alphas:

    ENet = make_pipeline(StandardScaler(), RobustScaler(), ElasticNet(alpha = alpha, l1_ratio = 0.9))

    ENet_cv.append(RMSE_CV(ENet).mean())

ENet_cv = pd.Series(lasso_cv, index = alphas)

print('The minimum error is {:.4f} at {:.4f}'.format(ENet_cv.min(), ENet_cv.idxmin()))
ridge = make_pipeline(StandardScaler(), RobustScaler(), Ridge(alpha = ridge_cv.idxmin()))

lasso = make_pipeline(StandardScaler(), RobustScaler(), Lasso(alpha = lasso_cv.idxmin()))

ENet = make_pipeline(StandardScaler(), RobustScaler(), ElasticNet(alpha = ENet_cv.idxmin(), l1_ratio = 0.9))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        for model in self.models_:

            model.fit(X, y)



        return self

    

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)
averaged_models = AveragingModels(models = (ENet, ridge, lasso))



score = RMSE_CV(averaged_models)

print('Averaged base models score: {:.4f}'.format(score.mean()))
averaged_models.fit(traindf, target)

pred = averaged_models.predict(testdf)

pred = np.expm1(pred)

sub = pd.DataFrame({'Id': Id, 'SalePrice': pred})

sub.to_csv('submission.csv',index = False)