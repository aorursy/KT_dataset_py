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
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')





from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer 

from sklearn.impute import SimpleImputer

from sklearn.ensemble import IsolationForest





from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

import sklearn.metrics as sm

from sklearn.model_selection import cross_val_score



from xgboost import XGBRegressor



#Use this code to show all the columns

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



plt.style.use('seaborn')

sns.set_style('whitegrid')





def customPlotGraph( data, plot_type, labels, colors, dimension, callback_function):

    

    global plt

    fig, ax = plt.subplots(figsize=(dimension['width'],dimension['height']))



    if plot_type == "scatter" :

        plt.scatter(x=data['x'], y=data['y'], color=data['color'], alpha=data['alpha'])

    if plot_type == "bar" :

        ax.bar(data['labels'], data['values'], color=colors['color'], edgecolor=colors['edgecolor'] )

    if plot_type == "barh" :

        ax.barh(data['labels'], data['values'], color=colors['color'], edgecolor=colors['edgecolor'] )

        for i in ax.patches:

            ax.text(i.get_width()+1, i.get_y()+0.5, str(round((i.get_width()), 2)), fontsize=10, fontweight='bold', color='grey')

    if plot_type == "heatmap" :

        sns.heatmap( data['values'], cmap=colors['color'])

    

    

    if 'graph_title' in labels:

        plt.title(labels['graph_title']['text'], fontsize=labels['graph_title']['fontsize'], weight=labels['graph_title']['weight'] )

    if 'xlabel' in labels:

        plt.xlabel(labels['xlabel']['text'], size=labels['xlabel']['fontsize'], weight=labels['xlabel']['weight'])

    if 'ylabel' in labels:

        plt.ylabel(labels['ylabel']['text'], size=labels['ylabel']['fontsize'], weight=labels['ylabel']['weight'])

    if 'xticks' in labels:

        plt.xticks(weight =labels['xticks']['weight'])

    if 'yticks' in labels:

        plt.yticks(weight =labels['yticks']['weight'])

    

    if callback_function and callback_function.strip() :

        plt = eval(callback_function)(data, plot_type, labels, colors, dimension, plt)

    

    

    return plt.show()

df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



train = df.drop(['SalePrice'], axis=1)

y = df.SalePrice



#train.drop(['SalePrice'], axis=1, inplace=True)

train.drop(['Id'], axis=1, inplace=True)

test = df_test.drop(['Id'], axis=1)





train.head(5)
train.shape
def customHeatMap(data, plot_type, labels, colors, dimension, returnPlt):

    returnPlt.yticks(weight =labels['yticks']['weight'], color=labels['yticks']['color'], rotation=labels['yticks']['rotation'] )

    return returnPlt



    

labels = { 

            'graph_title': {

                        'text': 'Numerical features correlation with the sale price',

                        'fontsize': '18',

                        'weight': 'bold'

                        } ,

            'xticks': {

                    'weight': 'bold'

                },

            'yticks': {

                    'weight': 'bold',

                    'color': 'dodgerblue',

                    'rotation': '0'

                }

         }



num=df.select_dtypes(exclude='object')

numcorr=num.corr()

data = { 'values': numcorr.sort_values(by=['SalePrice'], ascending=False).head(1) }

colors = { 'color':'Greens', 'edgecolor':'black' }

dimension = { 'width': 15, 'height' : 6}

customPlotGraph( data, "heatmap", labels, colors, dimension, 'customHeatMap' )







Num=numcorr['SalePrice'].sort_values(ascending=False).head(10).to_frame()

cm = sns.light_palette("Green", as_cmap=True)

s = Num.style.background_gradient(cmap=cm)

s


def getMissingValuePercentColumnWise(data):

    train_null_percent = data.isnull().sum()/len(data)*100 

    train_null_percent = train_null_percent.drop(train_null_percent[train_null_percent == 0].index).sort_values(ascending=False)

    return train_null_percent



    

def dropColumnHavingHigherNullPercent(train_null_percent, data):

    print( 'Columns {columns} identified with null values > 80%. Will be droping these columns '.format(columns=train_null_percent[train_null_percent > 80].index) )    

    #if all([item in train_null_percent[train_null_percent > 80].index for item in data.columns]): 

    #if 'PoolQC' in data.columns: 

    if  train_null_percent[train_null_percent > 80].index.isin(data.columns).all():

        print( 'Droping  columns')    

        data.drop(train_null_percent[train_null_percent > 80].index, axis=1,inplace=True)

        test.drop(train_null_percent[train_null_percent > 80].index, axis=1,inplace=True)

    return data

    
train_null_percent = getMissingValuePercentColumnWise(train)





labels = { 

            'graph_title': {

                        'text': 'Missing values percent column wise',

                        'fontsize': '18',

                        'weight': 'bold'

                        } ,

            'xlabel': {

                    'text': 'Percentage',

                     'weight': 'bold',

                     'fontsize': '13'

                },

            'ylabel': {

                    'text': 'Features',

                     'weight': 'bold',

                     'fontsize': '13'

                },

            'xticks': {

                    'weight': 'bold'

                },

            'yticks': {

                    'weight': 'bold',

                    'color': 'dodgerblue',

                    'rotation': '0'

                }

         }



data = { 'values': train_null_percent.values, 'labels' : train_null_percent.index }

colors = { 'color':'red', 'edgecolor':'black' }

dimension = { 'width': 15, 'height' : 6}

customPlotGraph( data, "barh", labels, colors, dimension, '' )





train_null_percent.to_frame().T
train = dropColumnHavingHigherNullPercent(train_null_percent, train)

train.head()
def getMissingValueByDataType(data):    

    NAcat = data.select_dtypes(include='object').isnull().sum()

    NAnum = data.select_dtypes(exclude='object').isnull().sum()

    

    return NAcat[NAcat > 0], NAnum[NAnum > 0]



    

NAcat, NAnum = getMissingValueByDataType(train)    

print('We have {cat_column_name} categorical features with missing values'.format( cat_column_name = NAcat.index.to_list()))

print('We have {num_column_name} numerical features with missing values'.format( num_column_name = NAnum.index.to_list()))

NAcat.to_frame().sort_values(by=[0]).T
NAnum.to_frame().sort_values(by=[0]).T
#SimpleImputer is for missing values, it follows different stratergies like mean, median, most_frequent & constant

#StandardScaler : idea behind StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1.

#StandardScaler & SimpleImputer anf than remove outliners



numerical_ix = train.select_dtypes(include=['int64', 'float64']).columns

categorical_ix = train.select_dtypes(include=['object', 'bool']).columns



numeric_transformer = make_pipeline(

        SimpleImputer(strategy='mean'),

        StandardScaler()

)



categorical_transformer = make_pipeline(

        SimpleImputer(strategy='constant', fill_value='None'),

        OneHotEncoder(handle_unknown='ignore')

    )



t = [

        ('num', numeric_transformer , numerical_ix),

        ('cat', categorical_transformer, categorical_ix) 

    ]

    

def getEncodingColumnTransformer():

    col_transform = ColumnTransformer(transformers=t)

    return col_transform

    

    
def getOutinerRemovedFromData(data, y):

    iso = IsolationForest(contamination=0.1)

    iso_model_pipeline =  make_pipeline(getEncodingColumnTransformer(),  iso )

    yhat = iso_model_pipeline.fit_predict(data)

    mask = yhat != -1 #-1 is the outliner

    outline_mask = yhat == -1 #-1 is the outliner

    x_train = train.iloc[mask, :]

    y_train = y[mask]

    x_outlined = train.iloc[mask, :] #data which is removed

    y_outlined = y[mask] #data which is removed

    

    return x_train, y_train, x_outlined, y_outlined

x_train , y_train, x_outlined, y_outlined = getOutinerRemovedFromData(train,y)



skew_kurt_comparison = pd.DataFrame.from_dict( {

                            'Labels' : train.skew().index,

                            'Pre Transformation Skew' : train.skew().values,

                            'Post Transformation Skew' : x_train.skew().values,

                            'Pre Transformation Kurt' : train.kurt().values,

                            'Post Transformation Kurt' : x_train.kurt().values    

                    } )



skew_kurt_comparison
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_train,y_train)
regressor = RandomForestRegressor( n_estimators = 15, random_state = 0)

RFModel =  make_pipeline( getEncodingColumnTransformer(), regressor )

RFModel.fit(x_train, y_train)

score = RFModel.score(x_test, y_test)



y_pred = RFModel.predict(x_test) 

#y_pred = RFModel.predict(test) 



scores = cross_val_score(RFModel, x_train, y_train, cv=5, n_jobs=-1)



##For Regression models

print("Training score: ", score)

print("Mean cross-validation score: %.2f" % scores.mean())

print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2)) 

print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2)) 

print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2)) 

print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2)) 

print("R2 score =", round(sm.r2_score(y_test, y_pred), 2))

## //For Regression models

xgbr =  make_pipeline( getEncodingColumnTransformer(), XGBRegressor(verbosity=0, n_estimators  = 750,learning_rate = 0.02, max_depth = 4) )

xgbr.fit(x_train, y_train)



score = xgbr.score(x_train, y_train)  

scores = cross_val_score(xgbr, x_train, y_train,cv=10)



#kfold = KFold(n_splits=10, shuffle=True)

#kf_cv_scores = cross_val_score(xgbr, x_train, y_train, cv=5 )

#print("K-fold CV average score: %.2f" % kf_cv_scores.mean())





y_pred = xgbr.predict(x_test)

#y_pred2 = xgbr.predict(x_test)

#mse = mean_squared_error(y_test, y_pred)

#print("MSE: %.2f" % mse)



##For Regression models

print("Training score: ", score)

print("Mean cross-validation score: %.2f" % scores.mean())

print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2)) 

print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2)) 

print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2)) 

print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2)) 

print("R2 score =", round(sm.r2_score(y_test, y_pred), 2))

## //For Regression models



final_submission = pd.DataFrame({

        "Id": df_test["Id"],

        "SalePrice": xgbr.predict(test)

    })

final_submission.to_csv('final_submission_ml_randomforest_xgboost.csv', index=False)

final_submission.head()