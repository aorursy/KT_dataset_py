import pandas as pd

import math

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



import datetime              

from tabulate              import tabulate

from scipy.stats           import chi2_contingency

from IPython.display       import Image

from IPython.core.display  import HTML



from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder

from boruta                import BorutaPy

from sklearn.ensemble      import RandomForestRegressor

from sklearn.metrics       import mean_absolute_error, mean_squared_error

from sklearn.linear_model  import LinearRegression, Lasso

from sklearn.ensemble      import RandomForestRegressor

import xgboost as xgb



import random

import warnings

warnings.filterwarnings( 'ignore' )



import pickle

from flask                 import Flask, request, Response

import simplejson as json

import requests
def cross_validation( X_training, kfold, model_name, model, verbose=False ):

    

    mae_list = []

    mape_list = []

    rmse_list = []



    for k in reversed( range( 1, kfold+1 ) ):

        if verbose:

            print( '\nKFold Number: {}'.format( k ) )

        # start and end date for validation

        validation_start_date = X_training['date'].max() - datetime.timedelta( days=k*6*7 )

        validation_end_date = X_training['date'].max() - datetime.timedelta( days=(k-1)*6*7 )



        # filtering dataset

        training = X_training[X_training['date'] < validation_start_date]

        validation = X_training[(X_training['date'] >= validation_start_date) & (X_training['date'] <= validation_end_date)]



        # training and validation dataset

        # training

        xtraining = training.drop( ['date', 'sales' ], axis=1 )

        ytraining = training['sales']



        #validation

        xvalidation = validation.drop( ['date', 'sales'], axis=1 )

        yvalidation = validation['sales']



        # model

        m = model.fit( xtraining, ytraining )



        # prediction

        yhat = m.predict( xvalidation )



        # performance

        m_result = ml_error( model_name, np.expm1( yvalidation ), np.expm1( yhat ) )



        # store performance of each kfold interation

        mae_list.append( m_result['MAE'] )

        mape_list.append( m_result['MAPE'] )

        rmse_list.append( m_result['RMSE'] )



    return pd.DataFrame( { 'Model Name': model_name,

                            'MAE CV': np.round( np.mean( mae_list ), 2 ).astype( str ) + ' +/- ' + np.round( np.std( mae_list ), 2 ).astype( str ),

                            'MAPE CV': np.round( np.mean( mape_list ), 2 ).astype( str ) + ' +/- ' + np.round( np.std( mape_list ), 2 ).astype( str ),

                            'RMSE CV': np.round( np.mean( rmse_list ), 2 ).astype( str ) + ' +/- ' + np.round( np.std( rmse_list ), 2 ).astype( str ) }, index=[0] )





def mean_absolute_percentage_error( y, yhat ):

    return np.mean( np.abs( (y - yhat ) / y ) )





def ml_error( model_name, y, yhat ):

    mae = mean_absolute_error( y, yhat )

    mape = mean_absolute_percentage_error( y, yhat )

    rmse = np.sqrt( mean_squared_error( y, yhat ) )

    

    return pd.DataFrame( { 'Model Name': model_name,

                           'MAE': mae,

                           'MAPE': mape,

                           'RMSE': rmse }, index=[0] )





def mean_percentage_error( y, yhat ):

    return np.mean( ( y - yhat ) / y )



def cramer_v( x, y ):

    cm = pd.crosstab( x, y ).values

    n = cm.sum()

    r, k = cm.shape

    

    chi2 = chi2_contingency( cm )[0]

    chi2corr = max( 0, chi2 - (k-1)*(r-1)/(n-1) )

    

    kcorr = k - (k-1)**2/(n-1)

    rcorr = r - (r-1)**2/(n-1)



    return np.sqrt( (chi2corr/n) / ( min( kcorr-1, rcorr-1 ) ) )



def jupyter_settings():

    %matplotlib inline

    %pylab inline

    

    plt.style.use( 'bmh' )

    plt.rcParams['figure.figsize'] = [25, 12]

    plt.rcParams['font.size'] = 24

    

    display( HTML( '<style>.container { width:100% !important; }</style>') )

    pd.options.display.max_columns = None

    pd.options.display.max_rows = None

    pd.set_option( 'display.expand_frame_repr', False )

    

    sns.set()
jupyter_settings()
df_sales_raw = pd.read_csv('../input/rossmann-store-sales/train.csv', low_memory=False)

df_store_raw = pd.read_csv('../input/rossmann-store-sales/store.csv', low_memory=False)



df_raw = pd.merge( df_sales_raw, df_store_raw, how='left', on='Store')
# check if the merge is correct.

df_raw.sample()
# At the beginning of each session make a copy of the dataset to help in case something goes wrong and we need to restart the process.

df1 = df_raw.copy()
# Rename the columns help us to unsderstand better what kind of info there is in each column.

df1.columns
# save the old name of the columns for security.

cols_old = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 

            'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 

            'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']





# rename.

df1.columns = ['store', 'day_of_week', 'date', 'sales', 'customers', 'open', 'promo', 'state_holiday', 'school_holiday', 'store_type', 

            'assortment', 'competition_distance', 'competition_open_since_month', 'competition_open_since_year', 'promo2', 

            'promo2_since_week', 'promo2_since_year', 'promo_interval']
# check!

df1.columns
print( 'Number of Rows: {}'.format( df1.shape[0] ) )

print( 'Number of Cols: {}'.format( df1.shape[1] ) )

# Evaluate the possibilite do use this project in your computer
df1.dtypes

# Observe date. It has a different data type.
# use "datetime" to transform the value into date.

df1['date'] = pd.to_datetime( df1['date'] )
# check if the tranformation was done.

df1.dtypes
# check if there is some missing value in the dataset.

df1.isna().sum()
# Ways to deal with NA



# competition_distance

df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 200000.0 if math.isnan( x ) else x )



# competition_open_since_month

df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)



# competition_open_since_year

df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)



# promo2_since_week

df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)



# promo2_since_year

df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)



# promo_interval

month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}



df1['promo_interval'].fillna(0, inplace=True)



df1['month_map'] = df1['date'].dt.month.map( month_map )



df1['is_promo'] = df1[['promo_interval', 'month_map']].apply( lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split( ',' ) else 0, axis=1 )

df1.isna().sum()
# Use "dtypes" to show what type of data has each column.

df1.dtypes
# competition

df1['competition_open_since_month'] = df1['competition_open_since_month'].astype( 'int64' )

df1['competition_open_since_year'] = df1['competition_open_since_year'].astype( 'int64' )



# promo2

df1['promo2_since_week'] = df1['promo2_since_week'].astype( 'int64' )

df1['promo2_since_year'] = df1['promo2_since_year'].astype( 'int64' )
df1.dtypes
num_attributes = df1.select_dtypes( include=['int64', 'float64'] )

cat_attributes = df1.select_dtypes( exclude=['int64', 'float64', 'datetime64[ns]'] )
num_attributes.sample()
cat_attributes.sample()
# Central Tendency - mean, median

ct1 = pd.DataFrame( num_attributes.apply( np.mean ) ).T

ct2 = pd.DataFrame( num_attributes.apply( np.median ) ).T



# Dispersion - std, min, max, range, skew, kurtosis

d1 = pd.DataFrame( num_attributes.apply( np.std ) ).T

d2 = pd.DataFrame( num_attributes.apply( min ) ).T

d3 = pd.DataFrame( num_attributes.apply( max ) ).T

d4 = pd.DataFrame( num_attributes.apply( lambda x: x.max() - x.min() ) ).T

d5 = pd.DataFrame( num_attributes.apply( lambda x: x.skew() ) ).T

d6 = pd.DataFrame( num_attributes.apply( lambda x: x.kurtosis() ) ).T



# concatenate

m = pd.concat( [d2, d3, d4, ct1, ct2, d1, d5, d6] ).T.reset_index()

m.columns = ( ['attributes', 'min', 'max', 'range', 'mean', 'median', 'std', 'skew', 'kurtosis'])
m
sns.distplot( df1['sales'] )
cat_attributes.apply( lambda x: x.unique().shape[0] )
aux1 = df1[(df1['state_holiday'] != '0' ) & (df1['sales'] > 0)]





plt.subplot(1, 3, 1)

sns.boxplot( x= 'state_holiday', y='sales' , data=aux1 )



plt.subplot(1, 3, 2)

sns.boxplot( x= 'store_type', y='sales' , data=aux1 )



plt.subplot(1, 3, 3)

sns.boxplot( x= 'assortment', y='sales' , data=aux1 )
df2 = df1.copy()
Image( "../input/hyphoteses-map/Hyphoteses_Map.png")
# year

df2['year'] = df2['date'].dt.year



# month

df2['month'] = df2['date'].dt.month



# day

df2['day'] = df2['date'].dt.day



# week of year

df2['week_of_year'] = df2['date'].dt.weekofyear



# year week

df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )





# competition since

df2['competition_since'] = df2.apply( lambda x: datetime.datetime( year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1 ), axis=1 )

df2['competition_time_month'] = ( ( df2['date'] - df2['competition_since'] )/30 ).apply( lambda x: x.days ).astype( int )



# promo since

df2['promo_since'] = df2['promo2_since_year'].astype( str ) + '-' + df2['promo2_since_week'].astype( str )

df2['promo_since'] = df2['promo_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )

df2['promo_time_week'] = ( ( df2['date'] - df2['promo_since'] )/7 ).apply( lambda x: x.days ).astype( int )



# assortment

df2['assortment'] = df2['assortment'].apply( lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended' )



# state holiday

df2['state_holiday'] = df2['state_holiday'].apply( lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day' )
df2.head().T
df3 = df2.copy()
df3.head()
df3 = df3[(df3['open']) != 0 & (df3['sales'] > 0)]
cols_drop = ['customers', 'open', 'promo_interval', 'month_map']

df3 = df3.drop( cols_drop, axis = 1)
df3.columns
df4 = df3.copy()
sns.distplot( df4['sales'], kde=False )
num_attributes.hist(bins=25);
cat_attributes.head()
df4['state_holiday'].drop_duplicates()
df4['store_type'].drop_duplicates()
df4['assortment'].drop_duplicates()
# state holiday



plt.subplot( 3, 2, 1)

a = df4[df4['state_holiday'] != 'regular_day']

sns.countplot( a['state_holiday'] )



plt.subplot( 3, 2, 2)

sns.kdeplot( df4[df4['state_holiday'] == 'public_holiday']['sales'], label='public_holiday', shade=True )

sns.kdeplot( df4[df4['state_holiday'] == 'easter_holiday']['sales'], label='easter_holiday', shade=True )

sns.kdeplot( df4[df4['state_holiday'] == 'christmas_holiday']['sales'], label='christmas_holiday', shade=True )



# store_type



plt.subplot( 3, 2, 3)

sns.countplot( df4['store_type'] )



plt.subplot( 3, 2, 4)

sns.kdeplot( df4[df4['store_type'] == 'a']['sales'], label='a', shade=True )

sns.kdeplot( df4[df4['store_type'] == 'b']['sales'], label='b', shade=True )

sns.kdeplot( df4[df4['store_type'] == 'c']['sales'], label='c', shade=True )

sns.kdeplot( df4[df4['store_type'] == 'd']['sales'], label='d', shade=True )



# assortment



plt.subplot( 3, 2, 5)

sns.countplot( df4['assortment'] )



plt.subplot( 3, 2, 6)

sns.kdeplot( df4[df4['assortment'] == 'extended']['sales'], label='extended', shade=True )

sns.kdeplot( df4[df4['assortment'] == 'basic']['sales'], label='basic', shade=True )

sns.kdeplot( df4[df4['assortment'] == 'extra']['sales'], label='extra', shade=True )
aux1 = df4[['assortment', 'sales']].groupby( 'assortment' ).sum().reset_index()

sns.barplot( x='assortment', y='sales', data=aux1);



aux2 = df4[['year_week', 'assortment', 'sales']].groupby( ['year_week', 'assortment'] ).sum().reset_index()

aux2.pivot( index='year_week', columns='assortment', values='sales' ).plot()



aux3 = aux2[aux2['assortment'] == 'extra']

aux3.pivot( index='year_week', columns='assortment', values='sales' ).plot()
aux1 = df4[['competition_distance', 'sales']].groupby( 'competition_distance' ).sum().reset_index()

plt.subplot(3,1,1)

sns.barplot( x='competition_distance', y='sales', data=aux1)
plt.subplot(1,3,1)

bins = list( np.arange(0, 20000, 1000) )

aux1['competition_distance_binned'] = pd.cut( aux1['competition_distance'], bins=bins )

aux2 = aux1[['competition_distance_binned', 'sales']].groupby( 'competition_distance_binned' ).sum().reset_index()

sns.barplot( x='competition_distance_binned', y='sales', data=aux2)

plt.xticks( rotation=90)



plt.subplot(1,3,2)

sns.scatterplot( x='competition_distance', y='sales', data=aux1 )



plt.subplot(1,3,3)

sns.heatmap( aux1.corr( method='pearson' ), annot=True );

aux1 = df4[['competition_open_since_month', 'sales']].groupby( 'competition_open_since_month' ).sum().reset_index()

sns.barplot( x='competition_open_since_month', y='sales', data=aux1);
aux1 = df4[['competition_time_month', 'sales']].groupby( 'competition_time_month' ).sum().reset_index()

sns.barplot( x='competition_time_month', y='sales', data=aux1);
plt.subplot( 1, 3, 1)

aux1 = df4[['competition_time_month', 'sales']].groupby( 'competition_time_month' ).sum().reset_index()

aux2 = aux1[( aux1['competition_time_month'] < 120 ) & ( aux1['competition_time_month'] != 0)]

sns.barplot( x='competition_time_month', y='sales', data=aux2)

plt.xticks( rotation=90 );



plt.subplot( 1, 3, 2)

sns.regplot( x='competition_time_month', y='sales', data=aux2)



plt.subplot( 1, 3, 3)

sns.heatmap( aux1.corr( method='pearson'), annot=True );
aux1 = df4[['promo_time_week', 'sales']].groupby('promo_time_week').sum().reset_index()

sns.barplot( x='promo_time_week', y='sales', data=aux1 );
aux1 = df4[['promo_time_week', 'sales']].groupby('promo_time_week').sum().reset_index()



grid = gridspec.GridSpec( 2, 3 )



plt.subplot(grid[0,0])

aux2 = aux1[aux1['promo_time_week'] > 0] # promo extendido

sns.barplot( x='promo_time_week', y='sales', data=aux2 );

plt.xticks( rotation=90 );



plt.subplot(grid[0,1])

sns.regplot( x='promo_time_week', y='sales', data=aux2 );



plt.subplot(grid[1,0])

aux3 = aux1[aux1['promo_time_week'] < 0] # promo regular

sns.barplot( x='promo_time_week', y='sales', data=aux3 );

plt.xticks( rotation=90 );



plt.subplot(grid[1,1])

sns.regplot( x='promo_time_week', y='sales', data=aux3 );



plt.subplot(grid[:,2])

sns.heatmap( aux1.corr( method='pearson' ), annot=True );
# Pq riscou?
df4[['promo', 'promo2', 'sales']].groupby( ['promo', 'promo2'] ).sum().reset_index()
aux1 = df4[( df4['promo'] == 1 ) & ( df4['promo2'] == 1 )][['year_week', 'sales']].groupby( 'year_week' ).sum().reset_index()

ax = aux1.plot()



aux2 = df4[( df4['promo'] == 1 ) & ( df4['promo2'] == 0 )][['year_week', 'sales']].groupby( 'year_week' ).sum().reset_index()

aux2.plot(ax=ax)



ax.legend( labels=['Tradicional & Extendida', 'Extendida'] );
aux = df4[df4['state_holiday'] != 'regular_day']



plt.subplot( 1, 2, 1)

aux1 = aux[['state_holiday', 'sales']].groupby( 'state_holiday' ).sum().reset_index()

sns.barplot( x='state_holiday', y='sales', data=aux1);



plt.subplot( 1, 2, 2)

aux2 = aux[['year', 'state_holiday', 'sales']].groupby( ['year', 'state_holiday'] ).sum().reset_index()

sns.barplot( x='year', y='sales', hue='state_holiday', data=aux2 );
aux1 = df4[['year', 'sales']].groupby( 'year' ).sum().reset_index()



plt.subplot( 1, 3, 1 )

sns.barplot( x='year', y='sales', data=aux1 );



plt.subplot( 1, 3, 2 )

sns.regplot( x='year', y='sales', data=aux1 );



plt.subplot( 1, 3, 3 )

sns.heatmap( aux1.corr( method='pearson' ), annot=True );
aux1 = df4[['month', 'sales']].groupby( 'month' ).sum().reset_index()



plt.subplot( 1, 3, 1 )

sns.barplot( x='month', y='sales', data=aux1 );



plt.subplot( 1, 3, 2 )

sns.regplot( x='month', y='sales', data=aux1 );



plt.subplot( 1, 3, 3 )

sns.heatmap( aux1.corr( method='pearson' ), annot=True );
aux1 = df4[['day', 'sales']].groupby( 'day' ).sum().reset_index()



plt.subplot( 2, 2, 1 )

sns.barplot( x='day', y='sales', data=aux1 );



plt.subplot( 2, 2, 2 )

sns.regplot( x='day', y='sales', data=aux1 );



plt.subplot( 2, 2, 3 )

sns.heatmap( aux1.corr( method='pearson' ), annot=True );



aux1['before_after'] = aux1['day'].apply( lambda x: 'before_after' if x <= 10 else 'after_10_days' )

aux2 = aux1[['before_after', 'sales']].groupby( 'before_after' ).sum().reset_index()



plt.subplot( 2, 2, 4 )

sns.barplot( x='before_after', y='sales', data=aux2 );
aux1 = df4[['day_of_week', 'sales']].groupby( 'day_of_week' ).sum().reset_index()



plt.subplot( 1, 3, 1 )

sns.barplot( x='day_of_week', y='sales', data=aux1 );



plt.subplot( 1, 3, 2 )

sns.regplot( x='day_of_week', y='sales', data=aux1 );



plt.subplot( 1, 3, 3 )

sns.heatmap( aux1.corr( method='pearson' ), annot=True );
aux1 = df4[['school_holiday', 'sales']].groupby( 'school_holiday' ).sum().reset_index()

plt.subplot( 2, 1, 1 )

sns.barplot( x='school_holiday', y='sales', data=aux1 );



aux2 = df4[['month', 'school_holiday', 'sales']].groupby( ['month', 'school_holiday'] ).sum().reset_index()

plt.subplot( 2, 1, 2 )

sns.barplot( x='month', y='sales', hue='school_holiday', data=aux2 );
tab = [['Hipoteses', 'Conclusao', 'Relevancia'],

       ['H1', 'Falsa', 'Baixa'],

       ['H2', 'Falsa', 'Media'],

       ['H3', 'Falsa', 'Media'],

       ['H4', 'Falsa', 'Baixa'],

       ['H5', '-', '-'],

       ['H6', 'Falsa', 'Baixa'],

       ['H7', 'Falsa', 'Media'],

       ['H8', 'Falsa', 'Alta'],

       ['H9', 'Falsa', 'Alta'],

       ['H10', 'Verdadeira', 'Alta'],

       ['H11', 'Verdadeira', 'Alta'],

       ['H12', 'Verdadeira', 'Baixa']

      ]

print( tabulate( tab, headers='firstrow' ) )
correlation = num_attributes.corr( method='pearson' )

sns.heatmap( correlation, annot=True )
# Only categorical data

a1 = cramer_v( a['state_holiday'], a['state_holiday'] )



# Calculate cramer V

a2 = cramer_v( a['state_holiday'], a['store_type'] )

a3 = cramer_v( a['state_holiday'], a['assortment'] )



a4 = cramer_v( a['store_type'], a['state_holiday'] )

a5 = cramer_v( a['store_type'], a['store_type'] )

a6 = cramer_v( a['store_type'], a['assortment'] )



a7 = cramer_v( a['assortment'], a['state_holiday'] )

a8 = cramer_v( a['assortment'], a['store_type'] )

a9 = cramer_v( a['assortment'], a['assortment'] )



# Final dataset

d = pd.DataFrame( {'state_holiday': [a1, a2, a3],

               'store_type': [a4, a5, a6],

               'assortment': [a7, a8, a9]} )



d = d.set_index( d.columns )



sns.heatmap( d, annot=True )
df5 = df4.copy()
# Go to 4.1.2. Numerical variable to see if there is any Gaussian distribution



# We don't have any data distribution like a Gaussian.
a = df5.select_dtypes( include=['int64', 'float64'] )
sns.boxplot( df5['competition_distance'] )
sns.boxplot( df5['competition_time_month'] )
sns.boxplot( df5['promo_time_week'] )
rs = RobustScaler()



# competition distance

df5['competition_distance'] = rs.fit_transform( df5[['competition_distance']].values )



# competition time month

df5['competition_time_month'] = rs.fit_transform( df5[['competition_time_month']].values )



mms = MinMaxScaler()



# promo time week

df5['promo_time_week'] = mms.fit_transform( df5[['promo_time_week']].values )



# year

df5['year'] = mms.fit_transform( df5[['year']].values )
sns.distplot( df5['competition_distance'] )
# state_holiday - One Hot Encoding

df5 = pd.get_dummies( df5, prefix=['state_holiday'], columns=['state_holiday'] )



le = LabelEncoder()



# store_type - Label Encoding

df5['store_type'] = le.fit_transform( df5['store_type'] )



# assortment - Ordinal Encoding

assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}

df5['assortment'] = df5['assortment'].map( assortment_dict )
df5['sales'] = np.log1p( df5['sales'] )
sns.distplot( df5['sales'] )
# day of week

df5['day_of_week_sin'] = df5['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )

df5['day_of_week_cos'] = df5['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )



# month

df5['month_sin'] = df5['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )

df5['month_cos'] = df5['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )



# day

df5['day_sin'] = df5['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )

df5['day_cos'] = df5['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )



# week of year

df5['week_of_year_sin'] = df5['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/52 ) ) )

df5['week_of_year_cos'] = df5['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/52 ) ) )
df5.head()
df6 = df5.copy()
df6.shape
cols_drop = ['week_of_year', 'day', 'month', 'day_of_week', 'promo_since', 'competition_since', 'year_week']

df6 = df6.drop( cols_drop, axis=1 )
df6[['store', 'date']].groupby( 'store' ).max().reset_index()['date'][0] - datetime.timedelta( days=6*7 )
# training dataset

X_train = df6[df6['date'] < '2015-06-19']

y_train = X_train['sales']



# test dataset

X_test = df6[df6['date'] >= '2015-06-19']

y_test = X_test['sales']



print( 'Training Min Date: {}'.format( X_train['date']. min() ) )

print( 'Training Max Date: {}'.format( X_train['date']. max() ) )



print( '\nTest Min Date: {}'.format( X_test['date']. min() ) )

print( 'Test Max Date: {}'.format( X_test['date']. max() ) )
# training and test dataset for Boruta

X_train_n = X_train.drop( ['date', 'sales'], axis=1 ).values

y_train_n = y_train.values.ravel()



# define RandomForestRegression

rf = RandomForestRegressor( n_jobs=-1 )



# define Boruta

###boruta = BorutaPy( rf, n_estimators='auto', verbose=2, random_state=42 ).fit( X_train_n, y_train_n)
####cols_selected = boruta.support_.tolist()



# best feature

#######X_train_fs = X_train.drop( ['date', 'sales'], axis=1 )

###########cols_selected_boruta = X_train_fs.iloc[:, cols_selected].columns.tolist()



# not selected boruta

#########cols_not_selected_boruta = list( np.setdiff1d( X_train_fs.columns, cols_selected_boruta ) )
############cols_selected_boruta
############cols_not_selected_boruta
tab = [['Hipoteses', 'Conclusao', 'Relevancia'],

       ['H1', 'Falsa', 'Baixa'],

       ['H2', 'Falsa', 'Media'],

       ['H3', 'Falsa', 'Media'],

       ['H4', 'Falsa', 'Baixa'],

       ['H5', '-', '-'],

       ['H6', 'Falsa', 'Baixa'],

       ['H7', 'Falsa', 'Media'],

       ['H8', 'Falsa', 'Alta'],

       ['H9', 'Falsa', 'Alta'],

       ['H10', 'Verdadeira', 'Alta'],

       ['H11', 'Verdadeira', 'Alta'],

       ['H12', 'Verdadeira', 'Baixa']

      ]

print( tabulate( tab, headers='firstrow' ) )
cols_selected_boruta = [

     'store',

     'promo',

     'store_type',

     'assortment',

     'competition_distance',

     'competition_open_since_month',

     'competition_open_since_year',

     'promo2',

     'promo2_since_week',

     'promo2_since_year',

     'competition_time_month',

     'promo_time_week',

     'day_of_week_sin',

     'day_of_week_cos',

     'month_sin',

     'month_cos',

     'day_sin',

     'day_cos',

     'week_of_year_sin',

     'week_of_year_cos']



# columns to add

feat_to_add = ['date', 'sales']



# final features

cols_selected_boruta_full = cols_selected_boruta.copy()

cols_selected_boruta_full.extend( feat_to_add )
cols_selected_boruta
cols_selected_boruta_full
x_train = X_train[ cols_selected_boruta ]

x_test = X_test[ cols_selected_boruta ]



# Time Series Data Preparartion

X_training = X_train[ cols_selected_boruta_full ]
aux1 = x_test.copy()

aux1['sales'] = y_test.copy()



# prediction

aux2 = aux1[['store', 'sales']].groupby( 'store' ).mean().reset_index().rename( columns={'sales': 'predictions'} )

aux1 = pd.merge( aux1, aux2, how='left', on='store' )

yhat_baseline = aux1['predictions']



# performance

baseline_result = ml_error( 'Average Model', np.expm1( y_test ), np.expm1( yhat_baseline ) )

baseline_result
# model

lr = LinearRegression().fit( x_train, y_train )



# prediction

yhat_lr = lr.predict( x_test )



# performance

lr_result = ml_error( 'Linear Regression', np.expm1( y_test ), np.expm1( yhat_lr ) )

lr_result
lr_result_cv = cross_validation( X_training, 5, 'Linear Regression', lr, verbose=False )

lr_result_cv
# model

lrr = Lasso( alpha=0.00001 ).fit( x_train, y_train )



# prediction

yhat_lrr = lrr.predict( x_test )



# performance

lrr_result = ml_error( 'Linear Regression - Lasso', np.expm1( y_test ), np.expm1( yhat_lrr ) )

lrr_result
lrr_result_cv = cross_validation( X_training, 5, 'Linear Regression - Lasso', lrr, verbose=False )

lrr_result_cv
# model

rf = RandomForestRegressor( n_estimators=100, n_jobs=-1, random_state=42 ).fit( x_train, y_train )



# prediction

yhat_rf = rf.predict( x_test )



# performance

rf_result = ml_error( 'Random Forest Regressor', np.expm1( y_test ), np.expm1( yhat_rf ) )

rf_result
rf_result_cv = cross_validation( X_training, 5, 'Random Forest Regressor', rf, verbose=True )

rf_result_cv
# model

model_xgb = xgb.XGBRegressor( objective='reg:squarederror',

                             n_estimators=100, 

                             eta=0.01,

                             max_depth=10,

                             subsample=0.7,

                             colsample_bytree=0.9 ).fit( x_train, y_train )



# prediction

yhat_xgb = model_xgb.predict( x_test )



# performance

xgb_result = ml_error( 'XGBoost Regressor', np.expm1( y_test ), np.expm1( yhat_xgb ) )

xgb_result
xgb_result_cv = cross_validation( X_training, 5, 'XGBoost Regressor', model_xgb, verbose=True )

xgb_result_cv
modelling_result = pd.concat( [baseline_result, lr_result, lrr_result, rf_result, xgb_result] )

modelling_result.sort_values( 'RMSE' )
modelling_result_cv = pd.concat( [lr_result_cv, lrr_result_cv, rf_result_cv, xgb_result_cv] )

modelling_result_cv
param = { 

        'n_estimators': [1500, 1700, 2500, 3000, 3500],

        'eta': [0.01, 0.03],

        'max_depth': [3, 5, 9],

        'subsample': [0.1, 0.5, 0.7],

        'colsample_bytree': [0.3, 0.7, 0.9],

        'min_child_weight': [3, 8, 15] 

        }



MAX_EVAL = 5
final_result = pd.DataFrame()



import random



for i in range( MAX_EVAL):

    #chose values for parameters randomly

    hp = {k: random.sample(v, 1)[0] for k, v in param.items()}

    print( hp )

    

    # model

    model_xgb = xgb.XGBRegressor( objective='reg:squarederror',

                                 n_estimators=hp['n_estimators'], 

                                 eta=hp['eta'],

                                 max_depth=hp['max_depth'],

                                 subsample=hp['subsample'],

                                 colsample_bytree=hp['colsample_bytree'],

                                 min_child_weight=hp['min_child_weight'] )



    # performance

    result = cross_validation( X_training, 5, 'XGBoost Regressor', model_xgb, verbose=False)

    final_result = pd.concat( [final_result, result] )



final_result
param_tuned = { 

        'n_estimators': 1500,

        'eta': 0.01,

        'max_depth': 9,

        'subsample': 0.5,

        'colsample_bytree': 0.9,

        'min_child_weight': 8

        }
# model

model_xgb_tuned = xgb.XGBRegressor( objective='reg:squarederror',

                            n_estimators=param_tuned['n_estimators'], 

                            eta=param_tuned['eta'],

                            max_depth=param_tuned['max_depth'],

                            subsample=param_tuned['subsample'],

                            colsample_bytree=param_tuned['colsample_bytree'],

                            min_child_weight=param_tuned['min_child_weight'] ).fit( x_train, y_train )



# prediction

yhat_xgb_tuned = model_xgb_tuned.predict( x_test )



# performance

xgb_result_tuned = ml_error( 'XGBoost Regressor', np.expm1( y_test ), np.expm1( yhat_xgb_tuned ) )

xgb_result_tuned
mpe = mean_percentage_error( np.expm1( y_test ), np.expm1( yhat_xgb_tuned ) )

mpe
df9 = X_test[ cols_selected_boruta_full ]



# rescale

df9['sales'] = np.expm1( df9['sales'] )

df9['predictions'] = np.expm1( yhat_xgb_tuned )
# sum of predictions

df91 = df9[['store', 'predictions']].groupby( 'store' ).sum().reset_index()



# MAE e MAPE

df9_aux1 = df9[['store', 'sales', 'predictions']].groupby( 'store' ).apply( lambda x: mean_absolute_error( x['sales'], x['predictions'] ) ).reset_index().rename( columns={0:'MAE'} )

df9_aux2 = df9[['store', 'sales', 'predictions']].groupby( 'store' ).apply( lambda x: mean_absolute_percentage_error( x['sales'], x['predictions'] ) ).reset_index().rename( columns={0:'MAPE'} )



# Merge

df9_aux3 = pd.merge( df9_aux1, df9_aux2, how='inner', on='store' )

df92 = pd.merge( df91, df9_aux3, how='inner', on='store' )



# Scenarios

df92['worst_scenario'] = df92['predictions'] - df92['MAE']

df92['best_scenario'] = df92['predictions'] + df92['MAPE']



# order columns

df92 = df92[['store', 'predictions', 'worst_scenario', 'best_scenario', 'MAE', 'MAPE']]
df9_aux1.head()
df92.sample(4)
df92.sort_values( 'MAPE', ascending=False ).head()
sns.scatterplot( x='store', y='MAPE', data=df92 )
df93 = df92[['predictions', 'worst_scenario', 'best_scenario']].apply( lambda x: np.sum( x ), axis=0 ).reset_index().rename( columns={'index': 'Scenario', 0:'Values'} )

df93['Values'] = df93['Values'].map( 'R${:,.2f}'.format )

df93
df9['error'] = df9['sales'] - df9['predictions']

df9['error_rate'] = df9['predictions'] / df9['sales']
plt.subplot( 2, 2, 1 )

sns.lineplot( x='date', y='sales', data=df9, label='SALES' )

sns.lineplot( x='date', y='predictions', data=df9, label='PREDICTIONS' )



plt.subplot( 2, 2, 2 )

sns.lineplot( x='date', y='error_rate', data=df9 )

plt.axhline( 1, linestyle='--' )



plt.subplot( 2, 2, 3 )

sns.distplot( df9['error'] )



plt.subplot( 2, 2, 4 )

sns.scatterplot( df9['predictions'], df9['error'] )
# Save the Model

# pickle.dump( model_xgb_tuned, open('/Users/favh2/github_projects/Data_Science_inProduction/model/model_rossmann.pkl', 'wb' ) )
!pip install inflection



import pickle

import inflection

import pandas as pd

import numpy as np

import math

import datetime



class Rossmann( object ):

    def __init__( self ):

        self.home_path='/Users/favh2/github_projects/Data_Science_inProduction/'

        self.competition_distance_scaler   = pickle.load( open( self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb' ) )

        self.competition_time_month_scaler = pickle.load( open( self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb' ) )

        self.promo_time_week_scaler        = pickle.load( open( self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb' ) )

        self.year_scaler                   = pickle.load( open( self.home_path + 'parameter/year_scaler.pkl', 'rb' ) )

        self.store_type_scaler             = pickle.load( open( self.home_path + 'parameter/store_type_scaler.pkl', 'rb' ) )

        

    def data_cleaning( self, df1 ):

        

        ## 1.1 Rename olumns

        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 

                    'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 

                    'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']





        snakecase = lambda x: inflection.underscore( x )



        cols_new = list( map( snakecase, cols_old ) )



        # rename

        df1.columns = cols_new



        ## 1.3. Data Types

        df1['date'] = pd.to_datetime( df1['date'] )



        ## 1.5. Fillout NA

        # competition_distance

        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 200000.0 if math.isnan( x ) else x )



        # competition_open_since_month

        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)



        # competition_open_since_year

        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)



        # promo2_since_week

        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)



        # promo2_since_year

        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)



        # promo_interval

        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}



        df1['promo_interval'].fillna(0, inplace=True)



        df1['month_map'] = df1['date'].dt.month.map( month_map )



        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply( lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split( ',' ) else 0, axis=1 )



        ## 1.6. Change Types

        # competition

        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype( 'int64' )

        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype( 'int64' )



        # promo2

        df1['promo2_since_week'] = df1['promo2_since_week'].astype( 'int64' )

        df1['promo2_since_year'] = df1['promo2_since_year'].astype( 'int64' )

        

        return df1



    def feature_engineering( self, df2 ):

        

        # year

        df2['year'] = df2['date'].dt.year



        # month

        df2['month'] = df2['date'].dt.month



        # day

        df2['day'] = df2['date'].dt.day



        # week of year

        df2['week_of_year'] = df2['date'].dt.weekofyear



        # year week

        df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )





        # competition since

        df2['competition_since'] = df2.apply( lambda x: datetime.datetime( year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1 ), axis=1 )

        df2['competition_time_month'] = ( ( df2['date'] - df2['competition_since'] )/30 ).apply( lambda x: x.days ).astype( int )



        # promo since

        df2['promo_since'] = df2['promo2_since_year'].astype( str ) + '-' + df2['promo2_since_week'].astype( str )

        df2['promo_since'] = df2['promo_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )

        df2['promo_time_week'] = ( ( df2['date'] - df2['promo_since'] )/7 ).apply( lambda x: x.days ).astype( int )



        # assortment

        df2['assortment'] = df2['assortment'].apply( lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended' )



        # state holiday

        df2['state_holiday'] = df2['state_holiday'].apply( lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day' )



        # 3.0. STEP 03 - VARIABLE FILTERING

        ## 3.1. Line Filtering

        df2 = df2[(df2['open']) != 0]



        ## 3.2. Column selection

        cols_drop = ['open', 'promo_interval', 'month_map']

        df2 = df2.drop( cols_drop, axis = 1)

        

        return df2

    

    def data_preparation( self, df5 ):

        

        ## 5.2. Rescaling

        # competition distance

        df5['competition_distance'] = self.competition_distance_scaler.fit_transform( df5[['competition_distance']].values )



        # competition time month

        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform( df5[['competition_time_month']].values )

        

        # promo time week

        df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform( df5[['promo_time_week']].values )

        

        # year

        df5['year'] = self.year_scaler.fit_transform( df5[['year']].values )



        ### 5.3.1 Encoding

        # state_holiday - One Hot Encoding

        df5 = pd.get_dummies( df5, prefix=['state_holiday'], columns=['state_holiday'] )

    

        # store_type - Label Encoding

        df5['store_type'] = self.store_type_scaler.fit_transform( df5['store_type'] )

        

        # assortment - Ordinal Encoding

        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}

        df5['assortment'] = df5['assortment'].map( assortment_dict )



        ### 5.3.3. Nature Transformation

        # day of week

        df5['day_of_week_sin'] = df5['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )

        df5['day_of_week_cos'] = df5['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )



        # month

        df5['month_sin'] = df5['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )

        df5['month_cos'] = df5['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )



        # day

        df5['day_sin'] = df5['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )

        df5['day_cos'] = df5['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )



        # week of year

        df5['week_of_year_sin'] = df5['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/52 ) ) )

        df5['week_of_year_cos'] = df5['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/52 ) ) )

        

        cols_selected = [ 'store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month', 'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year',

                                 'competition_time_month', 'promo_time_week', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_of_year_sin', 'week_of_year_cos' ]

        

        return df5[ cols_selected ]

    

    def get_prediction( self, model, original_data, test_data ):

        

        # prediction

        pred = model.predict( test_data )

        

        # join pred into the original data

        original_data['prediction'] = np.expm1( pred )

        

        return original_data.to_json( orient='records', date_format='iso' )
import pickle

import pandas as pd

from flask import Flask, request, Response

from rossmann.Rossmann import Rossmann



# loading model

# model = pickle.load( open( '/Users/favh2/github_projects/Data_Science_inProduction/model/model_rossmann.pkl', 'rb' ) )



# initialize API

app = Flask( __name__ )



@app.route( '/rossmann/predict', methods=['POST'] )

def rossmann_predict():

    test_json = request.get_json()

    

    if test_json: # there is data

        if isinstance( test_json, dict ): # unique example

            test_raw = pd.DataFrame( test_json, index=[0] )

        else: # multiple examples

            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )

            

        # Instantiate Rossmann Class

        pipeline = Rossmann()

        

        # data cleaning

        df1 = pipeline.data_cleaning( test_raw )

        

        # feature engineering

        df2 = pipeline.feature_engineering( df1 )

        

        # data preparation

        df3 = pipeline.data_preparation( df2 )

        

        # prediction

        df_response = pipeline.get_prediction( model, test_raw, df3 )

        

        return df_response

        

    else:

        return Response( '{}', status=200, mimetype='application/json' )



if __name__ == '__main__':

    app.run( '0.0.0.0' )
# loading test dataset

df10 = pd.read_csv( '../input/rossmann-store-sales/test.csv' )
# merge test dataset + store

df_test = pd.merge( df10, df_store_raw, how='left', on='Store' )



# choose store for prediction

df_test = df_test[df_test['Store'].isin([20, 10, 27])]



# remove closed days

df_test = df_test[df_test['Open'] != 0]

df_test = df_test[~df_test['Open'].isnull()]

df_test = df_test.drop( 'Id', axis=1 )
# convert DataFrame to json

data = json.dumps( df_test.to_dict( orient='records' ) )
# API call

#url = 'http://0.0.0.0:5000/rossmann/predict'

#url = 'https://rossmann-model-victorpereira.herokuapp.com/rossmann/predict'

#header = { 'Content-type': 'application/json' }

#data = data



#r = requests.post( url, data=data, headers=header )

#print( 'Status Code {}'.format( r.status_code ) )
#d1 = pd.DataFrame( r.json(), columns=r.json()[0].keys() )
#d1.sample(10)
#d2 = d1[['store', 'prediction']].groupby( 'store' ).sum().reset_index()



#for i in range( len( d2 )):

#    print( 'Store Number {} will sell R${:,.2f} in the next 6 weeks'.format(

#            d2.loc[i, 'store'],

#            d2.loc[i, 'prediction'] ) )
#d2