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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,15)

import seaborn as sns

import scipy.stats as st

import math
# Table 1: year variables

# Table 2: Month variables

london_table = pd.read_csv(r'/kaggle/input/housing-in-london/housing_in_london_monthly_variables.csv', sep=',')

london_table2 = pd.read_csv(r'/kaggle/input/housing-in-london/housing_in_london_yearly_variables.csv', sep=',')
# split dates

london_table['date'] = pd.to_datetime(london_table['date'], errors = 'coerce')

london_table = london_table.assign(day=london_table.date.dt.day.astype('uint16'),

                           year=london_table.date.dt.year.astype('uint16'),

                           month=london_table.date.dt.month.astype('uint16'),

                           )



london_table2['date'] = pd.to_datetime(london_table2['date'], errors = 'coerce')

london_table2 = london_table2.assign(day=london_table2.date.dt.day.astype('uint16'),

                           year=london_table2.date.dt.year.astype('uint16'),

                           month=london_table2.date.dt.month.astype('uint16'),

                           )
london_table
london_table2
# creation of joined key between the two previous tables

london_table["code_year"] = london_table["code"] + "_" + london_table["year"].astype('str')

london_table2["code_year"] = london_table2["code"] + "_" + london_table2["year"].astype('str')

london_table2["code_year_month"] = london_table2["code"] + "_" + london_table2["year"].astype('str') + "_" + london_table2["month"].astype('str')
# merge

table = london_table2.merge(london_table.drop(['day','year','month','borough_flag','area','code', 'date'], axis=1), how="inner", on="code_year")
# Take into account borough and not regions

table_borough = table[table['borough_flag']==1]

table_borough.replace([0,'#','na'],np.nan, inplace=True) # Bad data

table_borough.mean_salary = table_borough.mean_salary.astype('float64')

table_borough.recycling_pct = table_borough.recycling_pct.astype('float64')

print(table_borough.info()) # Verifying if no bad data like '-'



# Nan analysis

# Nan according each code -> E0900001 don't have data or few for No_of_crimes or life_satisfaction

# We can replaced them with a copy of an other borough closer than E0900001.

table_borough.groupby(by='code').agg({'no_of_crimes':lambda x : x.isna().sum(),

                                     'median_salary':lambda x : x.isna().sum(),

                                     'life_satisfaction':lambda x : x.isna().sum(),

                                     'recycling_pct':lambda x : x.isna().sum(),

                                     'population_size':lambda x : x.isna().sum(),

                                     'number_of_jobs':lambda x : x.isna().sum(),

                                     'area_size':lambda x : x.isna().sum(),

                                     'no_of_houses':lambda x : x.isna().sum()}).plot(kind='bar', width=0.9)



# Nan according each years -> 1999 and 2019 have no data, that's mean lot of borough have no data for these columns.

# We can replace them by regression over years.

table_borough.groupby(by='year').agg({'no_of_crimes':lambda x : x.isna().sum(),

                                     'median_salary':lambda x : x.isna().sum(),

                                     'life_satisfaction':lambda x : x.isna().sum(),

                                     'recycling_pct':lambda x : x.isna().sum(),

                                     'population_size':lambda x : x.isna().sum(),

                                     'number_of_jobs':lambda x : x.isna().sum(),

                                     'area_size':lambda x : x.isna().sum(),

                                     'no_of_houses':lambda x : x.isna().sum()}).plot(kind='bar', width=0.9)
# Take into account regions

table_region = table[table['borough_flag']==0]



# Replace bad data

table_region.replace([0,'-'],np.nan, inplace=True)

table_region.mean_salary = table_region.mean_salary.astype('float64')

table_region.recycling_pct = table_region.recycling_pct.astype('float64')



# link with table_borough

table_region['area_year_month'] = table_region.area + '_' + table_region.year.astype('str') + '_' + table_region.month.astype('str')

print(table_region.info()) # Verifying if there is no bad data



table_region.groupby(by='code').agg({'median_salary':lambda x : x.isna().sum(),

                                     'life_satisfaction':lambda x : x.isna().sum(),

                                     'recycling_pct':lambda x : x.isna().sum(),

                                     'population_size':lambda x : x.isna().sum(),

                                     'number_of_jobs':lambda x : x.isna().sum(),

                                     'area_size':lambda x : x.isna().sum(),

                                     'no_of_houses':lambda x : x.isna().sum()}).plot(kind='bar', width=0.9)

table_region.groupby(by='year').agg({'median_salary':lambda x : x.isna().sum(),

                                     'life_satisfaction':lambda x : x.isna().sum(),

                                     'recycling_pct':lambda x : x.isna().sum(),

                                     'population_size':lambda x : x.isna().sum(),

                                     'number_of_jobs':lambda x : x.isna().sum(),

                                     'area_size':lambda x : x.isna().sum(),

                                     'no_of_houses':lambda x : x.isna().sum()}).plot(kind='bar', width=0.9)



# No no_of_crimesand drop dupllicates data from table_borough and table_region

try:

    table_region.drop(['no_of_crimes','borough_flag','day','code_year','code','year','month','area'], axis=1, inplace=True)

except:

    pass

table_borough
table_region
# We divide each borough into two categories of regions 

if 'london' not in table_borough.columns:

    table_borough = table_borough.merge(pd.Series([clef for clef in {

    'city of london':'inner london',

    'barking and dagenham':'outer london',

    'barnet':'outer london',

    'bexley':'outer london',

    'brent':'outer london',

    'bromley':'outer london',

    'croydon':'outer london',

    'ealing':'outer london',

    'enfield':'outer london',

    'haringey':'outer london',

    'harrow':'outer london',

    'havering':'outer london',

    'hillingdon':'outer london',

    'hounslow':'outer london',

    'kingston upon thames':'outer london',

    'merton':'outer london',

    'newham':'outer london',

    'redbridge':'outer london',

    'richmond upon thames':'outer london',

    'sutton':'outer london',

    'waltham forest':'outer london',

    'camden':'inner london',

    'greenwich':'inner london',

    'hackney':'inner london',

    'hammersmith and fulham':'inner london',

    'islington':'inner london',

    'kensington and chelsea':'inner london',

    'lambeth':'inner london',

    'lewisham':'inner london',

    'southwark':'inner london',

    'tower hamlets':'inner london',

    'wandsworth':'inner london',

    'westminster':'inner london'}.values()], index = [value for value in {

    'city of london':'inner london',

    'barking and dagenham':'outer london',

    'barnet':'outer london',

    'bexley':'outer london',

    'brent':'outer london',

    'bromley':'outer london',

    'croydon':'outer london',

    'ealing':'outer london',

    'enfield':'outer london',

    'haringey':'outer london',

    'harrow':'outer london',

    'havering':'outer london',

    'hillingdon':'outer london',

    'hounslow':'outer london',

    'kingston upon thames':'outer london',

    'merton':'outer london',

    'newham':'outer london',

    'redbridge':'outer london',

    'richmond upon thames':'outer london',

    'sutton':'outer london',

    'waltham forest':'outer london',

    'camden':'inner london',

    'greenwich':'inner london',

    'hackney':'inner london',

    'hammersmith and fulham':'inner london',

    'islington':'inner london',

    'kensington and chelsea':'inner london',

    'lambeth':'inner london',

    'lewisham':'inner london',

    'southwark':'inner london',

    'tower hamlets':'inner london',

    'wandsworth':'inner london',

    'westminster':'inner london'

    }.keys()], name='london'), how='left', left_on='area', right_index=True)

    

    table_borough['london'] = table_borough.london + '_' + table_borough.year.astype('str') + '_' + table_borough.month.astype('str')



table_b_r = table_borough.merge(table_region.drop(['date','code_year_month'], axis=1), how='left', left_on='london', right_on='area_year_month')

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PowerTransformer

from sklearn.svm import OneClassSVM

from sklearn.covariance import EllipticEnvelope

from sklearn.ensemble import IsolationForest



# Encoding

try:

    cat_features = ['area', 'code', 'code_year', 'code_year_month', 'london','area_year_month']

    num_features = table_b_r.drop(cat_features, axis=1).drop(['average_price_x'], axis=1).columns

except NameError:

    print('n')



label_encoder = LabelEncoder()

for col in cat_features:

    table_b_r[col] = label_encoder.fit_transform(table_b_r[col])



table_b_r
# inputs and outputs detection

# flag nan columns 

def detection_nan(dataframe, append_col=True):

    col_nan = []

    for col in dataframe:

        if dataframe[col].isna().any():

            if append_col:

                dataframe[col+'_'+'nan'] = [1 if math.isnan(i) else 0 for i in dataframe[col]]

            col_nan.append(col)

    return dataframe, col_nan



def get_data_splits(dataframe, valid_fraction=0.1):

    dataframe = dataframe.sort_values(by='date')

    valid_size = int(len(dataframe) * valid_fraction)



    train = dataframe[:-valid_size * 2]

    # valid size == test size, last two sections of the data

    valid = dataframe[-valid_size * 2:-valid_size]

    test = dataframe[-valid_size:]

    

    return train, valid, test



table_b_r = detection_nan(table_b_r)[0]

train, valid, test = get_data_splits(table_b_r)

X_test = test.drop('average_price_x', axis=1)

y_test = test['average_price_x']

X_valid = valid.drop('average_price_x', axis=1)

y_valid = valid['average_price_x']

X_train_filt = train.drop('average_price_x', axis=1)

y_train_filt = train['average_price_x']
# Preprocessing numeric data

# outliers are deleted

# useless to delete outliers

"""def outliers(clf, y_train=y_train.copy(), X_train=X_train.copy(), features_x=['area_size_x','population_size_x'], y=True):

    

    if y:

        y_train = y_train.to_frame()

        X = y_train.values.reshape(-1,1)

        clf = clf.fit(X)

        y_train['res'] = clf.predict(X)

    X_train = X_train.loc[(y_train['res']==1),:]

    y_train = y_train.query('res==1').iloc[:,0]

    

    # Inutile de supprimer les outliers des autres colonnes

    for col in features_x:

        X = X_train[col].replace(np.nan, np.mean(X)).values.reshape(-1,1)

        clf = clf.fit(X)

        X_train[col+'_res'] = clf.predict(X)

    X_train.replace({col:-1 for col in features_x}, np.nan, inplace=True)

    X_train.dropna(subset=features_x, inplace=True)

    X_train = X_train.loc[:, [col for col in X_train.columns if 'res' not in col]]

    y_train = y_train.loc[X_train.index]

    

    return X_train, y_train



out = outliers(OneClassSVM(gamma='auto'))

X_train_filt = out[0]

y_train_filt = out[1]"""
# Fill nan by regression

def fill_nan(dataframe=X_train_filt.copy(), col_nan=detection_nan(X_train_filt)[1]):

    for code in dataframe.code.unique():

        data = dataframe[dataframe['code']==code]

        for col in col_nan:

            try:

                if data[col].isna().any():

                    data_regr = data.dropna(subset=[col])

                    regr = st.linregress(data_regr['code_year_month'], data_regr[col])

                    data.loc[data[col].isna().loc[data[col].isna()==True].index, col] = regr[0]*data.loc[data[col].isna().loc[data[col].isna()==True].index, 'code_year_month']+ regr[1]

                    dataframe.loc[dataframe['code']==code, col] = data[col].values

            except ValueError:

                print(code, col)

    return dataframe

                

X_train_filt = fill_nan()
# Copying data for columns no_of_crimes and life_satisfaction from the closest borough for code E0900001 (code==0)



for idx, price in y_train_filt.loc[X_train_filt[X_train_filt.code==0].index].iteritems():

    idx_closest = (y_train_filt.loc[X_train_filt[X_train_filt.code!=0].index]-price).abs().sort_values().index[0]

    X_train_filt.loc[idx,'no_of_crimes'] = X_train_filt.loc[idx_closest,'no_of_crimes']

    X_train_filt.loc[idx,'life_satisfaction_x'] = X_train_filt.loc[idx_closest,'life_satisfaction_x']

X_train_filt.info()
# We could also replace columns 23, 25 and 27 by regressions but i prefer to drop them.

X_train_filt = X_train_filt.drop(['date','borough_flag','life_satisfaction_y','recycling_pct_y','number_of_jobs_y','life_satisfaction_y_nan','recycling_pct_y_nan','number_of_jobs_y_nan'], axis=1)

X_test = X_test.drop(['date','borough_flag','life_satisfaction_y','recycling_pct_y','number_of_jobs_y','life_satisfaction_y_nan','recycling_pct_y_nan','number_of_jobs_y_nan'], axis=1)

num_features = num_features.drop(['life_satisfaction_y','recycling_pct_y','number_of_jobs_y','date','borough_flag','day','year','month'])

X_valid = X_valid.drop(['date','borough_flag','life_satisfaction_y','recycling_pct_y','number_of_jobs_y','life_satisfaction_y_nan','recycling_pct_y_nan','number_of_jobs_y_nan'], axis=1)

X_valid.dropna(how='any', inplace=True)



y_valid = y_valid.loc[X_valid.index]

X_test_filt = X_test.dropna(how='any')

y_test_filt = y_test.loc[X_test_filt.index]
# Correlation analysis

# We calculate mean correlation.

corr_table = X_train_filt[X_train_filt['code']==X_train_filt['code'].unique()[0]].corr().fillna(0)

for code in X_train_filt['code'].unique()[1:]:

    corr_table = corr_table + X_train_filt[X_train_filt['code']==code].corr().fillna(0)

corr_table = corr_table / len(X_train_filt['code'].unique())

corr_table
# prepocessing # useless

"""from sklearn.preprocessing import Normalizer



preprocessor = ColumnTransformer(

    transformers=[

        ('num', PowerTransformer(), num_features)

    ])



X_train_filt.loc[:,num_features] = preprocessor.fit_transform(X_train_filt)

y_train_filt_tf = np.log(y_train_filt)"""
#Data augmentation

X_train_filt['test'] = X_train_filt['average_price_y']*X_train_filt.area_year_month

X_valid['test'] = X_valid['average_price_y']*X_valid.area_year_month

X_test_filt['test'] = X_test_filt['average_price_y']*X_test_filt.area_year_month
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import SGDRegressor, LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn import svm

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error



models = [LinearRegression(),

          GradientBoostingRegressor(),

          SGDRegressor(),

          RandomForestRegressor(),

          DecisionTreeRegressor()]



for model in models:

    # pipeline

    my_pipeline = Pipeline(steps=[

                                  ('model', model)

                                 ])



    # fit le modele 

    my_pipeline.fit(X_train_filt, y_train_filt)



    # prédiction

    preds = my_pipeline.predict(X_valid)

    print(model, 'MAE:', mean_absolute_error(y_valid, preds))
from sklearn import tree

import graphviz



model = DecisionTreeRegressor(random_state=0).fit(pd.concat([X_train_filt,X_valid]), pd.concat([y_train_filt,y_valid]))



 # prédiction

preds = model.predict(X_test_filt)

print('MAE', mean_absolute_error(y_test_filt, preds))
# MAE per borough

results = model.predict(X_test_filt) - y_test_filt

analyse_results = pd.DataFrame({'code': X_test_filt.code, 'results': results})

analyse_result_gb = analyse_results.groupby(by='code').mean()

city = table_borough.area.drop_duplicates()

city.index = range(1,34)

analyse_result_gb['city'] = city.loc[analyse_result_gb.index]

analyse_result_gb
# graph of results



plt.scatter(y=y_test_filt, x= range(len(preds)), label='Prices', c='b')

plt.scatter(y=preds, x=range(len(preds)), label='Predicted prices', c='r')



plt.title('graph of the price vs predicted prices')

plt.legend()
#Most important features

import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(model, random_state=0).fit(X_valid, y_valid)

table_w = eli5.explain_weights_dfs(perm, top=None, feature_names = X_valid.columns.tolist())['feature_importances']

col_unused = table_w.iloc[[41,42],0].values

new_X_train = X_train_filt.drop(col_unused, axis=1)

new_X_valid = X_valid.drop(col_unused, axis=1)

table_w
# How the features influence the outputs

from pdpbox import pdp, get_dataset, info_plots



def isolate(feat):

    feature_names = [i for i in X_train_filt.columns if X_train_filt[i].dtype in [np.int64]]

    # Create the data that we will plot

    pdp_goals = pdp.pdp_isolate(model=model, dataset=X_valid, model_features=X_valid.columns, feature=feat)



    # plot it

    pdp.pdp_plot(pdp_goals, feat)

    plt.show()



isolate('area_size_x')
isolate('recycling_pct_x')
isolate('number_of_jobs_x')