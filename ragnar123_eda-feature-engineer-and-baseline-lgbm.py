import pandas as pd

pd.set_option('display.max_columns', 999)

import numpy as np

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import math

import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
print('Loading trian set...')

train = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')

print('Loading test set...')

test = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv')

print('We have {} rows and {} columns in our train set'.format(train.shape[0], train.shape[1]))

print('We have {} rows and {} columns in our test set'.format(test.shape[0], test.shape[1]))
train.head()
test.head()
def missing_values(train):

    df = pd.DataFrame(train.isnull().sum()).reset_index()

    df.columns = ['Feature', 'Frequency']

    df['Percentage'] = (df['Frequency']/train.shape[0])*100

    df['Percentage'] = df['Percentage'].astype(str) + '%'

    df.sort_values('Percentage', inplace = True, ascending = False)

    return df



missing_values(train).head()
missing_values(test)
for i in ['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80', 'DistanceToFirstStop_p20', 

          'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']:

    plt.figure(figsize = (12, 8))

    plt.scatter(train.index, train[i])

    plt.title('{} distribution'.format(i))
def tv_ratio(train, column):

    df = train[train[column]==0]

    ratio = df.shape[0] / train.shape[0]

    return ratio



target_variables = ['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80', 

                    'DistanceToFirstStop_p20', 'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']



for i in target_variables:

    print('{} have a 0 ratio of: '.format(i), tv_ratio(train, i))
def plot_dist(train, test, column, type = 'kde', together = True):

    if type == 'kde':

        if together == False:

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,8))

            sns.kdeplot(train[column], ax = ax1, color = 'blue', shade=True)

            ax1.set_title('{} distribution of the train set'.format(column))

            sns.kdeplot(test[column], ax = ax2, color = 'red', shade=True)

            ax2.set_title('{} distribution of the test set'.format(column))

            plt.show()

        else:

            fig , ax = plt.subplots(1, 1, figsize = (12,8))

            sns.kdeplot(train[column], ax = ax, color = 'blue', shade=True, label = 'Train {}'.format(column))

            sns.kdeplot(test[column], ax = ax, color = 'red', shade=True, label = 'Test {}'.format(column))

            ax.set_title('{} Distribution'.format(column))

            plt.show()

    else:

        if together == False:

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,8))

            sns.distplot(train[column], ax = ax1, color = 'blue', kde = False)

            ax1.set_title('{} distribution of the train set'.format(column))

            sns.distplot(test[column], ax = ax2, color = 'red', kde = False)

            ax2.set_title('{} distribution of the test set'.format(column))

            plt.show()

        else:

            fig , ax = plt.subplots(1, 1, figsize = (12,8))

            sns.distplot(train[column], ax = ax, color = 'blue', kde = False)

            sns.distplot(test[column], ax = ax, color = 'red', kde = False)

            plt.show()

    

plot_dist(train, test, 'Latitude', type = 'kde', together = True)

plot_dist(train, test, 'Latitude', type = 'other', together = False)
plot_dist(train, test, 'Longitude', type = 'kde', together = True)

plot_dist(train, test, 'Longitude', type = 'other', together = False)
def scatter_plot(data, column1, column2, city = 'All'):

    if city == 'All':

        plt.figure(figsize = (12, 8))

        sns.scatterplot(data[column1], data[column2])

        plt.title('{} vs {} scatter plot'.format(column1, column2))

        plt.show()

    elif city == 'Atlanta':

        data1 = data[data['City']=='Atlanta']

        plt.figure(figsize = (12, 8))

        sns.scatterplot(data1[column1], data1[column2])

        plt.title('{} vs {} scatter plot for Atlanta city'.format(column1, column2))

        plt.show()

    elif city == 'Boston':

        data1 = data[data['City']=='Boston']

        plt.figure(figsize = (12, 8))

        sns.scatterplot(data1[column1], data1[column2])

        plt.title('{} vs {} scatter plot for Boston city'.format(column1, column2))

        plt.show()

    elif city == 'Chicago':

        data1 = data[data['City']=='Chicago']

        plt.figure(figsize = (12, 8))

        sns.scatterplot(data1[column1], data1[column2])

        plt.title('{} vs {} scatter plot for Chicago'.format(column1, column2))

        plt.show()

    elif city == 'Philadelphia':

        data1 = data[data['City']=='Philadelphia']

        plt.figure(figsize = (12, 8))

        sns.scatterplot(data1[column1], data1[column2])

        plt.title('{} vs {} scatter plot for Philadelphia'.format(column1, column2))

        plt.show()



        

scatter_plot(train, 'Latitude', 'Longitude')
scatter_plot(train, 'Latitude', 'Longitude', city = 'Atlanta')

scatter_plot(train, 'Latitude', 'Longitude', city = 'Boston')

scatter_plot(train, 'Latitude', 'Longitude', city = 'Chicago')

scatter_plot(train, 'Latitude', 'Longitude', city = 'Philadelphia')
def get_correlation(train, column):

    df = train[[column, 'TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80',

               'DistanceToFirstStop_p20', 'DistanceToFirstStop_p40', 'DistanceToFirstStop_p80']]

    correlation = df.corr()

    plt.figure(figsize = (12, 8))

    sns.heatmap(correlation, annot = True)

    return df

    

df = get_correlation(train, 'Latitude')

df = get_correlation(train, 'Longitude')
train.head()
def get_frec(df, column):

    df1 = pd.DataFrame(df[column].value_counts(normalize = True)).reset_index()

    df1.columns = [column, 'Percentage']

    df1.sort_values(column, inplace = True, ascending = True)

    return df1





def plot_frec(train, test, column):

    df = get_frec(train, column)

    df1 = get_frec(test, column)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,8))

    sns.barplot(df[column], df['Percentage'], ax = ax1, color = 'blue')

    ax1.set_title('{} percentages for the train set'.format(column))

    sns.barplot(df1[column], df1['Percentage'], ax = ax2, color = 'red')

    ax2.set_title('{} percentages for the test set'.format(column))

    

plot_frec(train, test, 'Month')
def get_target_mean(train, column, target_variables):

    df = train.groupby([column])[target_variables].agg(['mean']).reset_index()

    df.columns = [column] + [x + '_mean' for x in target_variables]

    return df



get_target_mean(train, 'Month', target_variables)
plot_frec(train, test, 'Hour')

get_target_mean(train, 'Hour', target_variables)
plot_frec(train, test, 'Weekend')

get_target_mean(train, 'Weekend', target_variables)
# city

plot_frec(train, test, 'City')

get_target_mean(train, 'City', target_variables)
# EntryStreetName

def n_unique_cat(train, test, col):

    n_u_train = train[col].nunique()

    n_u_test = test[col].nunique()

    df = pd.DataFrame({'n_u_train_{}'.format(col): [n_u_train], 'n_u_test_{}'.format(col): [n_u_train]})

    return df

n_unique_cat(train, test, 'EntryStreetName')
# ExitStreetName

n_unique_cat(train, test, 'ExitStreetName')
# EntryHeading

plot_frec(train, test, 'EntryHeading')

get_target_mean(train, 'EntryHeading', target_variables)
# ExitHeading

plot_frec(train, test, 'ExitHeading')

get_target_mean(train, 'ExitHeading', target_variables)
# IntersectionId

n_unique_cat(train, test, 'IntersectionId')
# Road Encoding



road_encoding = {'Street': 0, 'St': 0, 'Avenue': 1, 'Ave': 1, 'Boulevard': 2, 'Road': 3,

                'Drive': 4, 'Lane': 5, 'Tunnel': 6, 'Highway': 7, 'Way': 8, 'Parkway': 9,

                'Parking': 10, 'Oval': 11, 'Square': 12, 'Place': 13, 'Bridge': 14}



def encode(x):

    if pd.isna(x):

        return 0

    for road in road_encoding.keys():

        if road in x:

            return road_encoding[road]

    return 0



for par in [train, test]:

    par['EntryType'] = par['EntryStreetName'].apply(encode)

    par['ExitType'] = par['ExitStreetName'].apply(encode)

    par['EntryType_1'] = pd.Series(par['EntryStreetName'].str.split().str.get(0))

    par['ExitType_1'] = pd.Series(par['ExitStreetName'].str.split().str.get(0))

    par['EntryType_2'] = pd.Series(par['EntryStreetName'].str.split().str.get(1))

    par['ExitType_2'] = pd.Series(par['ExitStreetName'].str.split().str.get(1))

    par.loc[par['EntryType_1'].isin(par['EntryType_1'].value_counts()[par['EntryType_1'].value_counts()<=500].index), 'EntryType_1'] = 'Other'

    par.loc[par['ExitType_1'].isin(par['ExitType_1'].value_counts()[par['ExitType_1'].value_counts()<=500].index), 'ExitType_1'] = 'Other'

    par.loc[par['EntryType_2'].isin(par['EntryType_2'].value_counts()[par['EntryType_2'].value_counts()<=500].index), 'EntryType_2'] = 'Other'

    par.loc[par['ExitType_2'].isin(par['ExitType_2'].value_counts()[par['ExitType_2'].value_counts()<=500].index), 'ExitType_2'] = 'Other'

    

    

    

    



# The cardinal directions can be expressed using the equation: θ/π

# Where  θ  is the angle between the direction we want to encode and the north compass direction, measured clockwise.

directions = {'N': 0, 'NE': 1/4, 'E': 1/2, 'SE': 3/4, 'S': 1, 'SW': 5/4, 'W': 3/2, 'NW': 7/4}

train['EntryHeading'] = train['EntryHeading'].map(directions)

train['ExitHeading'] = train['ExitHeading'].map(directions)

test['EntryHeading'] = test['EntryHeading'].map(directions)

test['ExitHeading'] = test['ExitHeading'].map(directions)



# EntryStreetName == ExitStreetName ?

# EntryHeading == ExitHeading ?

for par in [train, test]:

    par["same_street_exact"] = (par["EntryStreetName"] ==  par["ExitStreetName"]).astype(int)

    par["same_heading_exact"] = (par["EntryHeading"] ==  par["ExitHeading"]).astype(int)

    

# We have some intersection id that are in more than one city, it is a good idea to feature cross them

for par in [train, test]:

    par['Intersection'] = par['IntersectionId'].astype(str) + '_' + par['City'].astype(str)

    

# Add temperature (°F) of each city by month

monthly_av = {'Atlanta1': 43, 'Atlanta5': 69, 'Atlanta6': 76, 'Atlanta7': 79, 'Atlanta8': 78, 'Atlanta9': 73,

              'Atlanta10': 62, 'Atlanta11': 53, 'Atlanta12': 45, 'Boston1': 30, 'Boston5': 59, 'Boston6': 68,

              'Boston7': 74, 'Boston8': 73, 'Boston9': 66, 'Boston10': 55,'Boston11': 45, 'Boston12': 35,

              'Chicago1': 27, 'Chicago5': 60, 'Chicago6': 70, 'Chicago7': 76, 'Chicago8': 76, 'Chicago9': 68,

              'Chicago10': 56,  'Chicago11': 45, 'Chicago12': 32, 'Philadelphia1': 35, 'Philadelphia5': 66,

              'Philadelphia6': 76, 'Philadelphia7': 81, 'Philadelphia8': 79, 'Philadelphia9': 72, 'Philadelphia10': 60,

              'Philadelphia11': 49, 'Philadelphia12': 40}



for par in [train, test]:

    # Concatenating the city and month into one variable

    par['city_month'] = par["City"].astype(str) + par["Month"].astype(str)

    # Creating a new column by mapping the city_month variable to it's corresponding average monthly temperature

    par["average_temp"] = par['city_month'].map(monthly_av)

    

# Add climate data

monthly_rainfall = {'Atlanta1': 5.02, 'Atlanta5': 3.95, 'Atlanta6': 3.63, 'Atlanta7': 5.12, 'Atlanta8': 3.67, 'Atlanta9': 4.09, 

                    'Atlanta10': 3.11, 'Atlanta11': 4.10, 'Atlanta12': 3.82, 'Boston1': 3.92, 'Boston5': 3.24, 'Boston6': 3.22, 

                    'Boston7': 3.06, 'Boston8': 3.37, 'Boston9': 3.47, 'Boston10': 3.79,'Boston11': 3.98, 'Boston12': 3.73, 

                    'Chicago1': 1.75, 'Chicago5': 3.38, 'Chicago6': 3.63, 'Chicago7': 3.51, 'Chicago8': 4.62, 'Chicago9': 3.27, 

                    'Chicago10': 2.71,  'Chicago11': 3.01, 'Chicago12': 2.43, 'Philadelphia1': 3.52, 'Philadelphia5': 3.88, 

                    'Philadelphia6': 3.29, 'Philadelphia7': 4.39, 'Philadelphia8': 3.82, 'Philadelphia9':3.88 , 

                    'Philadelphia10': 2.75, 'Philadelphia11': 3.16, 'Philadelphia12': 3.31}



monthly_snowfall = {'Atlanta1': 0.6, 'Atlanta5': 0, 'Atlanta6': 0, 'Atlanta7': 0, 'Atlanta8': 0, 'Atlanta9': 0, 

                    'Atlanta10': 0, 'Atlanta11': 0, 'Atlanta12': 0.2, 'Boston1': 12.9, 'Boston5': 0, 'Boston6': 0, 

                    'Boston7': 0, 'Boston8': 0, 'Boston9': 0, 'Boston10': 0,'Boston11': 1.3, 'Boston12': 9.0, 

                    'Chicago1': 11.5, 'Chicago5': 0, 'Chicago6': 0, 'Chicago7': 0, 'Chicago8': 0, 'Chicago9': 0, 

                    'Chicago10': 0,  'Chicago11': 1.3, 'Chicago12': 8.7, 'Philadelphia1': 6.5, 'Philadelphia5': 0, 

                    'Philadelphia6': 0, 'Philadelphia7': 0, 'Philadelphia8': 0, 'Philadelphia9':0 , 'Philadelphia10': 0, 

                    'Philadelphia11': 0.3, 'Philadelphia12': 3.4}



monthly_daylight = {'Atlanta1': 10, 'Atlanta5': 14, 'Atlanta6': 14, 'Atlanta7': 14, 'Atlanta8': 13, 'Atlanta9': 12, 

                    'Atlanta10': 11, 'Atlanta11': 10, 'Atlanta12': 10, 'Boston1': 9, 'Boston5': 15, 'Boston6': 15, 

                    'Boston7': 15, 'Boston8': 14, 'Boston9': 12, 'Boston10': 11,'Boston11': 10, 'Boston12': 9, 

                    'Chicago1': 10, 'Chicago5': 15, 'Chicago6': 15, 'Chicago7': 15, 'Chicago8': 14, 'Chicago9': 12, 

                    'Chicago10': 11,  'Chicago11': 10, 'Chicago12': 9, 'Philadelphia1': 10, 'Philadelphia5': 14, 

                    'Philadelphia6': 15, 'Philadelphia7': 15, 'Philadelphia8': 14, 'Philadelphia9':12 , 'Philadelphia10': 11, 

                    'Philadelphia11': 10, 'Philadelphia12': 9}



monthly_sunshine = {'Atlanta1': 5.3, 'Atlanta5': 9.3, 'Atlanta6': 9.5, 'Atlanta7': 8.8, 'Atlanta8': 8.3, 'Atlanta9': 7.6, 

                    'Atlanta10': 7.7, 'Atlanta11': 6.2, 'Atlanta12': 5.3, 'Boston1': 5.3, 'Boston5': 8.6, 'Boston6': 9.6, 

                    'Boston7': 9.7, 'Boston8': 8.9, 'Boston9': 7.9, 'Boston10': 6.7,'Boston11': 4.8, 'Boston12': 4.6, 

                    'Chicago1': 4.4, 'Chicago5': 9.1, 'Chicago6': 10.4, 'Chicago7': 10.3, 'Chicago8': 9.1, 'Chicago9': 7.6, 

                    'Chicago10': 6.2,  'Chicago11': 3.6, 'Chicago12': 3.4, 'Philadelphia1': 5.0, 'Philadelphia5': 7.9, 

                    'Philadelphia6': 9.0, 'Philadelphia7': 8.9, 'Philadelphia8': 8.4, 'Philadelphia9':7.9 , 

                    'Philadelphia10': 6.6,  'Philadelphia11': 5.2, 'Philadelphia12': 4.4}





for par in [train, test]:

    # Creating a new column by mapping the city_month variable to it's corresponding average monthly rainfall

    par["average_rainfall"] = par['city_month'].map(monthly_rainfall)

    # Creating a new column by mapping the city_month variable to it's corresponding average monthly snowfall

    par['average_snowfall'] = par['city_month'].map(monthly_snowfall)

    # Creating a new column by mapping the city_month variable to it's corresponding average monthly daylight

    par["average_daylight"] = par['city_month'].map(monthly_daylight)

    # Creating a new column by mapping the city_month variable to it's corresponding average monthly sunsine

    par["average_sunshine"] = par['city_month'].map(monthly_sunshine)

    

# drop city month

train.drop('city_month', axis=1, inplace=True)

test.drop('city_month', axis=1, inplace=True)



# Add feature is day

train['is_day'] = train['Hour'].apply(lambda x: 1 if 5 < x < 20 else 0)

test['is_day'] = test['Hour'].apply(lambda x: 1 if 5 < x < 20 else 0)



# fill NaN categories

train.fillna(-999, inplace = True)

test.fillna(-999, inplace = True)





# distance from the center of the city

def add_distance(df):

    

    df_center = pd.DataFrame({"Atlanta":[33.753746, -84.386330],

                             "Boston":[42.361145, -71.057083],

                             "Chicago":[41.881832, -87.623177],

                             "Philadelphia":[39.952583, -75.165222]})

    

    df["CenterDistance"] = df.apply(lambda row: math.sqrt((df_center[row.City][0] - row.Latitude) ** 2 +

                                                          (df_center[row.City][1] - row.Longitude) ** 2) , axis=1)



add_distance(train)

add_distance(test)





# frequency encode

def encode_FE(df1, df2, cols):

    for col in cols:

        df = pd.concat([df1[col],df2[col]])

        vc = df.value_counts(dropna=True, normalize=True).to_dict()

        nm = col+'_FE'

        df1[nm] = df1[col].map(vc)

        df1[nm] = df1[nm].astype('float32')

        df2[nm] = df2[col].map(vc)

        df2[nm] = df2[nm].astype('float32')

        print(nm,', ',end='')

        

# COMBINE FEATURES

def encode_CB(col1, col2 , df1 = train, df2 = test):

    nm = col1+'_'+col2

    df1[nm] = df1[col1].astype(str)+'_'+df1[col2].astype(str)

    df2[nm] = df2[col1].astype(str)+'_'+df2[col2].astype(str) 

    print(nm,', ',end='')

    

# group aggregations nunique

def encode_AG2(main_columns, agg_col, train_df = train, test_df = test):

    for main_column in main_columns:  

        for col in agg_col:

            comb = pd.concat([train_df[[col]+[main_column]],test_df[[col]+[main_column]]],axis=0)

            mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()

            train_df[col+'_'+main_column+'_ct'] = train_df[col].map(mp).astype('float32')

            test_df[col+'_'+main_column+'_ct'] = test_df[col].map(mp).astype('float32')

            print(col+'_'+main_column+'_ct, ',end='')



def encode_AG(main_columns, agg_col, aggregations=['mean'], train_df = train, test_df = test, fillna=True, usena=False):

    # aggregation of main agg_cols

    for main_column in main_columns:  

        for col in agg_col:

            for agg_type in aggregations:

                new_col_name = main_column+'_'+col+'_'+agg_type

                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])

                if usena: temp_df.loc[temp_df[main_column]==-1,main_column] = np.nan

                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(

                                                        columns={agg_type: new_col_name})



                temp_df.index = list(temp_df[col])

                temp_df = temp_df[new_col_name].to_dict()   



                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')

                test_df[new_col_name]  = test_df[col].map(temp_df).astype('float32')

                

                if fillna:

                    train_df[new_col_name].fillna(-1,inplace=True)

                    test_df[new_col_name].fillna(-1,inplace=True)

                

                print("'"+new_col_name+"'",', ',end='')

                

# Frequency encode 

encode_FE(train, test, ['Hour', 'Month', 'EntryType', 'ExitType', 'EntryType_1', 'EntryType_2', 'ExitType_1', 'ExitType_2', 'Intersection', 'City'])

                

# Agreggations of main columns

encode_AG(['Longitude', 'Latitude', 'CenterDistance', 'EntryHeading', 'ExitHeading'], ['Hour', 'Weekend', 'Month', 'Intersection'], ['mean', 'std'])



# bucketize lat and lon

temp_df = pd.concat([train[['Latitude', 'Longitude']], test[['Latitude', 'Longitude']]]).reset_index(drop = True)

temp_df['Latitude_B'] = pd.cut(temp_df['Latitude'], 30)

temp_df['Longitude_B'] = pd.cut(temp_df['Longitude'], 30)



# feature cross lat and lon

temp_df['Latitude_B_Longitude_B'] = temp_df['Latitude_B'].astype(str) + '_' + temp_df['Longitude_B'].astype(str)

train['Latitude_B'] = temp_df.loc[:(train.shape[0]), 'Latitude_B']

test['Latitude_B'] = temp_df.loc[(train.shape[0]):, 'Latitude_B']

train['Longitude_B'] = temp_df.loc[:(train.shape[0]), 'Longitude_B']

test['Longitude_B'] = temp_df.loc[(train.shape[0]):, 'Longitude_B']

train['Latitude_B_Longitude_B'] = temp_df.loc[:(train.shape[0]), 'Latitude_B_Longitude_B']

test['Latitude_B_Longitude_B'] = temp_df.loc[(train.shape[0]):, 'Latitude_B_Longitude_B']



# feature crosses hour with month

encode_CB('Hour', 'Month')



# group aggregations nunique 

encode_AG2(['Intersection', 'Latitude_B_Longitude_B'], ['Hour', 'Month'])



# label encode

for i,f in enumerate(train.columns):

    if (np.str(train[f].dtype)=='category')|(train[f].dtype=='object'): 

        df_comb = pd.concat([train[f],test[f]],axis=0)

        df_comb,_ = df_comb.factorize(sort=True)

        if df_comb.max()>32000: print(f,'needs int32')

        train[f] = df_comb[:len(train)].astype('int16')

        test[f] = df_comb[len(train):].astype('int16')
# drop useless features

for par in [train, test]:

    par.drop(['RowId', 'Path', 'EntryStreetName', 'ExitStreetName'], axis = 1, inplace = True)

    

# drop target variables from the train set

preds = train.iloc[:,8:23]

# get target_variables

target1 = preds['TotalTimeStopped_p20']

target2 = preds['TotalTimeStopped_p50']

target3 = preds['TotalTimeStopped_p80']

target4 = preds['DistanceToFirstStop_p20']

target5 = preds['DistanceToFirstStop_p50']

target6 = preds['DistanceToFirstStop_p80']

train.drop(preds.columns.tolist(), axis=1, inplace =True)
# train lgb

param = {'num_leaves': 230, 

         'feature_fraction': 0.8115011063299449,

         'bagging_fraction': 0.9557214979912946,

         'max_depth': 19,

         'lambda_l1': 1.1159237398459447,

         'lambda_l2': 0.7092738973066476,

         'min_split_gain': 0.007200100317150616,

         'min_child_weight': 19.751392371168137,

         'learning_rate': 0.05,

         'objective': 'regression',

         'boosting_type': 'gbdt',

         'verbose': 1,'metric': 'rmse',

         'seed': 7}



def run_lgb(train, test):

    # get prediction dictonary were we are going to store predictions

    all_preds = {0 : [], 1 : [], 2 : [], 3 : [], 4 : [], 5 : []}

    # get a list with all the target variables

    all_target = [target1, target2, target3, target4, target5, target6]

    nfold = 5

    kf = KFold(n_splits=nfold, random_state=228, shuffle=True)

    for i in range(len(all_preds)):

        print('Training and predicting for target {}'.format(i+1))

        oof = np.zeros(len(train))

        all_preds[i] = np.zeros(len(test))

        n = 1

        for train_index, valid_index in kf.split(all_target[i]):

            print("fold {}".format(n))

            xg_train = lgb.Dataset(train.iloc[train_index],

                                   label=all_target[i][train_index]

                                   )

            xg_valid = lgb.Dataset(train.iloc[valid_index],

                                   label=all_target[i][valid_index]

                                   )   



            clf = lgb.train(param, xg_train, 10000, valid_sets=[xg_train, xg_valid], 

                            verbose_eval=100, early_stopping_rounds=200)

            oof[valid_index] = clf.predict(train.iloc[valid_index], num_iteration=clf.best_iteration) 



            all_preds[i] += clf.predict(test, num_iteration=clf.best_iteration) / nfold

            n = n + 1



        print("\n\nCV RMSE: {:<0.4f}".format(np.sqrt(mean_squared_error(all_target[i], oof))))

    return all_preds



all_preds = run_lgb(train, test)
submission = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/sample_submission.csv')

data2 = pd.DataFrame(all_preds).stack()

data2 = pd.DataFrame(data2)

submission['Target'] = data2[0].values

submission.to_csv('lgbm_baseline.csv', index=False)