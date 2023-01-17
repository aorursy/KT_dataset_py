#invite everybody

import numpy as np

import pandas as pd

from fastai.imports import *

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm, skew

from scipy import stats

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.preprocessing import MinMaxScaler, RobustScaler

import os



import warnings

warnings.simplefilter('ignore')



#function to display all columns, ideally pass a transpose to this function like df.head().T

def display_all(df) :

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)



def import_csv(filename): 

    df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/" + filename + ".csv")

    print(filename + '.csv loaded...')

    print(filename + ' set shape: ',df.shape)

    return df



def get_num_and_cat_features(df):

    num_features = [c for c in df.columns if df[c].dtype in ['int64', 'float64']]

    cat_features = [c for c in df.columns if df[c].dtype not in ['int64', 'float64']]

    return num_features, cat_features



def fix_missing_data_replace(df, impute_method, num_features, cat_features):

    #Replace NaN with None for categorical columns

    df[cat_features] = df[cat_features].apply( lambda x: x.fillna("None"), axis = 0)



    #Impute numerical features

    #Some numerical features will have to be replaced by 0 if no value exists for them. This makes sense as these values were 0 they were never updated by the collector and hence have NA

    for col in ('GarageCars', 'GarageArea') :

        df[col].fillna(0.0, inplace=True)

        

    if impute_method == 'median':

        df[num_features] = df[num_features].apply(lambda x: x.fillna(x.median), axis=0)

        print("Missing values imputed with median.")

    elif impute_method == 'mean':

        df[num_features] = df[num_features].apply(lambda x: x.fillna(x.mean(skipna=True)), axis=0)

        print("Missing values imputed with mean.")



    elif impute_method == 'mode':

        df[num_features] = df[num_features].apply(lambda x: x.fillna(x.mode), axis=0)

        print("Missing values imputed with mode.")

    return df



def feature_engineering(df):

    

    df['YrBltAndRemod'] = df['YearBuilt'] + df['YearRemodAdd']

    

    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']



    df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF'])



    df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))



    df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])



    print("Feature engineering: added combination of features.")

    

    df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

    df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

    df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

    df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

    df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    print("Feature engineering: added boolean features.")

    

    print('Shape of dataframe after feature engineering: ', df.shape)

    

    return df



def feature_transformation(df, feature_transform, num_features):

    if feature_transform == 'yes':

        skew_features = df[num_features].apply(lambda x: skew(x)).sort_values(ascending=False)

        high_skew = skew_features[skew_features > 0.5]

        skew_index = high_skew.index

        for i in skew_index:

            df[i] = boxcox1p(df[i], boxcox_normmax(df[i]+1))

        print("Transformed numerical features with high skewness factor.")

    

    return df



def feature_scaling_p(df, num_features, feature_scaling):

    scaler = None

    if feature_scaling == 'off':

        print('Feature scaling is off')

        return df

    elif feature_scaling == 'RobustScaler':

        scaler = RobustScaler()

    elif feature_scaling == 'StandardScaler':

        scaler = StandardScaler()

    elif feature_scaling == 'MinMaxScaler':

        scaler = MinMaxScaler(feature_range=(0, 1))

    

    for col in num_features:

        df[[col]] = scaler.fit_transform(df[[col]])

    print('Feature scaling completed successfully : ', df.shape)

    

    return df



def feature_selection_p(df, feature_selection):

    if feature_selection == 'no':

        print('Feature selection is off')

        return df

    

    #let's remove columns with little variance (to reduce overfitting)

    overfit = []

    

    for i in df.columns:

        counts = df[i].value_counts()

        zeros = counts.iloc[0]

        if zeros / len(df) * 100 > 99.9: # the threshold is set at 99.9%

            overfit.append(i)

            

    overfit = list(overfit)

    # let's make sure to keep data processing columns needed later on

    try:

        overfit.remove('dataset_Train')

        overfit.remove('dataset_Test')

    except:

        pass

    df.drop(overfit, axis=1, inplace=True)

    #print('overfit[] columns: ')

    #display_all(overfit)

    print('To prevent overfitting, {} columns were removed.'.format(len(overfit)))

    

    print('Feature selection completed: ', df.shape)

    return df



def transform_feature(features, method = 'log'):

    if method == 'log':

        print('Transforming features with log: ', features.shape)

        return np.log1p(features)

    elif method == 'sqrt':

        print('Transforming features with sqrt: ', features.shape)

        return np.sqrt(features)

    elif method == 'square':

        print('Transforming features with square: ', features.shape)

        return np.square(features)



def pre_process(df_raw, 

                feature_transform = 'yes',

                feature_scaling = 'RobustScaler',

                impute_method = 'mean',

                feature_selection= 'yes'):

    

    #Make copy of raw dataset for summary purpose at the end

    df = df_raw.copy()

    

    #We should drop Id field which is just a unique identifier

    df.drop('Id', axis = 1, inplace=True)

    print('Dataframe shape after droping Id field: ', df.shape)

    

    #Seperate numerical and categorical features

    num_features, cat_features = get_num_and_cat_features(df)

    print("Numerical features len: {}\nCategorical Features len: {}".format( len(num_features), len(cat_features)))

    for col in num_features:

        df[col] = df[col].astype(float)

    

    #Deal with missing data

    df = fix_missing_data_replace(df, impute_method, num_features, cat_features)

    

    #Feature engineering

    #Decompose features or combine features or add new feature with the help of existing features

    df = feature_engineering(df)

    num_features, cat_features = get_num_and_cat_features(df)

    

    #Feature transformation like log, sqrt, x2 etc.

    df = feature_transformation(df, feature_transform, num_features)

    

    #Label Encoding

    df = pd.get_dummies(df)

    print("Shape of df after Label encoding: ", df.shape)

    

    #Feature scaling

    df = feature_scaling_p(df, num_features, feature_scaling)

    

    #Feature selection

    df = feature_selection_p(df, feature_selection)

    

    #Summary

    print('\nShape of original dataset: {}\nShape of transformed dataset: {}'.format(df_raw.shape, df.shape))

    

    return df





def get_missing_data(df_train):

    total = df_train.isnull().sum().sort_values(ascending=False)

    percent = (df_train.isnull().sum()/df_train.isnull().count() * 100).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    

    return missing_data



def remove_unwanted_columns(df_train):

    missing_data = get_missing_data(df_train)

    df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

    print("Mising values after removing unwanted columns based on missing data: ", df_train.isnull().sum().max())

    

    return df_train



def combine_training_and_test_sets(df_train, df_test):

    #Set dataset attribute to later extract these out after preprocessing

    df_train['dataset'] = 'Train'

    df_test['dataset'] = 'Test'

    df_combine = pd.concat([df_train, df_test], sort=False)

    df_combine = df_combine.reset_index(drop=True)

    print('Combined dataframe shape: ', df_combine.shape)

    

    return df_combine



def plot_distplot(y):

    sns.distplot(y, fit=norm);

    fig = plt.figure()

    #res = stats.probplot(y, plot=plt)

    

def seperate_out_set(df, dataset_name):

    #Seperate out dataset_name dataset from df

    df_to_return = df[df[dataset_name]==1].copy()

    

    #Remove redundant metadata columns

    df_to_return.drop(['dataset_Train'], axis=1, inplace=True)

    df_to_return.drop(['dataset_Test'], axis=1, inplace=True)

    df_to_return = df_to_return.reset_index(drop=True)

    

    return df_to_return



def rmse(x, y):

    return sqrt(((x-y)**2).mean())



def compare_graph_n_estimators(X_train, y_train, X_test, y_test):

    n_estimators = [ e for e in range(50, 200, 50)]

    scores = []

    for e in n_estimators : 

        scores.append(rmse(RandomForestRegressor(n_estimators=e)

                           .fit(X_train, y_train)

                           .predict(X_test), y_test))

    #print(pd.DataFrame([scores, n_estimators]).head().T)

    plt.plot(n_estimators, scores)



def begin_training(df, y):

    print('\nTraining is begining...')

    

    #seperate out training set first

    df_train_ml = seperate_out_set(df, 'dataset_Train')

    print('Shape of training dataframe: ', df_train_ml.shape)

    

    #seperate out test set next

    df_test_ml = seperate_out_set(df, 'dataset_Test')

    print('Shape of test dataframe: ', df_test_ml.shape)

    

    print('\nSplitting training data into train and test sets...')

    X_train, X_test, y_train, y_test = train_test_split(df_train_ml,

                                                    y,

                                                    test_size=0.2,

                                                    stratify=df_train_ml['OverallQual'],

                                                    random_state=42)

    print('Training data shape: ', df_train_ml.shape)

    print('X_train shape: ', X_train.shape)

    print('X_test shape: ', X_test.shape)

    

    model = RandomForestRegressor(n_estimators=100,

                                  random_state=0,

                                  n_jobs=-1,

                                  min_samples_leaf= 2 ,

                                  oob_score = True);

    %time model.fit(X_train, y_train)

    

    #compare_graph_n_estimators(X_train, y_train, X_test, y_test)

    

    #predictions = model.predict(X_test)

    score = [rmse(model.predict(X_train), y_train), rmse(model.predict(X_test), y_test),

             model.score(X_train, y_train), model.score(X_test, y_test)]

    

    print(score)

    

    return model, df_test_ml 



def main():

    #import the raw data

    df_train_raw = import_csv("train")

    df_test_raw = import_csv("test")



    #copy raw data

    df_train = df_train_raw.copy()

    df_test = df_test_raw.copy()



    #Seperate target variable and drop it from training set

    y = df_train.SalePrice

    df_train = df_train.drop('SalePrice', axis= 1)

    print('Training set shape after droping target variable SalePrice', df_train.shape)

    

    #Remove some columns based on missing data

    #df_train = remove_unwanted_columns(df_train)

    

    #Just keep same columns for test set

    df_test = df_test[df_train.columns]

    

    #add training and test sets together so preprocessing will become a one time event

    #just confirm they have the same shape

    print('\nTraining set shape : {}\nTest set shape : {}\n'.format(df_train.shape, df_test.shape))

    df_combine = combine_training_and_test_sets(df_train, df_test)

    

    df = pre_process(df_combine, impute_method="mean")

    y = transform_feature(y)

    

    #Lets confirm once there is no missing data

    print('Missing data count: ', get_missing_data(df)['Total'].sum())

    

    #Lets plot the y feature which out dependent/target feature

    #plot_distplot(y)

    print('Skewness of y (SalePrice) : ', y.skew())

    

    #Begin prediction step

    model, df_test_ml = begin_training(df, y)

    

    #return df, df_test, df_combine

    return model, df_test_ml

    

    



model, df_test_ml = main()

def predict_and_write_to_csv(model, df_test_ml):

    pred = model.predict(df_test_ml)

    ids = [int(x) for x in range(1461,2920)]

    final = pd.DataFrame(data = [ids, np.exp(1)**pred], index= ['Id', 'SalePrice']).T

    final['Id'] = final['Id'].astype(int)

    final.to_csv('submission.csv', index=False)

    return final

    



f = predict_and_write_to_csv(model, df_test_ml)