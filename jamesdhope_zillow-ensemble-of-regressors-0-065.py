# Load in our libraries

import pandas as pd

import numpy as np

import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from sklearn.svm import LinearSVR

from sklearn.cross_validation import KFold;
# Load in the train and test datasets

train = pd.read_csv('../input/properties_2016.csv')

train_label = pd.read_csv('../input/train_2016_v2.csv')



# Store our passenger ID for easy access

ParcelID = train['parcelid']
train.head(3)
# OneHotEncoding

train['has_basement'] = train["basementsqft"].apply(lambda x: 0 if np.isnan(x) else 1).astype(float)

train['hashottuborspa'] = train["hashottuborspa"].apply(lambda x: 0 if np.isnan(x) else 1).astype(float)

train['has_pool'] = train["poolcnt"].apply(lambda x: 0 if np.isnan(x) else 1).astype(float)

train['has_airconditioning'] = train["airconditioningtypeid"].apply(lambda x: 0 if np.isnan(x) else 1).astype(float)



# Columns to be consolidated

train['yardbuildingsqft17'] = train['yardbuildingsqft17'].apply(lambda x: 0 if np.isnan(x) else x).astype(float)

train['yardbuildingsqft26'] = train['yardbuildingsqft26'].apply(lambda x: 0 if np.isnan(x) else x).astype(float)

train['yard_building_square_feet'] = train['yardbuildingsqft17'].astype(int) + train['yardbuildingsqft26'].astype(float)



# Assume some more friendly feature names

train.rename(columns={'fireplacecnt':'fireplace_count'}, inplace=True)

train.rename(columns={'bedroomcnt':'bedroom_count'}, inplace=True)

train.rename(columns={'bathroomcnt':'bathroom_count'}, inplace=True)

train.rename(columns={'calculatedfinishedsquarefeet':'square_feet'}, inplace=True)

train.rename(columns={'garagecarcnt':'garage_car_count'}, inplace=True)

train.rename(columns={'garagetotalsqft':'garage_square_feet'}, inplace=True)

train.rename(columns={'hashottuborspa':'has_hottub_or_spa'}, inplace=True)



train.rename(columns={'landtaxvaluedollarcnt':'land_tax'}, inplace=True)

train.rename(columns={'lotsizesquarefeet':'lot_size_square_feet'}, inplace=True)

train.rename(columns={'taxvaluedollarcnt':'tax_value'}, inplace=True)

train.rename(columns={'taxamount':'tax_amount'}, inplace=True)

train.rename(columns={'structuretaxvaluedollarcnt':'structure_tax_value'}, inplace=True)

train.rename(columns={'yearbuilt':'year_built'}, inplace=True)



train.rename(columns={'roomcnt':'room_count'}, inplace=True)
# Impute zero for NaN for these features

train['fireplace_count'] = train['fireplace_count'].apply(lambda x: 0 if np.isnan(x) else x).astype(float)



# Impute median value for NaN for these features

train['bathroom_count'] = train['bathroom_count'].fillna(train['bathroom_count'].median()).astype(float)

train['bedroom_count'] = train['bedroom_count'].fillna(train['bedroom_count'].median()).astype(float)

train['room_count'] = train['room_count'].fillna(train['room_count'].median()).astype(float)



train['tax_amount'] = train['tax_amount'].fillna(train['tax_amount'].median()).astype(float)

train['land_tax'] = train['land_tax'].fillna(train['land_tax'].median()).astype(float)

train['tax_value'] = train['tax_value'].fillna(train['tax_value'].median()).astype(float)

train['structure_tax_value'] = train['structure_tax_value'].fillna(train['structure_tax_value'].median()).astype(float)

train['garage_square_feet'] = train['garage_square_feet'].fillna(train['garage_square_feet'].median()).astype(float)

train['garage_car_count'] = train['garage_car_count'].fillna(train['garage_car_count'].median()).astype(float)

train['fireplace_count'] = train['fireplace_count'].fillna(train['fireplace_count'].median()).astype(float)

train['square_feet'] = train['square_feet'].fillna(train['square_feet'].median()).astype(float)

train['year_built'] = train['year_built'].fillna(train['year_built'].median()).astype(float)

train['lot_size_square_feet'] = train['lot_size_square_feet'].fillna(train['lot_size_square_feet'].median()).astype(float)



train['longitude'] = train['longitude'].fillna(train['longitude'].median()).astype(float)

train['latitude'] = train['latitude'].fillna(train['latitude'].median()).astype(float)
# Drop indistinct features

drop_elements = ['assessmentyear']



# Drop any columns insufficiently described

drop_elements = drop_elements + ['airconditioningtypeid', 'basementsqft', 'architecturalstyletypeid', 'buildingclasstypeid', 'buildingqualitytypeid', 'calculatedbathnbr', 'decktypeid', 'finishedfloor1squarefeet',

                 'fips', 'heatingorsystemtypeid', 'rawcensustractandblock',

                 'numberofstories', 'storytypeid', 'threequarterbathnbr', 'typeconstructiontypeid', 'unitcnt', 'censustractandblock', 'fireplaceflag', 'taxdelinquencyflag', 'taxdelinquencyyear',

                ]



# Drop any duplicated columns

drop_elements = drop_elements + ['fullbathcnt', 'finishedsquarefeet6', 'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50', 'yardbuildingsqft17', 'yardbuildingsqft26']



# Land use data

drop_elements = drop_elements + ['propertycountylandusecode', 'propertylandusetypeid', 'propertyzoningdesc']



# We'll make do with a binary feature here

drop_elements = drop_elements + ['pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'poolsizesum', 'poolcnt']



# We'll use the longitude and latitutde as features 

drop_elements = drop_elements + ['regionidzip', 'regionidneighborhood', 'regionidcity', 'regionidcounty']



print("dropping features", drop_elements)



train = train.drop(drop_elements, axis = 1)
train.head(5)
# Create Numpy arrays of train, test and target dataframes to feed into our models

import pandas as pd

import random



# Obtain the training by merging both train and train_label on the parcelid

common = train.merge(train_label,on=['parcelid'])

#common.to_csv('common.csv')



# Split the merged set into the training set and labels

y_train = common['logerror']

x_train = common.drop(['logerror', 'parcelid'], axis=1)



# OneHotEncode the date information and drop the original

x_train = common

x_train['year'], x_train['month'], x_train['day'] = x_train['transactiondate'].str.split('-').str

x_train = common.drop(['transactiondate'], axis=1)
#config

colormap = plt.cm.viridis

plt.figure(figsize=(15,15))

plt.title('Pearson Correlation of Features', y=1.05, size=15)



#create the correlation map

corr = x_train.astype(float).corr()



#create a mask for the null values

mask = corr.isnull()



#plot the heatmap

sns.heatmap(corr, mask=mask, linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
x_train = x_train.drop(['logerror', 'parcelid'], axis=1)



# Convert training set and labels to Numpy array

x_train = x_train.values

y_train = y_train.values
# Obtain the full test set, with the date for each time period as an additional feature

train = train.drop(['parcelid'], axis=1)

#train.to_csv('train.csv')



def get_random_day():

    return pd.DataFrame(np.random.randint(1,30,len(train)))

                   

# Create a Test Set with date column for each time period we are being asked to predict for

x_test_201610 = train

x_test_201610['year'], x_test_201610['month'], x_test_201610['day'] = [2016,10,get_random_day()]

#x_test_201610.to_csv("x_train_201610.csv")

x_test_201610 = x_test_201610.values



x_test_201611 = train

x_test_201611['year'], x_test_201611['month'], x_test_201611['day'] = [2016,11,get_random_day()]

x_test_201611 = x_test_201611.values



x_test_201612 = train

x_test_201612['year'], x_test_201612['month'], x_test_201612['day'] = [2016,12,get_random_day()]

x_test_201612 = x_test_201612.values



x_test_201710 = train

x_test_201710['year'], x_test_201710['month'], x_test_201710['day'] = [2017,10,get_random_day()]

x_test_201710 = x_test_201710.values



x_test_201711 = train

x_test_201711['year'], x_test_201711['month'], x_test_201711['day'] = [2017,11,get_random_day()]

x_test_201711 = x_test_201711.values



x_test_201712 = train

x_test_201712['year'], x_test_201712['month'], x_test_201712['day'] = [2017,12,get_random_day()]

#x_test_201712.to_csv("x_train_201712.csv")

x_test_201712 = x_test_201712.values



print("test sets populated with random transaction dates")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test_201610 = scaler.transform(x_test_201610)

x_test_201611 = scaler.transform(x_test_201611)

x_test_201612 = scaler.transform(x_test_201612)

x_test_201710 = scaler.transform(x_test_201710)

x_test_201711 = scaler.transform(x_test_201711)

x_test_201712 = scaler.transform(x_test_201712)
# Class to extend the Sklearn classifier

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def feature_importances(self,x,y):

        return(self.clf.fit(x,y).feature_importances_)
def get_oof(clf, x_train, y_train, x_test_201610, x_test_201611, x_test_201612, x_test_201710, x_test_201711, x_test_201712):

    oof_train = np.zeros((ntrain,))

    

    oof_test_201610 = np.zeros((ntest,))

    oof_test_201611 = np.zeros((ntest,))

    oof_test_201612 = np.zeros((ntest,))

    oof_test_201710 = np.zeros((ntest,))    

    oof_test_201711 = np.zeros((ntest,))

    oof_test_201712 = np.zeros((ntest,))

    

    oof_test_skf_201610 = np.empty((NFOLDS, ntest))

    oof_test_skf_201611 = np.empty((NFOLDS, ntest))

    oof_test_skf_201612 = np.empty((NFOLDS, ntest))

    oof_test_skf_201710 = np.empty((NFOLDS, ntest))

    oof_test_skf_201711 = np.empty((NFOLDS, ntest))

    oof_test_skf_201712 = np.empty((NFOLDS, ntest))

    

    #train_index: indicies of training set

    #test_index: indicies of testing set

     

    for i, (train_index, test_index) in enumerate(kf):

        #break the dataset down into two sets, train and test

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        

        clf.train(x_tr, y_tr)

        

        #make a predition on the test data subset

        oof_train[test_index] = clf.predict(x_te)

        

        #use the model trained on the first fold to make a prediction on the entire test data 

        oof_test_skf_201610[i, :] = clf.predict(x_test_201610)

        oof_test_skf_201611[i, :] = clf.predict(x_test_201611)

        oof_test_skf_201612[i, :] = clf.predict(x_test_201612)

        oof_test_skf_201710[i, :] = clf.predict(x_test_201710)

        oof_test_skf_201711[i, :] = clf.predict(x_test_201711)

        oof_test_skf_201712[i, :] = clf.predict(x_test_201712)

    

    #take an average of all of the folds

    oof_test_201610[:] = oof_test_skf_201610.mean(axis=0)

    oof_test_201611[:] = oof_test_skf_201611.mean(axis=0)

    oof_test_201612[:] = oof_test_skf_201612.mean(axis=0)

    oof_test_201710[:] = oof_test_skf_201710.mean(axis=0)

    oof_test_201711[:] = oof_test_skf_201711.mean(axis=0)

    oof_test_201712[:] = oof_test_skf_201712.mean(axis=0)

    

    return oof_train.reshape(-1, 1), oof_test_201610.reshape(-1, 1), oof_test_201611.reshape(-1, 1), oof_test_201612.reshape(-1, 1), oof_test_201710.reshape(-1, 1), oof_test_201711.reshape(-1, 1), oof_test_201712.reshape(-1, 1)
SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction



# Put in our parameters for said classifiers

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}



# Extra Trees Parameters

et_params = {

    'n_jobs': -1,

    'n_estimators':500,

    #'max_features': 0.5,

    'max_depth': 8,

    'min_samples_leaf': 2,

    'verbose': 0

}



# AdaBoost parameters

ada_params = {

    'n_estimators': 400,

    'learning_rate' : 0.75

}



# Support Vector Classifier parameters 

svm_params = {

    'C' : 0.025,

    'epsilon':0.1

    }



# Gradient Boosting parameters

gb_regressor_params = {

    'n_estimators':500, 

    'learning_rate':0.1,

    'max_depth':1, 

    'random_state':0, 

    'loss':'ls'

}
### Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestRegressor, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostRegressor, seed=SEED, params=ada_params)

gb_regressor = SklearnHelper(clf=GradientBoostingRegressor, seed=SEED, params=gb_regressor_params)

svm = SklearnHelper(clf=LinearSVR, seed=SEED, params=svm_params)
ntrain = x_train.shape[0]

print(ntrain)

ntest = x_test_201610.shape[0] #need the size of a test set

print(ntest)



kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)
#svm_oof_train, svm_oof_test_201610, svm_oof_test_201611, svm_oof_test_201612, svm_oof_test_201710, svm_oof_test_201711, svm_oof_test_201712 = get_oof(svm,x_train, y_train, x_test_201610, x_test_201611, x_test_201612, x_test_201710, x_test_201711, x_test_201712) # Support Vector Classifier

print("SVM Training is complete")
#et_oof_train, et_oof_test_201610, et_oof_test_201611, et_oof_test_201612, et_oof_test_201710, et_oof_test_201711, et_oof_test_201712 = get_oof(et, x_train, y_train, x_test_201610, x_test_201611, x_test_201612, x_test_201710, x_test_201711, x_test_201712) # Extra Trees

print("Extra Trees Regressor Training is complete")
#rf_oof_train, rf_oof_test_201610, rf_oof_test_201611, rf_oof_test_201612, rf_oof_test_201710, rf_oof_test_201711, rf_oof_test_201712 = get_oof(rf,x_train, y_train, x_test_201610, x_test_201611, x_test_201612, x_test_201710, x_test_201711, x_test_201712) # Random Forest

print("Random Forest Regressor Training is complete")
#ada_oof_train, ada_oof_test_201610, ada_oof_test_201611, ada_oof_test_201612, ada_oof_test_201710, ada_oof_test_201711, ada_oof_test_201712 = get_oof(ada, x_train, y_train, x_test_201610, x_test_201611, x_test_201612, x_test_201710, x_test_201711, x_test_201712) # AdaBoost 

print("Ada Boost Regressor Training is complete")
#gb_regressor_oof_train, gb_regressor_oof_test_201610, gb_regressor_oof_test_201611, gb_regressor_oof_test_201612, gb_regressor_oof_test_201710, gb_regressor_oof_test_201711, gb_regressor_oof_test_201712 = get_oof(gb_regressor,x_train,y_train,x_test_201610, x_test_201611, x_test_201612, x_test_201710, x_test_201711, x_test_201712)

print("Gradient Boost Regressor Training is complete")



print("Training is complete")
rf_feature = rf.feature_importances(x_train,y_train)

print("rf_feature", rf_feature)

et_feature = et.feature_importances(x_train, y_train)

print("et_feature", et_feature)

ada_feature = ada.feature_importances(x_train, y_train)

print("ada_feature", ada_feature)

gb_regressor_feature = gb_regressor.feature_importances(x_train,y_train)

print("gb_regressor_feature", gb_regressor_feature)
rf_features = [0.03177995, 0.02151717,  0.15271272,  0.00370392,  0.00672652,  0.01860676,

  0.00366968,  0.08548836,  0.06296792,  0.06384852,  0.01510663,  0.05273821,

  0.10866998,  0.09620006,  0.06764444,  0.11359643,  0.00080572,  0.01050976,

  0.00891371,  0.0032457,   0,          0.02464635,  0.0469015]

et_features = [0.06465583,  0.0572915,   0.12032578,  0.00991171,  0.01124228,  0.00960876,

  0.01485536,  0.06740794,  0.05175181,  0.05436677,  0.01772004,  0.05594463,

  0.09801529,  0.0533328,   0.0450343,   0.08253611,  0.00201233,  0.02357432,

  0.03032919,  0.00296583,  0,          0.061361,    0.06575642] 

ada_features = [8.36785346e-03,   3.79894667e-03,   7.05391914e-02,   7.82563418e-05,

   3.22502690e-07,   1.36595920e-02,   0.00000000e+00,   6.15640675e-02,

   5.32715094e-02,   3.73193212e-02,   1.70693107e-02,   1.18344505e-01,

   1.72005440e-01,   3.48224492e-02,   4.48032666e-02,   3.87885107e-02,

   0.00000000e+00,   7.54103834e-03,   0.00000000e+00,   1.06703836e-02,

   0.00000000e+00,   1.25617605e-01,   1.81738430e-01]

gb_regressor_features = [0.02,   0.012,  0.246,  0,     0.016,  0.002,  0.004,  0.114,  0.068,  0.02,

  0.004,  0.012,  0.158,  0.056,  0.06,   0.16,   0,     0.022,  0,     0,     0,

  0.026,  0 ]
cols = train.columns.values

# Create a dataframe with features

feature_dataframe = pd.DataFrame( { 

    'features': cols,

    'Random Forest feature importances': rf_features,

    'Extra Trees  feature importances': et_features,

    'AdaBoost feature importances': ada_features,

    'Gradient Regressor feature importances': gb_regressor_features

    })



feature_dataframe.head(3)
# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['Random Forest feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['Random Forest feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Random Forest feature importances',

    hovermode= 'closest',

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)





# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['Extra Trees  feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['Extra Trees  feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Extra Trees  feature importances',

    hovermode= 'closest',

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)



# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['AdaBoost feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['AdaBoost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'AdaBoost feature importances',

    hovermode= 'closest',

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)







# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['Gradient Regressor feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['Gradient Regressor feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Gradient Regressor Feature Importance',

    hovermode= 'closest',

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise

feature_dataframe.head(50)
y = feature_dataframe['mean'].values

x = feature_dataframe['features'].values

data = [go.Bar(

            x= x,

             y= y,

            width = 0.5,

            marker=dict(

               color = feature_dataframe['mean'].values,

            colorscale='Portland',

            showscale=True,

            reversescale = False

            ),

            opacity=0.6

        )]



layout= go.Layout(

    autosize= True,

    title= 'Barplots of Mean Feature Importance',

    hovermode= 'closest',

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='bar-direct-labels')
#predictions from first layer become data input for second layer

base_predictions_train = pd.DataFrame( 

    {

    'RandomForest': rf_oof_train.ravel(),

    'ExtraTrees': et_oof_train.ravel(),

    'AdaBoost': ada_oof_train.ravel(),

    'GradientRegressor': gb_regressor_oof_train.ravel()

    })

#predictions for all instances in the training set

base_predictions_train.head(3)
data = [

    go.Heatmap(

        z= base_predictions_train.astype(float).corr().values ,

        x=base_predictions_train.columns.values,

        y= base_predictions_train.columns.values,

          colorscale='Portland',

            showscale=True,

            reversescale = True

    )

]

py.iplot(data, filename='labelled-heatmap')
x_train = np.concatenate(( et_oof_train, rf_oof_train, gb_regressor_oof_train, svm_oof_train), axis=1)



x_test_201610 = np.concatenate(( et_oof_test_201610, rf_oof_test_201610, gb_regressor_oof_test_201610, svm_oof_test_201610), axis=1)

x_test_201611 = np.concatenate(( et_oof_test_201611, rf_oof_test_201611, gb_regressor_oof_test_201611, svm_oof_test_201611), axis=1)

x_test_201612 = np.concatenate(( et_oof_test_201612, rf_oof_test_201612, gb_regressor_oof_test_201612, svm_oof_test_201612), axis=1)

x_test_201710 = np.concatenate(( et_oof_test_201710, rf_oof_test_201710, gb_regressor_oof_test_201710, svm_oof_test_201710), axis=1)

x_test_201711 = np.concatenate(( et_oof_test_201711, rf_oof_test_201711, gb_regressor_oof_test_201711, svm_oof_test_201711), axis=1)

x_test_201712 = np.concatenate(( et_oof_test_201712, rf_oof_test_201712, gb_regressor_oof_test_201712, svm_oof_test_201712), axis=1)
#gbm = xgb.XGBRegressor(

    #learning_rate = 0.02,

# n_estimators= 2000,

# max_depth= 4,

# min_child_weight= 2,

# #gamma=1,

# gamma=0.9,                        

# subsample=0.8,

# colsample_bytree=0.8,

# objective= 'reg:linear',

# nthread= -1,

# scale_pos_weight=1

# ).fit(x_train, y_train)
#generate the predictions

#predictions_201610 = gbm.predict(x_test_201610).round(4)

#predictions_201611 = gbm.predict(x_test_201611).round(4)

#predictions_201612 = gbm.predict(x_test_201612).round(4)



#predictions_201710 = gbm.predict(x_test_201710).round(4)

#predictions_201711 = gbm.predict(x_test_201711).round(4)

#predictions_201712 = gbm.predict(x_test_201712).round(4)
 #StackingSubmission = pd.DataFrame({ '201610': predictions_201610, 

 #                                         '201611': predictions_201611,

 #                                         '201612': predictions_201612,

 #                                         '201710': predictions_201710,

 #                                         '201711': predictions_201711,

 #                                         '201712': predictions_201712,

 #                                         'ParcelId': ParcelID,

 #                            })

#print(StackingSubmission)