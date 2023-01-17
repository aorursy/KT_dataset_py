# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





from math import radians, sin, cos, acos

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedShuffleSplit



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Path of the file to read

rail_file_path = '../input/renfe.csv'

#Read file from the csv

rail_df = pd.read_csv(rail_file_path)

rail_df.shape
#Basic Data exploration

rail_df.head()
# Getting the information about data types 

rail_df.info()
# find unique values of the column

rail_df.fare.unique() 
rail_df.isna().sum()
# Null value handling 

rail_df['price'].fillna(rail_df['price'].mean(),inplace=True)

rail_df.dropna(inplace=True)
rail_df.isna().sum()
rail_df.head()
dictt_latitude = {

    'MADRID' : 40.416775, 

    'SEVILLA' : 37.382641, 

    'PONFERRADA' : 42.546329, 

    'BARCELONA' : 41.385063, 

    'VALENCIA'  : 28.521076

}
dictt_longitude = {

    'MADRID' : -3.703790, 

    'SEVILLA' : -5.996300, 

    'PONFERRADA': -6.590830, 

    'BARCELONA' : 2.173404, 

    'VALENCIA' : -81.465523 

}
rail_df['start_latitude']= rail_df['origin'].map(dictt_latitude)

rail_df['start_longitude'] = rail_df['origin'].map(dictt_longitude)

rail_df['end_latitude'] = rail_df['destination'].map(dictt_latitude)

rail_df['end_longitude'] = rail_df['destination'].map(dictt_longitude)
                                         #dist = 6371.01 * acos(sin(slat)*sin(elat) + cos(slat)*cos(elat)*cos(slon - elon))

rail_df['distance'] =  rail_df.apply(lambda row: 

                                     6371.01 * acos(sin(radians(row.start_latitude))*sin(radians(row.end_latitude)) + cos(radians(row.start_latitude))* 

                                                                                        cos(radians(row.end_latitude))*cos(radians(row.start_longitude)- 

                                                                                                                          radians(row.end_longitude))),axis=1)
# converting Object in date time format

rail_df['book_date'] = pd.to_datetime(rail_df['insert_date'])

rail_df['travel_start_date'] = pd.to_datetime(rail_df['start_date'])

rail_df['travel_end_date'] = pd.to_datetime(rail_df['end_date'])

rail_df['book_leadtime'] = (rail_df.travel_start_date - rail_df.book_date).dt.days

rail_df['journey_duration'] = ((rail_df.travel_end_date - rail_df.travel_start_date))/np.timedelta64(1,'h')

rail_df['origdest'] = rail_df.origin +"_"+rail_df.destination

#rail_df['train_classification'] = rail_df.train_type +"_"+rail_df.train_class + "_" + rail_df.fare
drop_features = ["Unnamed: 0","insert_date" , "start_date" , "end_date"]

rail_df_modified = rail_df.drop(drop_features, axis = 1)
encoding = rail_df.groupby("origdest").size()

encoding = encoding / len(rail_df)

rail_df["origdest_enc"] = rail_df.origdest.map(encoding)





encoding_train_type = rail_df.groupby("train_type").size()

encoding_train_type = encoding_train_type / len(rail_df)

rail_df["encoding_train_type"] = rail_df.train_type.map(encoding_train_type)



encoding_train_class = rail_df.groupby("train_class").size()

encoding_train_class = encoding_train_class / len(rail_df)

rail_df["encoding_train_class"] = rail_df.train_class.map(encoding_train_class)



encoding_fare = rail_df.groupby("fare").size()

encoding_fare = encoding_fare / len(rail_df)

rail_df["encoding_fare"] = rail_df.fare.map(encoding_fare)
rail_df["journey_start_weekday"]=rail_df.travel_start_date.dt.weekday

rail_df["journey_start_month"]=rail_df.travel_start_date.dt.month

rail_df["journey_start_hour"]=rail_df.travel_start_date.dt.hour
drop_features = ["Unnamed: 0","insert_date" , "origin" , "destination","start_date" , "end_longitude" ,"end_latitude" , "start_longitude" , "start_latitude",

                 "end_date", "train_type" , "train_class" ,"fare" , "book_date" ,

                 "travel_start_date" , "travel_end_date" ,"origdest" , "train_class","fare" ,"train_class" , "train_type"]

rail_df_modified = rail_df.drop(drop_features, axis = 1)
rail_df_modified.info()

rail_df_modified.dropna()

rail_df_modified.head()
rail_df_modified.hist(bins=50)
#sns.regplot(x=rail_df_modified['book_leadtime'], y=rail_df_modified['price'])
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(rail_df_modified.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
pd.plotting.scatter_matrix (rail_df_modified.loc[1:1000,:],figsize = (12,14))
rail_df_modified.head()
rail_df_modified.columns
from sklearn.decomposition import PCA



array = rail_df_modified.values

X = array[:,1:12]

Y = array[:,0]



# feature extraction

pca = PCA(n_components=10)

fit = pca.fit(X)

# summarize components

print("Explained Variance: %s",fit.explained_variance_)

print("Explained Variance Ration: %s",fit.explained_variance_ratio_)

print("Explained Variance cumulative: %s",fit.explained_variance_ratio_.cumsum())

print("Singular values",fit.singular_values_) 

print(fit.components_)
from sklearn.ensemble import ExtraTreesClassifier



array = rail_df_modified.values

X = array[:,1:12]

Y = array[:,0]

Y=Y.astype('int')





# feature extraction

forest = ExtraTreesClassifier()

forest.fit(X, Y)

importances = forest.feature_importances_

print(importances)
std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), indices)

plt.xlim([-1, X.shape[1]])

plt.show()
rail_df_modified.info()
drop_features = ["origdest_enc","encoding_train_type" , "encoding_train_type" , "encoding_train_class","encoding_fare","journey_start_month","journey_start_month"]

rail_df_final = rail_df_modified.drop(drop_features, axis = 1)
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(rail_df_final.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
rail_df_final.head()

rail_df_final_subset = rail_df_final.sample(frac=0.01)
from sklearn.model_selection import train_test_split

train, test = train_test_split(rail_df_final_subset, test_size=0.2)

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models

y_train = train['price'].ravel()

y_test= test['price'].ravel()

train = train.drop(['price'], axis=1)

test = test.drop(['price'], axis=1)

x_train = train.values # Creates an array of the train data

x_test = test.values # Creats an array of the test data

train.shape
# Some useful parameters which will come in handy later on

ntrain = train.shape[0]

ntest = test.shape[0]

print(train.shape[0])

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

#calculate accuracy of model

log_cols = ["Classifier", "Accuracy"]

log 	 = pd.DataFrame(columns=log_cols)

acc_dict = {}

kf = KFold( n_splits= NFOLDS, random_state=SEED)

sss = StratifiedShuffleSplit(n_splits=NFOLDS, test_size=0.1, random_state=SEED)



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

        print(self.clf.fit(x,y).feature_importances_)

    

# Class to extend XGboost classifer
def get_oof(clf, x_train, y_train, x_test):  

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i,(train_index, test_index) in enumerate(kf.split(x_train)):

        print(i)

        print(train_index)

        print(test_index)

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        clf.train(x_tr, y_tr.astype(int))

        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
# Put in our parameters for said classifiers

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': False, 

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

    'n_estimators': 500,

    'learning_rate' : 0.75

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

     'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }
# Going to use these 5 base models for the stacking

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)



from sklearn.svm import SVC
# Create 5 objects that represent our 4 models

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 

#gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost

#svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier



print("Training is complete")
rf_feature = rf.feature_importances(x_train,y_train.astype(int))

et_feature = et.feature_importances(x_train, y_train.astype(int))

ada_feature = ada.feature_importances(x_train, y_train.astype(int))

#gb_feature = gb.feature_importances(x_train,y_train)
rf_features = [0.31691064,0.04371718,0.48272728,0.02617127,0.13047363]

et_features = [0.34598855,0.04926204,0.41258659,0.04506057,0.14710226]

ada_features = [0.348,0.064,0.408,0.0,    0.18  ]
cols = train.columns.values

# Create a dataframe with features

feature_dataframe = pd.DataFrame( {'features': cols,

     'Random Forest feature importances': rf_features,

     'Extra Trees  feature importances': et_features,

     'AdaBoost feature importances': ada_features

    #'Gradient Boost feature importances': gb_features

    })
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['Random Forest feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['Random Forest feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Random Forest Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['Extra Trees  feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['Extra Trees  feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Extra Trees Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



#Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['AdaBoost feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['AdaBoost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'AdaBoost Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



# # Scatter plot 

# trace = go.Scatter(

#     y = feature_dataframe['Gradient Boost feature importances'].values,

#     x = feature_dataframe['features'].values,

#     mode='markers',

#     marker=dict(

#         sizemode = 'diameter',

#         sizeref = 1,

#         size = 25,

# #       size= feature_dataframe['AdaBoost feature importances'].values,

#         #color = np.random.randn(500), #set color equal to a variable

#         color = feature_dataframe['Gradient Boost feature importances'].values,

#         colorscale='Portland',

#         showscale=True

#     ),

#     text = feature_dataframe['features'].values

# )

# data = [trace]



# layout= go.Layout(

#     autosize= True,

#     title= 'Gradient Boosting Feature Importance',

#     hovermode= 'closest',

# #     xaxis= dict(

# #         title= 'Pop',

# #         ticklen= 5,

# #         zeroline= False,

# #         gridwidth= 2,

# #     ),

#     yaxis=dict(

#         title= 'Feature Importance',

#         ticklen= 5,

#         gridwidth= 2

#     ),

#     showlegend= False

# )

# fig = go.Figure(data=data, layout=layout)

# py.iplot(fig,filename='scatter2010')
# Create the new column containing the average of values



feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise

feature_dataframe.head(3)
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

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='bar-direct-labels')
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel()

    })

base_predictions_train.head()
data = [

    go.Heatmap(

        z= base_predictions_train.astype(float).corr().values ,

        x=base_predictions_train.columns.values,

        y= base_predictions_train.columns.values,

          colorscale='Viridis',

            showscale=True,

            reversescale = True

    )

]

py.iplot(data, filename='labelled-heatmap')
X_train = np.concatenate(( et_oof_train, ada_oof_train,rf_oof_train), axis=1)

X_test = np.concatenate(( et_oof_test, ada_oof_test,rf_oof_test), axis=1)
X_test.shape
import xgboost as xgb



gbm = xgb.XGBRegressor(

 learning_rate = 0.02,

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 #objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(X_train, y_train) # X_train is the 

x_test.shape
from sklearn.metrics import r2_score, mean_squared_error ,explained_variance_score ,mean_squared_log_error



r2 = r2_score(y_test, gbm.predict(X_test))

m2e = mean_squared_error (y_test, gbm.predict(X_test))

var_Score = explained_variance_score (y_test, gbm.predict(X_test))

mslerror = mean_squared_log_error(y_test, gbm.predict(X_test))

print ("r2 score" , r2)

print ("mean_squared_error" , m2e)

print ("var_Score", var_Score)

print ("mslerror", mslerror)
import xgboost as xgb

from sklearn.model_selection import GridSearchCV



from sklearn.model_selection import train_test_split



y = rail_df_final_subset['price']

X = rail_df_final_subset.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)



print("Parameter optimization")

xgb_model = xgb.XGBRegressor()

reg_xgb = GridSearchCV(xgb_model,

                   {'max_depth': [2,4,6],

                    'n_estimators': [50,100,200]}, verbose=1)

reg_xgb.fit(X_train, y_train)
reg_xgb.score(X_train, y_train)
print('Best score: ', reg_xgb.best_score_)

best = reg_xgb.best_estimator_

print('R2: ', r2_score(y_pred = best.predict(X), y_true = y))