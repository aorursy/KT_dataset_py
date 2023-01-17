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

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC

from sklearn.model_selection import KFold

train=pd.read_csv('../input/learn-together/train.csv')

test=pd.read_csv('../input/learn-together/test.csv')

sample_submission=pd.read_csv('../input/learn-together/sample_submission.csv')
colormap = plt.cm.RdBu

plt.figure(figsize=(30,30))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
# Some useful parameters which will come in handy later on

ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(n_splits= NFOLDS, random_state=SEED)



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

    

    def featureimportances(self,x,y):

        return(self.clf.fit(x,y).featureimportances)
def get_oof(clf, x_train, y_train, x_test,train):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf.split(train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
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

    'n_estimators': 500,

    'learning_rate' : 0.75

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}
# Create 5 objects that represent our 4 models

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
# # Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models

y_train = train['Cover_Type'].ravel()

train = train.drop(['Cover_Type'], axis=1)

x_train = train.values # Creates an array of the train data

x_test = test.values # Creats an array of the test data
# # Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test,train) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test,train) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test,train) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test,train) # Gradient Boost



print("Training is complete")
rffeatures = rf.feature_importances(x_train,y_train)

et_feature = et.feature_importances(x_train, y_train)

ada_feature = ada.feature_importances(x_train, y_train)

gb_feature = gb.feature_importances(x_train,y_train)
rf_features=[0.0653386317, 0.268108074, 1.42839657e-02, 7.19156381e-03,

 4.13514491e-02, 2.41487372e-02, 6.32447442e-02, 2.51975425e-02,

 7.10712493e-03 ,1.29448946e-02, 2.84739582e-02, 1.51107746e-02,

 4.83119888e-03 ,3.39534421e-02, 1.13941100e-01, 1.95066524e-04,

 4.00883743e-03 ,4.32800819e-02, 1.76773018e-02, 3.25226534e-04,

 1.49393911e-03 ,0.00000000e+00, 0.00000000e+00, 5.88897147e-06,

 5.66192773e-02 ,1.26682285e-03, 7.09571857e-04, 1.62340618e-02,

 3.83372651e-04 ,0.00000000e+00 ,1.28555088e-04, 5.86794893e-03,

 8.11840877e-05 ,4.31002213e-06, 1.28590880e-04, 2.02550994e-05,

 6.26636751e-03 ,5.47858778e-03 ,8.78188005e-04, 0.00000000e+00,

 2.31388696e-05 ,1.33793100e-05 ,0.00000000e+00, 3.06472875e-03,

 1.67556242e-03 ,5.34528513e-04 ,3.71206392e-03, 6.74646390e-04,

 1.18542434e-05 ,1.18870469e-03 ,3.03438386e-06, 2.56667624e-04,

 4.42542736e-02 ,3.66349294e-02 ,2.16718502e-02]



et_features=[2.00323300e-02, 1.57974391e-01, 1.04367874e-02, 8.01935244e-03,

 1.54490349e-02, 8.94297793e-03, 3.69506432e-02, 1.65402604e-02,

 7.05963184e-03, 1.16087339e-02, 1.63213070e-02, 5.41784033e-02,

 7.22718969e-03, 4.36806516e-02, 1.70251320e-01, 1.24754074e-03,

 8.38209284e-03, 5.98490030e-02, 2.49438533e-02, 1.19938161e-03,

 4.40891025e-03, 0.00000000e+00, 0.00000000e+00, 6.62546072e-05,

 7.16434246e-02, 1.54439192e-03, 1.00454046e-02, 9.62085947e-03,

 2.13425345e-03, 0.00000000e+00, 3.22089792e-04 ,1.49918988e-02,

 1.14671967e-03, 7.53632614e-05 ,4.61637387e-04, 6.34530298e-06,

 1.05107416e-02, 6.16968222e-03 ,1.43969050e-03, 0.00000000e+00,

 7.92748272e-05, 1.09216273e-05 ,1.11310231e-05, 1.20959440e-02,

 2.47280244e-02, 1.03838327e-03 ,4.89115919e-03, 1.75953346e-03,

 1.82531122e-05, 2.33314207e-03, 9.98047251e-06, 7.45852092e-04,

 5.56504812e-02, 5.04751490e-02, 3.12702164e-02]



ada_features=[0. ,   0.334, 0. ,    0. ,    0.328, 0.004 ,0. ,   0. ,   0. ,   0. ,   0. ,   0.002,

 0.  ,  0.  ,  0.332, 0.   , 0. ,   0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0. ,   0. ,

 0.  ,  0.  ,  0.  ,  0.  ,  0. ,   0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0. ,

 0.  ,  0.  ,  0.  ,  0.  ,  0. ,   0.  ,  0.  ,  0.  ,  0.  ,  0. ,   0.  ,  0. ,

 0.  ,  0.  ,  0.  ,  0.  ,  0. ,   0.  ,  0.   ]



gb_features=[9.23807651e-02, 4.53059872e-01 ,1.82701273e-02, 7.15936708e-03,

 5.05055314e-02, 2.21329794e-02, 5.68438923e-02, 4.63737905e-02,

 1.72796716e-02 ,1.45278433e-02, 5.35590597e-02, 3.81380933e-03,

 1.24769898e-03 ,8.56185829e-03, 2.13915431e-03, 8.41919839e-04,

 5.54011455e-03 ,1.14838423e-02, 1.22423982e-02, 1.98441022e-03,

 2.45840472e-03 ,0.00000000e+00, 0.00000000e+00, 1.79202952e-05,

 3.52959393e-02 ,2.16911650e-03, 4.42834495e-03, 1.41830077e-02,

 1.17709301e-04 ,0.00000000e+00, 4.82907945e-04, 3.92275992e-03,

 4.94074158e-05 ,2.09058102e-04, 1.21503285e-03, 1.30829819e-04,

 3.78783475e-03 ,3.28079497e-03, 1.26110953e-03, 0.00000000e+00,

 3.19075903e-04 ,2.61379053e-04, 2.08531237e-04, 1.95439744e-03,

 2.03778112e-02 ,1.75494613e-03, 7.67160302e-03, 3.38316878e-03,

 3.73420741e-04 ,7.60061654e-04, 2.02655172e-04, 6.42735625e-07,

 2.46647368e-03, 6.51226716e-03, 7.95281876e-04]
cols = train.columns.values

# Create a dataframe with features

feature_dataframe = pd.DataFrame( {'features': cols,

     'Random Forest feature importances': rf_features,

     'Extra Trees  feature importances': et_features,

      'AdaBoost feature importances': ada_features,

    'Gradient Boost feature importances': gb_features

    })
feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise

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

py.iplot(fig,filename='basic-scatter')



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

py.iplot(fig,filename='basic-scatter')



# Scatter plot 

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

py.iplot(fig,filename='basic-scatter')



# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['Gradient Boost feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['Gradient Boost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Gradient Boosting Feature Importance',

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

py.iplot(data, filename='basic-scatter')
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

py.iplot(data, filename='grouped-bar-direct-labels')
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel()

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

py.iplot(data, filename='pandas-heatmap')
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test), axis=1)
gbm = xgb.XGBClassifier(

    #learning_rate = 0.02,

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)
output = pd.DataFrame({'Id': test.Id,

                      'Cover_Type': predictions})

output.to_csv('sample_submission.csv', index=False)