# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# import the csv into a dataframe

dataframe = pd.read_csv('../input/mushrooms.csv')

dataframe.head(5)
# Apply one hot encoding on all the predictor variables (not dependent variable)

dataframe[dataframe.columns[1:23]].head(5)

Labledmushrooms = pd.get_dummies(dataframe[dataframe.columns[1:23]])

Labledmushrooms.head(5)
# Convert the classification labels into binary digits

from sklearn.preprocessing import LabelEncoder



Labledmushrooms['class'] = LabelEncoder().fit_transform(dataframe['class'])

Labledmushrooms.head(5)
# split the dataframe into test and train (60% - train 40% - test)



from sklearn.model_selection import train_test_split



train, test = train_test_split(Labledmushrooms, test_size=0.4)
# check which features are highly correlated with 'class'



df_num_corr = train.corr()['class'][:-1] # -1 because the last row is class

golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)

print("There are {} strongly correlated values with class:\n{}".format(len(golden_features_list), golden_features_list))
from sklearn.cross_validation import KFold;

# Some useful parameters which will come in handy later on

ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)



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



    for i, (train_index, test_index) in enumerate(kf):

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



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }
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
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
# Create Numpy arrays of train, test and target dataframes to feed into our models

y_train = train['class'].ravel()

train = train.drop(['class'], axis=1)

x_train = train.values # Creates an array of the train data

x_test = test.drop(['class'], axis=1).values # Creats an array of the test data
# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier



print("Training is complete")
rf_feature = rf.feature_importances(x_train,y_train)

et_feature = et.feature_importances(x_train, y_train)

ada_feature = ada.feature_importances(x_train, y_train)

gb_feature = gb.feature_importances(x_train,y_train)
# simply copied these values from the previous result into arrays



rf_features = [ 1.09933352e-03, 0.00000000e+00, 3.31604164e-04, 1.25604826e-04

, 3.84436946e-04, 5.85765610e-04, 4.96563934e-03, 6.15687151e-05

, 3.83359703e-03, 1.03188819e-03, 1.31974204e-03, 2.55374992e-04

, 6.94622587e-04, 7.18087103e-04, 1.19924501e-03, 1.03416153e-03

, 1.42208440e-04, 1.61936024e-04, 2.26738317e-03, 2.63980344e-03

, 2.62849341e-02, 2.54988109e-02, 4.26665512e-03, 8.38218479e-03

, 7.81849276e-02, 5.05969243e-03, 6.72418638e-04, 1.26698724e-01

, 1.13648255e-02, 3.96953051e-03, 6.64036098e-03, 1.01236515e-03

, 5.77700436e-04, 1.91641180e-02, 1.83704757e-02, 6.98676012e-02

, 6.40741143e-02, 4.79758134e-02, 3.14467864e-04, 5.64966280e-04

, 9.91736391e-04, 1.53130207e-04, 1.02566947e-03, 1.68234149e-04

, 1.66311393e-04, 5.46101822e-04, 7.93828378e-04, 1.52942168e-03

, 7.89890008e-05, 8.51535373e-03, 9.54730355e-03, 9.75651680e-03

, 1.43539101e-02, 6.12520557e-03, 1.09756231e-02, 1.51545284e-03

, 3.85428506e-03, 5.39394393e-02, 2.34922494e-02, 1.66239503e-04

, 4.86265250e-03, 3.92016229e-02, 1.34092441e-02, 2.52800344e-03

, 1.37016511e-03, 6.08798442e-04, 2.09458230e-04, 1.37640237e-03

, 1.03338895e-03, 8.84882581e-04, 1.02425977e-03, 6.32664915e-03

, 1.78083543e-04, 1.52618803e-03, 5.44017592e-04, 1.81727915e-04

, 1.42783390e-03, 1.00475704e-03, 1.19912910e-03, 1.19165987e-03

, 5.01796315e-03, 6.13223401e-04, 0.00000000e+00, 3.81358979e-04

, 7.59864000e-05, 6.35769040e-04, 1.09889791e-04, 6.97684555e-04

, 4.73405978e-03, 6.68991835e-03, 7.58643949e-03, 6.03251183e-04

, 2.04044709e-02, 6.00950973e-04, 3.75201096e-02, 6.78084129e-05

, 4.85935645e-02, 1.08687474e-02, 9.43236950e-03, 5.76559360e-05

, 5.74750743e-03, 8.66434826e-04, 1.25750033e-02, 7.46856542e-05

, 7.65403378e-04, 1.00052881e-03, 2.94397492e-03, 5.14232177e-03

, 1.77456026e-02, 3.52087039e-03, 4.78182954e-03, 5.84478834e-03

, 8.26626082e-04, 2.10138323e-03, 3.97095313e-03, 6.72934184e-03

, 1.09090762e-03]

et_features = [ 1.88703460e-03, 0.00000000e+00, 4.31585124e-04, 4.73182972e-04

, 3.92765398e-04, 6.13083998e-04, 4.28778412e-03, 9.88414829e-05

, 4.26284687e-03, 1.83935199e-03, 1.19673300e-03, 3.73001924e-04

, 6.33520218e-04, 1.33975768e-03, 9.37675125e-04, 1.16082203e-03

, 1.03928661e-04, 1.68635461e-04, 2.99017100e-03, 3.87697817e-03

, 2.95108606e-02, 3.04136756e-02, 5.32162431e-03, 9.25042383e-03

, 8.03515201e-02, 5.48359938e-03, 9.10315345e-04, 1.34070374e-01

, 1.40089574e-02, 3.91140227e-03, 3.76257570e-03, 8.37739272e-04

, 5.41948148e-04, 1.52514343e-02, 1.95611466e-02, 6.10942175e-02

, 5.65501187e-02, 3.88718039e-02, 3.33591450e-04, 4.18850413e-04

, 8.90619363e-04, 1.58992629e-04, 1.89350978e-03, 4.51715241e-05

, 1.54153629e-04, 8.42735773e-04, 4.69539738e-04, 1.75509160e-03

, 1.48491081e-04, 1.21229590e-02, 1.18750841e-02, 9.81544616e-03

, 1.45702111e-02, 6.10788362e-03, 1.24407466e-02, 1.44588195e-03

, 3.16254057e-03, 4.28487696e-02, 2.05583705e-02, 3.24089302e-04

, 4.79289529e-03, 3.66126661e-02, 1.06137255e-02, 3.13491734e-03

, 9.51405752e-04, 9.53167951e-04, 3.12308238e-05, 1.09342660e-03

, 1.22142293e-03, 1.83581057e-03, 1.18761003e-03, 5.13867576e-03

, 2.10528116e-04, 2.64282997e-03, 1.11620301e-03, 5.11776457e-06

, 1.12419974e-03, 2.49029997e-03, 9.01295155e-04, 1.61212185e-03

, 7.32676075e-03, 9.75825258e-04, 0.00000000e+00, 2.49237247e-04

, 1.76010336e-04, 1.25850676e-03, 2.87055660e-04, 6.14717492e-04

, 5.05107522e-03, 5.53740164e-03, 7.13153963e-03, 9.81913488e-04

, 1.62319669e-02, 7.41505912e-04, 3.73063568e-02, 0.00000000e+00

, 4.56401409e-02, 1.07350774e-02, 8.96201707e-03, 3.16362129e-05

, 5.91316149e-03, 1.27347115e-03, 1.56709352e-02, 3.40078372e-06

, 1.19733297e-03, 1.08854281e-03, 2.64565608e-03, 4.28184505e-03

, 2.17512228e-02, 3.38783548e-03, 8.47367727e-03, 8.30029339e-03

, 1.06849795e-03, 2.74104547e-03, 4.00403308e-03, 1.07437645e-02

, 1.39282369e-03]

ada_features = [0.002,  0.0,     0.0,     0.0,     0.0,     0.0,     0.002,  0.0,     0.002,  0.002,  0.0,

  0.002,  0.0,     0.0,     0.006,  0.0,     0.0,     0.0,     0.016,  0.0,     0.064,

  0.056,  0.01,   0.024,  0.026,  0.01,   0.0,     0.024,  0.016,  0.0,     0.0,     0.0,

  0.0,     0.024,  0.008,  0.042,  0.042,  0.022,  0.0,     0.0,     0.0,     0.0,     0.0,

  0.0,     0.0,     0.0,     0.0,     0.002,  0.0,     0.006,  0.004,  0.002,  0.006,  0.0,

  0.0,     0.002,  0.006,  0.096,  0.018,  0.0,     0.004,  0.0,     0.014,  0.104,  0.0,

  0.0,     0.0,     0.0,     0.0,     0.0,     0.0,     0.004,  0.0,     0.0,     0.0,     0.0,

  0.0,     0.016,  0.002,  0.0,     0.006,  0.0,     0.0,     0.0,     0.0,     0.0,     0.0,

  0.0,     0.002,  0.028,  0.002,  0.002,  0.0,     0.0,     0.018,  0.0,     0.0,     0.004,

  0.006,  0.0,     0.034,  0.002,  0.046,  0.0,     0.0,     0.134,  0.0,     0.0,     0.006,

  0.002,  0.014,  0.0,     0.0,     0.0,     0.0,     0.0,     0.008,]

gb_features = [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00

, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.28091635e-04

, 0.00000000e+00, 3.33626202e-04, 0.00000000e+00, 0.00000000e+00

, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00

, 0.00000000e+00, 0.00000000e+00, 1.48842261e-03, 0.00000000e+00

, 1.49765891e-03, 3.14086275e-03, 2.39605782e-03, 0.00000000e+00

, 1.41862357e-03, 2.62634061e-03, 7.00943804e-05, 9.51542919e-02

, 9.50364670e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00

, 0.00000000e+00, 1.50038698e-03, 4.35033617e-04, 1.53458912e-03

, 1.92855525e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00

, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00

, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00

, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.54267174e-03

, 6.69786227e-04, 2.09815378e-02, 4.75223795e-04, 4.92955745e-03

, 0.00000000e+00, 7.28823904e-05, 1.07702935e-04, 2.24355593e-04

, 0.00000000e+00, 0.00000000e+00, 7.32165126e-05, 7.01669444e-03

, 0.00000000e+00, 7.00675774e-05, 0.00000000e+00, 0.00000000e+00

, 5.11722172e-05, 0.00000000e+00, 0.00000000e+00, 1.39902130e-04

, 0.00000000e+00, 0.00000000e+00, 1.07821302e-04, 0.00000000e+00

, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00

, 2.53114380e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00

, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.87603256e-04

, 2.27648462e-04, 7.21009716e-05, 1.58350366e-04, 0.00000000e+00

, 0.00000000e+00, 1.08798568e-04, 1.19355102e-04, 0.00000000e+00

, 7.11436910e-04, 0.00000000e+00, 1.25399658e-05, 0.00000000e+00

, 4.19736455e-03, 1.01651638e-04, 2.48571766e-04, 0.00000000e+00

, 0.00000000e+00, 1.29542836e-03, 0.00000000e+00, 0.00000000e+00

, 0.00000000e+00, 5.16168159e-05, 6.99296103e-05, 0.00000000e+00

, 2.58644603e-06, 0.00000000e+00, 0.00000000e+00, 5.16310735e-04

, 0.00000000e+00]
cols = train.columns.values

# Create a dataframe with features

feature_dataframe = pd.DataFrame( {'features': cols,

     'Random Forest feature importances': rf_features,

     'Extra Trees  feature importances': et_features,

      'AdaBoost feature importances': ada_features,

    'Gradient Boost feature importances': gb_features

    })
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

py.iplot(fig,filename='scatter2010')



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

py.iplot(fig,filename='scatter2010')
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

py.iplot(data, filename='labelled-heatmap')
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
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
test['predicted'] = predictions

test.head(50)
from sklearn.metrics import precision_recall_fscore_support

precision_recall_fscore_support(test['class'], test['predicted'],average='macro')
n_folds = 5

early_stopping = 10

params = {'eta': 0.02, 'max_depth': 5, 'subsample': 0.7, 'colsample_bytree': 0.7, 'objective': 'binary:logistic', 'seed': 99, 'silent': 1, 'eval_metric':'auc', 'nthread':4}



xg_train = xgb.DMatrix(x_train, label=y_train);



cv = xgb.cv(params, xg_train, 5000, nfold=n_folds, early_stopping_rounds=early_stopping, verbose_eval=1)