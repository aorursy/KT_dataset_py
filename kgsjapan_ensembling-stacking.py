import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import xgboost as xgb

# Going to use these 5 base models for the stacking
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, 
                              GradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.model_selection import KFold

# for plot
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py # comment test
py.init_notebook_mode(connected=True)

#import warnings
#warnings.filterwarnings('ignore')

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Load in the train and test datasets
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

# Store our passenger ID for easy access
Id = test['Id']

train.head(10)
#ラベルを整数に変換
def replacePdSimple(pdTgtTrain,pdTgtTest):
    tempTgt = pdTgtTest.select_dtypes(include='number')
    tempMedian = pdTgtTrain[tempTgt.columns].median()

    pdTgtTrain[tempTgt.columns] = pdTgtTrain[tempTgt.columns].fillna(tempMedian)
    pdTgtTest[tempTgt.columns] = pdTgtTest[tempTgt.columns].fillna(tempMedian)
    
    pdTgtTrain = pdTgtTrain.fillna("LostValue")
    pdTgtTest = pdTgtTest.fillna("LostValue")

    le = LabelEncoder()
    for column in pdTgtTest.select_dtypes(exclude='number').columns:
        le = le.fit(pdTgtTrain[column].append(pdTgtTest[column]))
        pdTgtTrain[column] = le.transform(pdTgtTrain[column])
        pdTgtTest[column] = le.transform(pdTgtTest[column])
    return pdTgtTrain,pdTgtTest

train,test = replacePdSimple(train,test)
train.head(10)
test.head(10)

# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(n_splits=NFOLDS, shuffle=False)
#kf = KFold(n_splits=NFOLDS, shuffle=False, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        if clf != SVR:
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
#colormap = plt.cm.RdBu
#plt.figure(figsize=(255,255))
#sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
#            square=True, cmap=colormap, linecolor='white', annot=True)
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train,y_train)):
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
#     'warm_start': True, 
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
svr_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }
# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train['SalePrice'].values.ravel()
x_train = (train.drop(['SalePrice'], axis=1)).values # Creates an array of the train data
x_test = test.values # Creats an array of the test data

# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostRegressor, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingRegressor, seed=SEED, params=gb_params)
svr = SklearnHelper(clf=SVR, seed=SEED, params=svr_params)

# Create our OOF train and test predictions. These base results will be used as new features
print("calc rf")
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
print("calc et")
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
print("calc ada")
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
print("calc gb")
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
print("calc svr")
svr_oof_train, svr_oof_test = get_oof(svr,x_train, y_train, x_test) # Support Vector 

rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)

print("Training is complete")
rf_features = [2.85471447e-03,3.93199134e-03,1.79799243e-03,9.70061798e-03,1.89939669e-02,1.04711520e-05,1.91018032e-04,1.76339141e-03,1.39993704e-03,0.00000000e+00,9.00244090e-04,9.48561835e-04,8.19981234e-03,2.62295258e-04,3.23231610e-05,9.70013601e-04,2.02408927e-03,1.24795389e-01,2.89256906e-03,4.60565267e-02,2.02119577e-02,2.71817368e-03,3.55177294e-04,2.04368790e-03,2.08878685e-03,2.28516096e-03,1.48219409e-02,5.28189515e-02,3.81444108e-04,1.16369243e-02,4.16709915e-02,2.38110345e-04,2.79967911e-03,1.71627981e-03,3.51330322e-02,2.96092209e-04,2.85292807e-04,5.18991269e-03,6.45911180e-02,1.26650359e-04,6.98757442e-03,1.39710915e-03,3.85411036e-04,6.00962481e-02,2.62145333e-02,1.01802943e-05,9.50968541e-02,1.70524109e-03,6.50873838e-05,3.53638715e-02,3.52531097e-03,5.84600129e-03,8.81736509e-04,3.28154476e-02,2.04111940e-02,2.67433564e-04,1.94385226e-02,4.63250819e-03,1.43021105e-02,3.13453225e-02,7.73770276e-03,6.28344549e-02,5.03900502e-02,1.34846579e-03,1.47761919e-03,7.55920125e-04,6.43749437e-03,9.86213203e-03,3.00853315e-04,2.49631356e-05,9.21855507e-04,1.50709094e-04,1.18292884e-05,2.72740170e-04,2.46732933e-05,4.19185104e-05,2.15563179e-03,1.53650988e-03,9.57218227e-04,2.83427178e-03]
et_features = [2.44130392e-03,1.84724190e-03,2.95427053e-03,2.07345583e-03,6.08692864e-03,1.02332254e-05,2.68189558e-04,1.97119723e-03,2.03138552e-03,0.00000000e+00,1.88084196e-03,2.48876472e-03,4.52528657e-03,5.55086244e-04,3.17686661e-04,1.89166092e-03,1.10207582e-03,2.37827637e-01,2.50027936e-03,2.97568999e-02,7.21129447e-03,1.46222082e-03,2.55333223e-04,1.59856779e-03,1.63231940e-03,1.15707979e-03,4.40001203e-03,9.57616850e-02,4.89892106e-04,8.17944479e-04,6.01355890e-02,3.25811907e-04,2.72974237e-03,1.91478068e-03,1.38912879e-02,2.31234280e-04,3.69551237e-04,1.50385696e-03,1.91123028e-02,1.43486969e-04,7.52297810e-04,6.58976466e-03,2.32858615e-04,2.28282272e-02,1.70886013e-02,2.28523831e-04,8.86793286e-02,5.40442246e-03,2.57647657e-04,4.68877685e-02,3.47368715e-03,4.99435991e-03,1.76347544e-03,5.14957349e-02,9.93757044e-03,4.09039581e-04,1.75444903e-02,1.26677042e-03,1.64946347e-02,8.86048123e-03,1.83281465e-03,1.42392325e-01,1.82838810e-02,7.75368008e-04,3.88519103e-04,5.92022542e-04,2.83057066e-03,1.74952564e-03,3.30607996e-04,6.59854041e-05,1.11579839e-03,3.91284056e-05,4.69937054e-06,1.80360247e-04,2.17977753e-05,9.75369965e-06,1.49524059e-03,9.87622365e-04,2.20976916e-03,1.83010585e-03]
ada_features = [1.08130018e-02,4.38971136e-04,5.38913894e-05,1.34497015e-02,2.86528176e-02,0.00000000e+00,0.00000000e+00,5.10804843e-05,9.71629461e-03,0.00000000e+00,8.37602783e-04,0.00000000e+00,6.65408677e-02,0.00000000e+00,1.81171601e-06,2.08868074e-05,8.15316385e-05,1.72606726e-01,2.04820329e-04,7.89474119e-03,1.70383094e-02,7.29604855e-04,5.57576342e-06,1.33353971e-03,1.37355723e-03,3.67923229e-03,4.60339778e-03,1.47074958e-03,1.41537668e-04,8.10833809e-05,1.01089068e-02,0.00000000e+00,5.70575831e-03,1.65324732e-03,2.93241460e-02,0.00000000e+00,0.00000000e+00,2.21790635e-03,6.96822707e-02,0.00000000e+00,4.91101078e-04,3.31516949e-04,0.00000000e+00,1.14535753e-02,1.52661441e-01,0.00000000e+00,1.24808951e-01,1.99725112e-03,2.36073255e-07,3.96144034e-03,1.23952962e-03,6.63243959e-03,0.00000000e+00,3.56398918e-02,3.16840226e-02,9.23685129e-05,1.80858665e-02,3.63036202e-03,2.06700886e-03,1.33560741e-02,6.31701608e-04,6.14012808e-02,3.84849483e-03,0.00000000e+00,0.00000000e+00,0.00000000e+00,1.35857797e-02,3.26101095e-02,0.00000000e+00,0.00000000e+00,3.10290359e-03,2.28936693e-07,4.64252089e-07,1.16550015e-05,0.00000000e+00,0.00000000e+00,1.43694850e-02,1.35839736e-03,7.95773940e-05,3.53275139e-04]
gb_features = [2.56592174e-03,4.45627182e-04,3.89509890e-03,6.79885048e-03,1.55808694e-02,0.00000000e+00,3.46039075e-05,1.47976278e-03,5.73261485e-04,0.00000000e+00,5.67029615e-04,2.85710265e-04,1.10642630e-02,5.23026257e-04,2.02714238e-05,1.97503055e-05,2.91153245e-04,5.80389597e-01,6.76671514e-03,1.08737474e-02,8.50787953e-03,1.33241420e-04,3.60102187e-05,1.57633903e-03,3.67455494e-04,5.38569154e-04,2.88120334e-03,7.20403096e-03,4.16144800e-04,1.73027046e-04,1.16832126e-02,9.39183523e-05,2.01440649e-03,1.61397497e-03,2.91059357e-02,2.28359151e-04,6.15123482e-04,2.42327743e-03,4.14774504e-02,7.44146379e-07,1.21220501e-04,2.50148434e-03,2.82720287e-05,1.83252467e-02,3.43887057e-02,2.51992327e-05,1.13171840e-01,1.80246655e-03,1.49985122e-04,4.01383867e-03,6.37983624e-04,4.58166076e-04,1.56456605e-03,4.32510846e-03,5.33884316e-03,9.75387345e-04,4.29350895e-03,2.20853239e-04,3.41581833e-03,4.15185857e-03,9.11077403e-04,1.86394259e-02,1.22005409e-02,5.45300748e-04,1.33484613e-04,4.73442170e-04,2.01358517e-03,3.18408910e-03,4.02786170e-04,5.97624365e-05,8.17449976e-04,1.45118353e-04,7.02069138e-06,8.37531668e-05,1.09893391e-05,3.40648927e-05,2.43572081e-03,4.37859911e-04,1.03373371e-03,3.25487842e-03]
cols = test.columns.values

# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_features,
     'Extra Trees  feature importances': et_features,
      'AdaBoost feature importances': ada_features,
    'Gradient Boost feature importances': gb_features
    })

feature_dataframe

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
x_train_dush = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svr_oof_train), axis=1)
x_test_dush = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svr_oof_test), axis=1)
print('x_train.shape : ', x_train.shape)
print('x_test.shape : ', x_test.shape)
print('x_train_dush.shape : ', x_train_dush.shape)
print('x_test_dush.shape : ', x_test_dush.shape)

gbm = xgb.XGBRegressor(
    learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 #objective= 'reg:linear',
 nthread= -1,
 scale_pos_weight=1).fit(x_train_dush, y_train)
predictions = gbm.predict(x_test_dush)
# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'Id': Id,
                            'SalePrice': predictions })
StackingSubmission.to_csv("Submission.csv", index=False)