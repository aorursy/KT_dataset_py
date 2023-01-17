# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

pd.set_option('use_inf_as_na', True)



# Import statements required for Plotly 

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls





from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, log_loss



import csv

from sklearn import preprocessing, neighbors

from sklearn.preprocessing import StandardScaler 

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier 

from sklearn.tree import DecisionTreeClassifier

import pandas as pd

import patsy

import pylab as P

import sys

from matplotlib import pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')





# Import and suppress warnings

import warnings

warnings.filterwarnings('ignore')
path_test = '../input/equipfails/equip_failures_test_set.csv'

path_train = '../input/equipfails/equip_failures_training_set.csv'

#path_test = '/content/equip_failures_test_set.csv.zip'

#path_train = '/content/equip_failures_training_set.csv.zip'
trainData = pd.read_csv(path_train, na_values = 'na')

testData = pd.read_csv(path_test, na_values = 'na')

trainData.head()
#checking for nnull entries

trainData.isnull().any()
target = trainData['target']

trainData.drop('target', inplace=True, axis=1)

target.head()
testId = testData['id']

testData.drop('id', inplace=True, axis=1)

trainData.drop('id', inplace=True, axis=1)

trainData.head()
test_count = testData.shape[0]

train_count = trainData.shape[0]

print(test_count)

print(train_count)

completeData = trainData

completeData=completeData.append(testData, ignore_index=True)

print(completeData.shape[0])
def fill_everything(dat):

    columns=dat.columns.values.tolist()

    for col in columns:

        dat[col].fillna(dat[col].mean(skipna=True), inplace=True)

    return dat
completeData=fill_everything(completeData)

completeData.head()
completeData.describe()
def make_log(dat):

    columns=dat.columns.values.tolist()

    for col in columns:

        result = np.log(dat[col])

        result.replace([np.inf, -np.inf], np.nan)

        dat[col+"log"] = result

        #dat[col] = result

        dat[col+"log"].fillna(dat[col+"log"].mean(skipna=True), inplace=True)

    return dat
completeData = make_log(completeData)

completeData.head()
#checking for nnull entries

completeData.isnull().any()
completeData.describe()
scaler = preprocessing.StandardScaler().fit(completeData)

completeData=pd.DataFrame(scaler.transform(completeData), columns=completeData.columns)

completeData.head()
# def make_dummies(dat):

#     columns=dat.columns.values.tolist()

#     for col in columns:

#         n=len(set(dat[col].values.tolist()))

#         if n<=5:

#             dummies = pd.get_dummies(dat[col],prefix=col)

#             dat = pd.concat([dat, dummies],axis=1)

#     return dat
# completeData=make_dummies(completeData)

# completeData.head()
train=completeData.ix[0:train_count-1]

test=completeData.ix[train_count:]

train.head()
print(train_count)

print(train.shape)
# Import the train_test_split method

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



# Split data into train and test sets as well as for validation and testing

trainData, validData, target_train, target_val = train_test_split(train, target, train_size= 0.75,random_state=0);

#train, test, target_train, target_val = StratifiedShuffleSplit(attrition_final, target, random_state=0);
#Random Forest Classifier

seed = 2   # We set our random seed to zero for reproducibility

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 400,

    'warm_start': True, 

    'max_features': 0.3,

    'max_depth': 9,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'random_state' : seed,

    'verbose': 0

}

rf = RandomForestClassifier(**rf_params)
trainData.describe()
rf.fit(trainData, target_train)

#print("Fitting of Random Forest has finished")
rf_predictions = rf.predict(validData)

print("Predictions finished")
accuracy_score(target_val, rf_predictions)
rf_predictions = rf.predict(test)

predicted=pd.DataFrame(rf_predictions, columns=['target'])

test_out = pd.concat([testId, predicted],axis=1)
test_out.to_csv('randomForestOut.csv', index=False)
trainData.head()
features = pd.DataFrame()

features['feature'] = trainData.columns.values.tolist()

features['importance'] = rf.feature_importances_

important_features=features.sort_values('importance', ascending=False)['feature'].values.tolist()

important_features=important_features[0:70]

print(important_features)

print(features)
extracted_train=trainData[important_features]

extracted_test=test[important_features]

extracted_val=validData[important_features]
#feature Ranking with random forest

# Scatter plot 

trace = go.Scatter(

    y = rf.feature_importances_,

    x = trainData.columns.values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 13,

        #size= rf.feature_importances_,

        #color = np.random.randn(500), #set color equal to a variable

        color = rf.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = trainData.columns.values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Random Forest Feature Importance',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')
#visualizing tree diagram with Graphviz

from sklearn import tree

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

import re



decision_tree = tree.DecisionTreeClassifier(max_depth = 4)

decision_tree.fit(extracted_train, target_train)



# Predicting results for valid dataset

y_pred = decision_tree.predict(extracted_val)

print("Accuracy:",accuracy_score(target_val, y_pred))

# Export our trained model as a .dot file

with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(decision_tree,

                              out_file=f,

                              max_depth = 4,

                              impurity = False,

                              feature_names = extracted_train.columns.values,

                              class_names = ['No', 'Yes'],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")
import xgboost as xgb

clf = xgb.XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.01,  

                      colsample_bytree = 0.25,

                      subsample = 0.8,

                      objective='binary:logistic', 

                      n_estimators=1500, 

                      reg_alpha = 0.3,

                      max_depth=5, 

gamma=10)
# clf.fit(extracted_train, target_train, eval_metric='auc', verbose=False,

#             eval_set=[(extracted_val, target_val)], early_stopping_rounds=200)

# y_pre = clf.predict(extracted_val)

# #y_pro = clf.predict_proba(test_data)[:, 1]

# #print "AUC Score : %f" % metrics.roc_auc_score(test_label, y_pro)

# print("Accuracy : %.4g" % accuracy_score(target_val, y_pre))
clf2 = xgb.XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.01,  

                      colsample_bytree = 0.8,

                      subsample = 0.8,

                      objective='binary:logistic', 

                      n_estimators=500, 

                      reg_alpha = 0.005,

                      nthread=4,

                      max_depth=7, 

gamma=10)

clf2.fit(trainData, target_train, eval_metric='logloss', verbose=True,

            eval_set=[(validData, target_val)], early_stopping_rounds=200)

y_pre = clf2.predict(validData)

#y_pro = clf.predict_proba(test_data)[:, 1]

#print "AUC Score : %f" % metrics.roc_auc_score(test_label, y_pro)

print("Accuracy : %.4g" % accuracy_score(target_val, y_pre))
xgb_predictions = clf2.predict(test)

xgb_predicted=pd.DataFrame(xgb_predictions, columns=['target'])

test_out = pd.concat([testId, xgb_predicted],axis=1)

test_out.to_csv('xgbOutlogLossfinal.csv', index=False)
# Gradient Boosting Parameter

gb_params ={

    'n_estimators': 500,

    'max_features': 0.9,

    'learning_rate' : 0.2,

    'max_depth': 11,

    'min_samples_leaf': 2,

    'subsample': 1,

    'max_features' : 'sqrt',

    'random_state' : seed,

    'verbose': 0

}
gb = GradientBoostingClassifier(**gb_params)

# Fit the model to our SMOTEd train and target

gb.fit(trainData, target_train)

# Get our predictions

gb_predictions = gb.predict(validData)

print("Predictions have finished")
accuracy_score(target_val, gb_predictions)
gb_predictions_test = gb.predict(test)
# Scatter plot 

trace = go.Scatter(

    y = gb.feature_importances_,

    x = train.columns.values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 13,

        #size= rf.feature_importances_,

        #color = np.random.randn(500), #set color equal to a variable

        color = gb.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = train.columns.values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Gradient Boosting Model Feature Importance',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter')