# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#import lightgbm as lgb
from sklearn.metrics import mean_squared_error

import warnings


#Libraries for plots
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import plotly.figure_factory as ff
import plotly.offline as py 
import plotly.graph_objs as go 
import plotly.tools as tls

#Plots for notebook mode
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import tools 
py.init_notebook_mode (connected = True)
#Reading train and test and  data
train_data=pd.read_csv(r"../input/train_V2.csv")

test_data=pd.read_csv(r'../input/test_V2.csv')
#Printing sample train data
train_data.head(10)
#Looking at data summary for train data
train_data.info()
#Looking at data summary for test data
test_data.info()
#Looking for unique values for matchtype variable
train_data['matchType'].unique()
#Creating New Variable to store numeric values for match type column in train data

train_data.loc[train_data['matchType'] == 'squad-fpp', 'matchType_Num'] = 1
train_data.loc[train_data['matchType'] == 'duo', 'matchType_Num'] = 2
train_data.loc[train_data['matchType'] == 'solo-fpp', 'matchType_Num'] = 3
train_data.loc[train_data['matchType'] == 'squad', 'matchType_Num'] = 4
train_data.loc[train_data['matchType'] == 'duo-fpp', 'matchType_Num'] = 5
train_data.loc[train_data['matchType'] == 'solo', 'matchType_Num'] = 6
train_data.loc[train_data['matchType'] == 'normal-squad-fpp', 'matchType_Num'] = 7
train_data.loc[train_data['matchType'] == 'crashfpp', 'matchType_Num'] = 8
train_data.loc[train_data['matchType'] == 'flaretpp', 'matchType_Num'] = 9
train_data.loc[train_data['matchType'] == 'normal-solo-fpp', 'matchType_Num'] = 10
train_data.loc[train_data['matchType'] == 'flarefpp', 'matchType_Num'] = 11
train_data.loc[train_data['matchType'] == 'normal-duo-fpp', 'matchType_Num'] = 12
train_data.loc[train_data['matchType'] == 'normal-duo', 'matchType_Num'] = 13
train_data.loc[train_data['matchType'] == 'normal-squad', 'matchType_Num'] = 14
train_data.loc[train_data['matchType'] == 'crashtpp', 'matchType_Num'] = 15
train_data.loc[train_data['matchType'] == 'normal-solo', 'matchType_Num'] = 16
#Creating New Variable to store numeric values for match type column in test data

test_data.loc[test_data['matchType'] == 'squad-fpp', 'matchType_Num'] = 1
test_data.loc[test_data['matchType'] == 'duo', 'matchType_Num'] = 2
test_data.loc[test_data['matchType'] == 'solo-fpp', 'matchType_Num'] = 3
test_data.loc[test_data['matchType'] == 'squad', 'matchType_Num'] = 4
test_data.loc[test_data['matchType'] == 'duo-fpp', 'matchType_Num'] = 5
test_data.loc[test_data['matchType'] == 'solo', 'matchType_Num'] = 6
test_data.loc[test_data['matchType'] == 'normal-squad-fpp', 'matchType_Num'] = 7
test_data.loc[test_data['matchType'] == 'crashfpp', 'matchType_Num'] = 8
test_data.loc[test_data['matchType'] == 'flaretpp', 'matchType_Num'] = 9
test_data.loc[test_data['matchType'] == 'normal-solo-fpp', 'matchType_Num'] = 10
test_data.loc[test_data['matchType'] == 'flarefpp', 'matchType_Num'] = 11
test_data.loc[test_data['matchType'] == 'normal-duo-fpp', 'matchType_Num'] = 12
test_data.loc[test_data['matchType'] == 'normal-duo', 'matchType_Num'] = 13
test_data.loc[test_data['matchType'] == 'normal-squad', 'matchType_Num'] = 14
test_data.loc[test_data['matchType'] == 'crashtpp', 'matchType_Num'] = 15
test_data.loc[test_data['matchType'] == 'normal-solo', 'matchType_Num'] = 16
train_data['matchType_Num']=train_data['matchType_Num'].astype(int)
#Preparing dataset for models

#Define X,Y

#Y = diagnosis (target)
#X = features

# Def X and Y
Y = train_data['winPlacePerc']

Y = Y.fillna(0) #Making na values to zeros

Y = Y.astype(int)


X = train_data.drop(['winPlacePerc','Id','groupId','matchId','matchType'],axis=1)
#Standardization

X=pd.DataFrame(StandardScaler().fit_transform(X),columns = X.columns)


#Running LDA Model
LDA = LinearDiscriminantAnalysis()
LDA.fit(X, Y)

#Preparing test data for prediction
test_LDA=test_data.drop(['Id','groupId','matchId','matchType'],axis=1)
#Predicting values for test data
pred=LDA.predict(test_LDA)
#Predicting probability for test data 
pred_prob=LDA.predict_proba(test_LDA)
test_data['winPlacePerc']=pred
#Submitting results

sub=test_data[['Id','winPlacePerc']]

sub.to_csv('submission_LDA.csv',index=False)

print(os.listdir())