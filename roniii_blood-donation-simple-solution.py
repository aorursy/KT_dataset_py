# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedShuffleSplit
import collections
from sklearn.metrics import make_scorer, log_loss
from matplotlib import pyplot as plt

from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier 
# Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
# Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv") 
test = pd.read_csv("../input/test.csv")

print(train.head(2))
#train = train.columns[1:]
#test = test.columns[1:]
train.drop(train.columns[[0]],axis =1,inplace=True)
print ('Datasets:' , 'train:' , train.shape )
#ids = test.columns[0]
ids = test[test.columns[0]]
print(ids)
test.drop(test.columns[[0]],axis =1,inplace=True)
print ('Datasets:' , 'test:' , test.shape )
print ('Datasets:' , 'ids:' , ids.shape )
print ('Description of train:' , train.describe )
print ('Train head:' , train.head(5) )
print('Train columns with null values:\n', train.isnull().sum())
print("-"*10)
print('Test/Validation columns with null values:\n', test.isnull().sum())
print("-"*10)

############## Split Training and Testing Data
Target = train[['Made Donation in March 2007']]
trainX = train[['Months since Last Donation','Number of Donations','Total Volume Donated (c.c.)','Months since First Donation']]

train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(trainX, Target, random_state = 0)
print('Target',Target)
model = RandomForestClassifier(n_estimators=100)
model.fit( trainX , Target )
# Score the model 
print (model.score( train_x , train_y ) , model.score( test_x , test_y ))
print(test.shape)
test_Y = model.predict(test) 
test_Y = float(np(test_Y)) 
#ids = train.Unnamed 
test = pd.DataFrame( { 'Id': ids , 'can donate': test_Y } ) 
#test = pd.DataFrame( {  'can donate': test_Y } ) 
test.shape 
test.head() 
test.to_csv( 'canDonate.csv' , index = False )