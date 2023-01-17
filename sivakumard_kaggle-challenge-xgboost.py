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
from sklearn.model_selection import train_test_split  

from sklearn.model_selection import cross_val_score



import xgboost as xgb

import time



import numpy as np

import pandas as pd
from scipy.stats import norm

from scipy.stats import kurtosis

from scipy.stats import skew

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
train_df = pd.read_csv("../input/X_train.csv")

test_df = pd.read_csv("../input/X_test.csv")

y = pd.read_csv("../input/y_train.csv")

sub = pd.read_csv("../input/sample_submission.csv")
def _kurtosis(x):

    return kurtosis(x)



def CPT5(x):

    den = len(x)*np.exp(np.std(x))

    return sum(np.exp(x))/den



def skewness(x):

    return skew(x)



def SSC(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1 

    xn_i1 = x[0:len(x)-2]  # xn-1

    ans = np.heaviside((xn-xn_i1)*(xn-xn_i2),0)

    return sum(ans[1:]) 



def wave_length(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1 

    return sum(abs(xn_i2-xn))

    

def norm_entropy(x):

    tresh = 3

    return sum(np.power(abs(x),tresh))



def SRAV(x):    

    SRA = sum(np.sqrt(abs(x)))

    return np.power(SRA/len(x),2)



def mean_abs(x):

    return sum(abs(x))/len(x)



def zero_crossing(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1

    return sum(np.heaviside(-xn*xn_i2,0))

def feature_extraction(raw_frame):

    frame = pd.DataFrame()

    raw_frame['angular_velocity'] = raw_frame['angular_velocity_X'] + raw_frame['angular_velocity_Y'] + raw_frame['angular_velocity_Z']

    raw_frame['linear_acceleration'] = raw_frame['linear_acceleration_X'] + raw_frame['linear_acceleration_Y'] + raw_frame['linear_acceleration_Y']

    raw_frame['velocity_to_acceleration'] = raw_frame['angular_velocity'] / raw_frame['linear_acceleration']

    

    for col in raw_frame.columns[3:]:

        frame[col + '_mean'] = raw_frame.groupby(['series_id'])[col].mean()        

        frame[col + '_CPT5'] = raw_frame.groupby(['series_id'])[col].apply(CPT5) 

        frame[col + '_SSC'] = raw_frame.groupby(['series_id'])[col].apply(SSC) 

        frame[col + '_skewness'] = raw_frame.groupby(['series_id'])[col].apply(skewness)

        frame[col + '_wave_lenght'] = raw_frame.groupby(['series_id'])[col].apply(wave_length)

        frame[col + '_norm_entropy'] = raw_frame.groupby(['series_id'])[col].apply(norm_entropy)

        frame[col + '_SRAV'] = raw_frame.groupby(['series_id'])[col].apply(SRAV)

        frame[col + '_kurtosis'] = raw_frame.groupby(['series_id'])[col].apply(_kurtosis) 

        frame[col + '_mean_abs'] = raw_frame.groupby(['series_id'])[col].apply(mean_abs) 

        frame[col + '_zero_crossing'] = raw_frame.groupby(['series_id'])[col].apply(zero_crossing) 

    return frame
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
def multiclass_accuracy(preds, train_data):

    labels = train_data.get_label()

    pred_class = np.argmax(preds.reshape(9, -1).T, axis=1)

    return 'multi_accuracy', np.mean(labels == pred_class), True
train_df=feature_extraction(train_df)
test_df=feature_extraction(test_df)
le = LabelEncoder()

target = le.fit_transform(y['surface'])
# Create 0.75/0.25 train/test split

X_train, X_test, y_train, y_test = train_test_split(train_df, target, test_size=0.25, train_size=0.75,

                                                    random_state=42)
# Leave most parameters as default

param = {'objective': 'multi:softmax', # Specify multiclass classification

         'num_class': 9, # Number of possible output classes

         'tree_method': 'hist' # Use gpu_hist for GPU accelerated algorithm.

         }
# Convert input data from numpy to XGBoost format

dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)
dtext_X = xgb.DMatrix(test_df)
num_round = 500

gpu_res = {} # Store accuracy result

tmp = time.time()

# Train model

bst=xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)

print("CPU Training Time: %s seconds" % (str(time.time() - tmp))) #tried to time it aganst GPU 
predictions = bst.predict(dtext_X)  
sub['surface'] = le.inverse_transform(predictions.astype(int))

sub.to_csv('submission.csv', index=False)
sub.head()
# Leave most parameters as default

param = {'objective': 'multi:softprob', # usieng Probilities for multiclass classification

         'num_class': 9, # Number of possible output classes

         'tree_method': 'gpu_hist' 

         }
num_round = 500 

gpu_res = {} # Store accuracy result

tmp = time.time()

# Train model

bst=xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)

print("GPU Training Time: %s seconds" % (str(time.time() - tmp))) #tried to time it aganst GPU 
predictions = bst.predict(dtext_X)  
predictions.shape
sub['surface'] = le.inverse_transform(predictions.argmax(axis=1))

sub.to_csv('submission-xgb-prob.csv', index=False)