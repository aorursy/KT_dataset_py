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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

%matplotlib inline
from scipy.stats import moment



def getQuantileAray():

    l = [a  for a in np.arange(0.01,0.10,0.01)]

    m = [a  for a in np.arange(0.9,1.0,0.01)]

    return l+m



def preprocessData(df):

    #gets a aray of data and returns

    mean = df.mean()     

    skew = df.skew()    

    kurtosis = df.kurtosis()

    df_norm = (df - df.mean())/df.std()

    mean_norm = df_norm.mean()

    skew_norm = df_norm.skew()

    kurtosis_norm = df_norm.kurtosis()

    min_val = df_norm.min()

    max_val = df_norm.max()

    quant_arr = getQuantileAray() 

    df_ret = pd.DataFrame({'Mean':mean,'skew':skew,'kurtosis':kurtosis,

                           'mean_norm':mean_norm,'mean_skew':skew_norm,

                            'kurtoris_norm':kurtosis_norm,

                            'min':min_val,'max':max_val}, index=[0])

    for i,val in enumerate(quant_arr):

        df_ret['percent'+ str(i)] = df_norm.quantile(val)

      #  print(val,df.quantile(val))

    df_ret['moment'] = moment(df_norm)

    return df_ret   
filetrain = pd.read_csv('../input/train.csv',chunksize=4096)
numrow = 0

count = 0

for df in tqdm(filetrain):

    temp = preprocessData(df['acoustic_data'])

    temp['target'] = df.time_to_failure.mean()

    if count == 0:

        df_new = temp

    else:

        

        df_new = df_new.append(temp)

        

    count += 1

    if count > 5000:

        break



df_new
df_new.describe()
import xgboost as xg

from sklearn.model_selection import train_test_split

X = df_new.iloc[:,df.columns != 'target']

Y= df_new['target']

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3,random_state=47)

dtrain = xg.DMatrix(X_train,y_train)

dtest = xg.DMatrix(X_test,y_test)



parameters = {'max_depth':2,'eta':0.02,'eval_metric':['mae']}

numrounds = 10000

evallist = [(dtrain, 'train'),(dtest, 'eval') ]

model = xg.train(parameters,dtrain,numrounds,early_stopping_rounds=100,evals=evallist)

df_new.describe()

Y