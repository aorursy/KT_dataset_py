# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


#--------------------------------------------------------------------------

#--------------------------------------------------------------------------



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



%matplotlib inline



#plt.rcParams['figure.figsize']   = [10, 5]

#plt.rcParams['font.size']        = 16

#plt.rcParams['legend.fontsize']  = 'large'

#plt.rcParams['figure.titlesize'] = 'large'



print("Ready")


#--------------------------------------------------------------------------

# Constants

#--------------------------------------------------------------------------



pop  = 39.56e6 # CA poop

d0   = 0       # day 0

d1   = 14      # current day

d3   = 43      # day to predict to

tmin = 48      # first nonzero data point

 

conf_max = 2500

dead_max = 50



print("Ready")



#--------------------------------------------------------------------------

# Read Data

#--------------------------------------------------------------------------



import os



df_train = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

df_test  = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')



junk      =['Id','Country/Region','Lat','Long','Province/State']

junk_test =['ForecastId','Country/Region','Lat','Long','Province/State']



df_train.drop(junk    , axis=1, inplace=True)

df_test.drop(junk_test, axis=1, inplace=True)



# Drop null data at low t

dmin     = df_train['Date'].values[tmin]

day0     = "Day 0: " + dmin

df_train = df_train[tmin:]

df_train.reset_index(inplace=True)



print(df_train.shape)

df_train



#--------------------------------------------------------------------------

#--------------------------------------------------------------------------



n_pred                    = df_test.count()

df_test['ForecastId']     = df_test.index

df_test['ConfirmedCases'] = np.ones(n_pred)

df_test['Fatalities']     = np.ones(n_pred)

df_test.drop(['Date'], axis=1, inplace=True)



print(n_pred)

print(df_test.shape)



#--------------------------------------------------------------------------

# Predict & Save submission file

#--------------------------------------------------------------------------



df_test

df_test.to_csv('submission.csv')

os.listdir()



df_test
