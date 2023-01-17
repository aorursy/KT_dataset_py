# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score
import os

print((os.listdir('../input')))
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')     #obtaining data from training set

df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')       #obtaining data from test set
df_test['G3']=df_test['F3']**4                                #feature variable manipulations for test set

df_test['G4']=df_test['F3']*df_test['F4']

df_test['G5']=df_test['F13']*df_test['F16']

df_test['G6']=df_test['F4']*df_test['F8']

df_test['G7']=df_test['F11']**df_test['F8']

df_test['G8']=df_test['F5']*df_test['F7']

df_test.head()
df_train['G3']=df_train['F3']**4                          #feature variable manipulations for training set

df_train['G4']=df_train['F3']*df_train['F4']

df_train['G5']=df_train['F13']*df_train['F16']

df_train['G6']=df_train['F4']*df_train['F8']

df_train['G7']=df_train['F11']**df_train['F8']

df_train['G8']=df_train['F5']*df_train['F7']

df_train.head()
test_index=df_test['Unnamed: 0']                   
df_train.drop(['F1','F2'], axis = 1, inplace = True)              #dropping features from training set  
train_X = df_train.loc[:, 'F3':'G8'].drop('O/P',axis=1)             #feature and target split

train_y = df_train.loc[:, 'O/P']
rf = RandomForestRegressor(n_estimators=350, random_state=42)            #implementing the model
rf.fit(train_X, train_y)                                #training on the dataset

rf.score(train_X,train_y)
df_test = df_test.loc[:, 'F3':'G8']

pred = rf.predict(df_test)                                      #predicting on the test set
print(pred)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred)                  #storing the predicted output in form of dataframe

result.head()
result.to_csv('output.csv', index=False)
