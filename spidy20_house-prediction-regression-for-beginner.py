import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

from sklearn.preprocessing import LabelEncoder  ###for encode a categorical values

from sklearn.model_selection import train_test_split  ## for spliting the data

from lightgbm import LGBMRegressor    ## for import our model

from sklearn.preprocessing import LabelEncoder

print(os.listdir("../input"))
train_dataset = pd.read_csv('../input/train.csv')
train_dataset.head()
train_dataset.shape
x = train_dataset.iloc[:,1:-1]

y = train_dataset.iloc[:,-1] 
x.isnull().sum()
col_miss_val = [col for col in train_dataset.columns if train_dataset[col].isnull().any()]

print(col_miss_val)
for col in col_miss_val:

    if(x[col].dtype == np.dtype('O')):

         x[col]=x[col].fillna(x[col].value_counts().index[0])    #replace nan with most frequent

    else:

        x[col] = train_dataset[col].fillna(x[col].median()) 
x.isnull().sum()
##So first we will find a columns thats contain characters value 

x.select_dtypes(include=['object'])
LE = LabelEncoder()

for col in x.select_dtypes(include=['object']):

    x[col] = LE.fit_transform(x[col])
x.head()
y.isnull().sum()
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y ,test_size = 0.1,random_state = 0)
lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=8,

                                       learning_rate=0.0385, 

                                       n_estimators=3500,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose= 0,

                                       )
lightgbm.fit(x_train,y_train)
lightgbm.score(x_test,y_test)
test_dataset = pd.read_csv('../input/test.csv')
test_dataset.isnull().sum()

test_dataset = test_dataset.iloc[:,1:]
test_dataset.isnull().sum()
test_col_miss_val = [col for col in test_dataset.columns if test_dataset[col].isnull().any()]

print(test_col_miss_val)
for col in test_col_miss_val:

    if(test_dataset[col].dtype == np.dtype('O')):

        test_dataset[col] = test_dataset[col].fillna(test_dataset[col].value_counts().index[0])    #replace nan with most frequent

        

    else:

        test_dataset[col] = test_dataset[col].fillna(test_dataset[col].median()) 
for col in test_dataset.select_dtypes(include=['object']):

    test_dataset[col] = LE.fit_transform(test_dataset[col])   
test_dataset.head()
prediction = lightgbm.predict(test_dataset)
print(prediction)
ss = pd.read_csv('../input/sample_submission.csv')
output = pd.DataFrame({'Id': ss.Id,'SalePrice': prediction})

output.to_csv('submission.csv', index=False)

output.head()