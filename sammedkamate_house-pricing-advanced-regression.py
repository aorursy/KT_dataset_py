# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_org=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train.head()
train.shape

all=pd.concat([train,test_org],sort=False)
np.array(all.isnull().sum())
sns.heatmap(train.isnull(),yticklabels=False, cbar=False,cmap='viridis')
all.drop(['Id','Alley','PoolQC','Fence','MiscFeature'],axis=1, inplace=True)

all.shape
train.info()
cat_data=[]

num_data=[]

for col in all:

    if all[col].dtypes==object:

       cat_data.append(col)

    else:

        num_data.append(col)    

        

del num_data[-1]

num_data
mszone=all['MSZoning'].value_counts()

sns.barplot(mszone.index,mszone.values)

plt.title('Distribution of MSZoning')

plt.xlabel('Carrier', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()
for col in cat_data:

    all[col].fillna(all[col].value_counts().mode()[0],inplace=True)

for col in num_data:

    all[col].fillna(all[col].mean(),inplace=True)
np.array(all.isnull().sum())
all=pd.get_dummies(all,columns=cat_data,drop_first=True)

all.shape
train=all[all['SalePrice'].notna()]

test=all[all['SalePrice'].isna()]

test.drop('SalePrice',axis=1,inplace=True)

y=train['SalePrice']

x=train.drop('SalePrice',axis=1)



import xgboost



classifier=xgboost.XGBRegressor()
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_squared_error



hyperparameter={

    'n_estimators':[100,200,500,900,1100],

    'learning_rate':[0.05,0.1,0.15,0.2],

    'booster':['gbtree','gblinear'],

    'max_depth':[2,3,5,10,15],

    'min_child_weight':[2,3,4,5,6],

    'base_Score':[0.25,0.5,0.75,1]

    

}
randomcv=RandomizedSearchCV(estimator=classifier,param_distributions=hyperparameter,cv=10,n_iter=25,n_jobs=4,

                            scoring='neg_root_mean_squared_error',return_train_score=True,verbose=5,random_state=1)
randomcv.fit(x,y)

randomcv.best_estimator_
classifier=xgboost.XGBRegressor(base_Score=0.25, base_score=0.5, booster='gbtree',

             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,

             gamma=0, gpu_id=-1, importance_type='gain',

             interaction_constraints='', learning_rate=0.1, max_delta_step=0,

             max_depth=2, min_child_weight=2,

             monotone_constraints='()', n_estimators=1100, n_jobs=0,

             num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,

             scale_pos_weight=1, subsample=1, tree_method='exact',

             validate_parameters=1, verbosity=None)

classifier.fit(x,y)

test_pred=classifier.predict(test)
test_pred=pd.DataFrame(test_pred)

submit=pd.concat([test_org['Id'],test_pred],axis=1)

submit.columns=['Id','SalePrice']

submit
submit.to_csv("submission.csv", index = False)