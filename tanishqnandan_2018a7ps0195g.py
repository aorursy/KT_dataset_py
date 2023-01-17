# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy import mean

import xgboost as xgb

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import train_test_split

from imblearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))









# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/minorprojectinput/train.csv")





print( df.shape )



df.drop( ['id'], axis = 1, inplace=True )





df.drop_duplicates( inplace=True )



print( df.shape )



labels = df.columns[:-1]



X = df[labels]



y = df['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)



print(X_test.shape)





#df.describe()
target_count=df.target.value_counts()



print('Class 0: ', target_count[0] )

print('Class 1: ', target_count[1] )



print(' Proportion: ', round(target_count[0] / target_count[1], 2 ) , ': 1' )



target_count.plot(kind='bar', title='Count(target)' );
over = SMOTE(sampling_strategy=0.1)

under = RandomUnderSampler(sampling_strategy=0.5)



steps = [('over', over), ('under', under)]



pipeline = Pipeline(steps=steps)



X_train , y_train = pipeline.fit_resample(X_train, y_train)



print(X_train.shape)

print(y_train.shape)


xgb_model = xgb.XGBClassifier(tree_method="gpu_hist")



parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'objective':['binary:logistic'],

              'learning_rate': [0.05], #so called `eta` value

              'max_depth': [6],

              'min_child_weight': [11],

              'silent': [1],

              'subsample': [0.8],

              'colsample_bytree': [0.7],

              'n_estimators': [5], #number of trees, change it to 1000 for better results

              'missing':[-999],

              'seed': [1337]}





cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)



clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 

                   cv=cv, 

                   scoring='roc_auc',

                   verbose=2, refit=True)



clf.fit(X_train,y_train)
yhat=clf.predict(X)



auc = roc_auc_score(yhat,y)



print(auc)

df_test=pd.read_csv("/kaggle/input/minorprojectinput/test.csv")



labels = df_test.columns[1:]



X_submit = df_test[labels]



target_submit = clf.predict(X_submit)



sample=pd.DataFrame(columns=['id','target'])



sample['id'] = df_test['id']

sample['target'] = target_submit

sample.to_csv("submission.csv", index=False)