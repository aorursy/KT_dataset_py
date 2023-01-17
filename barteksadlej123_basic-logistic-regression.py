import numpy as np

import pandas as pd

import os,time,random,tqdm

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import sklearn

from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.multioutput import MultiOutputClassifier

from sklearn.utils import shuffle



# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



import tensorflow as tf



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_targets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

# train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

sample_submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
train_features
train_targets
test_features
sample_submission
train_features.describe()
test_features.describe()
fig,ax = plt.subplots(2,2,figsize=(10,10))

sns.countplot(x='cp_type',data=train_features,ax=ax[0][0])

ax[0][0].set_title('Train cp type')

sns.countplot(x='cp_dose',data=train_features,ax=ax[0][1])

ax[0][1].set_title('Train cp dose')

sns.countplot(x='cp_type',data=test_features,ax=ax[1][0])

ax[1][0].set_title('Test cp type')

sns.countplot(x='cp_dose',data=test_features,ax=ax[1][1])

ax[1][1].set_title('Test cp dose')

train_features['cp_time'] = train_features['cp_time'].apply(lambda x:str(x))

test_features['cp_time'] = test_features['cp_time'].apply(lambda x:str(x))

train_features = train_features.join(pd.get_dummies(train_features[['cp_time','cp_type','cp_dose']])).drop(['cp_time','cp_type','cp_dose'],axis=1)

test_features = test_features.join(pd.get_dummies(test_features[['cp_time','cp_type','cp_dose']])).drop(['cp_time','cp_type','cp_dose'],axis=1)
test_ids = test_features['sig_id']

test_features.drop(['sig_id'],axis=1,inplace=True)



train_features.drop(['sig_id'],axis=1,inplace=True)

train_targets.drop(['sig_id'],axis=1,inplace=True)



X,y = np.array(train_features.values),np.array(train_targets.values)
X,y = shuffle(X,y)

# X,y=X[:100],y[:100]

# y[0,:]=1



# cv = KFold(n_splits=5,shuffle=True)



pipeline = sklearn.pipeline.make_pipeline(StandardScaler(),MultiOutputClassifier(LogisticRegression(verbose=True),n_jobs=-1))

# score = cross_val_score(pipeline,train_features,train_targets,cv=cv,n_jobs=5)

# score
start = time.time()



pipeline.fit(X,y)



stop = time.time()



print(f'Time taken: {stop-start}')
test_X = np.array(test_features.values)

test_X
preds = pipeline.predict_proba(test_X)

preds
preds = np.array(preds)[:,:,1]

preds
sample_submission
sample_submission[sample_submission.columns.to_list()[1:]] = preds.T
sample_submission
sample_submission.to_csv('submission.csv',index=False)