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
import sys

!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/


import cudf 

import cupy as cp

from cuml.neighbors import KNeighborsRegressor

from cuml import SVR

from cuml.linear_model import Ridge, Lasso

from cuml.metrics import mean_absolute_error, mean_squared_error

import pandas as pd

import pydicom

import os

import numpy as np

#from matplotlib import cm

from matplotlib import pyplot as plt

import cv2

#import seaborn as sns

from tqdm import tqdm
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

train_df.drop_duplicates(keep = False, inplace = True, subset = ['Patient', 'Weeks'])

test_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")





sub_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")

sub_df['Patient'] = sub_df['Patient_Week'].apply(lambda x:x.split('_')[0])

sub_df['Weeks'] = sub_df['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub_df =  sub_df[['Patient','Weeks','Confidence','Patient_Week']]

sub_df = sub_df.merge(test_df.drop('Weeks', axis=1), on="Patient")

train_df['WHERE'] = 'train'

test_df['WHERE'] = 'val'

sub_df['WHERE'] = 'test'



df = train_df.append([test_df, sub_df])

df['min_week'] = df['Weeks']

df.loc[df.WHERE=='test','min_week'] = np.nan

df['min_week'] = df.groupby('Patient')['min_week'].transform('min')
baseline_df = df.loc[df.Weeks == df.min_week]

baseline_df = baseline_df[['Patient', 'FVC']].copy()

baseline_df.columns = ['Patient', 'min_FVC']

baseline_df['nb'] = 1

baseline_df['nb'] = baseline_df.groupby('Patient')['nb'].transform('cumsum')

baseline_df = baseline_df[baseline_df.nb==1]

baseline_df.drop('nb', axis=1, inplace=True)

df = df.merge(baseline_df, on='Patient', how='left')

df['base_week'] = df['Weeks'] - df['min_week']

# convert string labels to numeric labels

columns = ['Sex', 'SmokingStatus']

features = []

for feat in columns: 

    for mode in df[feat].unique():

        features.append(mode)

        df[mode]  = (df[feat] == mode).astype(int)

    

features += ['Age', 'Percent', 'min_FVC', 'base_week']

features
# Zero-center normalization 



for feat in features:

    df[feat] = (df[feat] - np.mean(df[feat]))/np.std(df[feat])

df.head()
cudf = cudf.from_pandas(df)

tr_cudf = cudf.loc[cudf.WHERE=='train']

val_cudf = cudf.loc[cudf.WHERE=='val']

test_cudf = cudf.loc[cudf.WHERE=='test']
# evaluation metric for the comp

def score(y_true, y_pred):

    C1, C2 = cp.asarray(70.0), cp.asarray(70.0)

    sigma = y_pred[:, 2] - y_pred[:, 0]

    fvc_pred = y_pred[:, 1]

    

    #sigma_clip = sigma + C1

    sigma_clip = cp.maximum(sigma, C1)

    delta = cp.absolute(y_true - fvc_pred)

    delta = cp.minimum(delta, C2)

    sq2 = cp.sqrt( 2.0)

    metric = (delta / sigma_clip)*sq2 + cp.log(sigma_clip* sq2)

    return cp.mean(metric)


from sklearn.model_selection import KFold



Kfold = 5

kf = KFold(n_splits=Kfold)



X, y, X_test = tr_cudf[features].values, tr_cudf['FVC'].values, test_cudf[features].values
%%time





model_score, model_zoo, y_preds_scores = [], [], []

for c1, c2, c3 in [(1, 50, 500)]:

    

    y_preds = cp.zeros((X.shape[0], 3))

    y_test_preds = cp.zeros((X_test.shape[0], 3))

    

    model_container = []

    SVR_kfold_ensemble,Ridge_kfold_ensemble = [], [] 

    for train_ind, val_ind in kf.split(X):

        X_train, X_val = X[train_ind,:], X[val_ind,:]

        y_train, y_val = y[train_ind], y[val_ind]

        

        model_1 = SVR(C=c1, cache_size=3000.0)

        model_1.fit(X_train, y_train)

        

        model_2 = SVR(C=c2, cache_size=3000.0)

        model_2.fit(X_train, y_train)

        

        model_3 = SVR(C=c3, cache_size=3000.0)

        model_3.fit(X_train, y_train)

        

        

        

        y_preds[val_ind,0] = model_1.predict(X_val)

        y_preds[val_ind,1] = model_2.predict(X_val)

        y_preds[val_ind,2] = model_3.predict(X_val)

        

        y_test_preds[:,0] += model_1.predict(X_test) 

        y_test_preds[:,1] += model_2.predict(X_test) 

        y_test_preds[:,2] += model_3.predict(X_test) 

        

    y_test_preds *= 1/cp.asarray(Kfold)

    y_preds_scores.append(y_test_preds)

    model_score.append(score(y,y_preds))

    model_zoo.append([model_1, model_2, model_3])
model_score, y_preds_scores
y_np = cp.asnumpy(y)

y_preds_np = cp.asnumpy(y_preds)

idxs = np.random.randint(0, y_np.shape[0], 100)

plt.plot(y_np[idxs], label="ground truth")

plt.plot(y_preds_np[idxs, 0], alpha = 0.5, label="c=1")

plt.plot(y_preds_np[idxs, 1],  alpha = 0.5, label="c=5")

plt.plot(y_preds_np[idxs, 2],  alpha = 0.5, label="c=500")

plt.legend(loc="best")

plt.show()
test_cudf['FVC1'] = 1.0*y_test_preds[:,1]

test_cudf['Confidence1'] = y_test_preds[:,2] - y_test_preds[:,0]

test_cudf
submission_cudf = test_cudf[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()

submission_cudf.loc[~submission_cudf.FVC1.isnull()].head(10)

submission_cudf["FVC"] = submission_cudf["FVC1"]

submission_cudf["Confidence"] = submission_cudf["Confidence1"]
submission_cudf.describe().T
submission_cudf[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)