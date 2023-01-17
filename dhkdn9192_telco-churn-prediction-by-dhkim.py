# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn import metrics

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!ls ../input -al
data_df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')

data_df.info()
data_df.head()
data_df.loc[data_df['TotalCharges'] == ' ', 'TotalCharges'] = '0'
data_df['TotalCharges'] = pd.to_numeric(data_df['TotalCharges'])
data_df['TotalCharges'].describe()
label_enc_cols = ['gender', 'Partner', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 

                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 

                  'Contract', 'PaperlessBilling', 'PaymentMethod']

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

target_col = ['Churn']
for feat in tqdm(label_enc_cols):

    enc = LabelEncoder()

    data_df[feat] = enc.fit_transform(data_df[feat])

    

data_df[label_enc_cols].head()

data_df['Churn'].value_counts()
data_df.loc[data_df['Churn'] == 'No', 'Churn'] = 0

data_df.loc[data_df['Churn'] == 'Yes', 'Churn'] = 1

data_df['Churn'].describe()
minmax_enc = MinMaxScaler(feature_range=(0, 1))

data_df[numeric_cols] = minmax_enc.fit_transform(data_df[numeric_cols])



data_df[numeric_cols].describe()
plt.figure(figsize=(13,10))

sns.heatmap(data = data_df.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
train_df, test_df = train_test_split(data_df, test_size=0.2)

print('data_df :', data_df.shape)

print('train_df :', train_df.shape)

print('test_df :', test_df.shape)
train_df['Churn'].value_counts()
test_df['Churn'].value_counts()
xgb = XGBClassifier(max_depth=3,

                    learning_rate=0.1,

                    n_estimators=100,

                    verbosity=1,

                    objective='binary:logistic',

                    booster='gbtree',

                    n_jobs=1,

                    gamma=0,

                    min_child_weight=1,

                    max_delta_step=0,

                    subsample=1,

                    colsample_bytree=1,

                    colsample_bylevel=1,

                    colsample_bynode=1,

                    reg_alpha=0,

                    reg_lambda=1,

                    scale_pos_weight=1,

                    base_score=0.5,

                    random_state=0)
history = xgb.fit(X=train_df[label_enc_cols + numeric_cols], y=train_df['Churn'])
res_score = xgb.score(X=test_df[label_enc_cols + numeric_cols], y=test_df['Churn'])

res_score
sample_c0 = test_df[test_df['Churn'] == 0]

xgb.score(X=sample_c0[label_enc_cols + numeric_cols], y=sample_c0['Churn'])
sample_c1 = test_df[test_df['Churn'] == 1]

xgb.score(X=sample_c1[label_enc_cols + numeric_cols], y=sample_c1['Churn'])
pred = xgb.predict_proba(test_df[label_enc_cols + numeric_cols])

fpr, tpr, thresholds = metrics.roc_curve(y_true=test_df['Churn'].tolist(), y_score=[a[1] for a in pred])

res_auc = metrics.auc(fpr, tpr)



print(f'res_auc : {res_auc}')
