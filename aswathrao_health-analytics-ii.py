# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train  = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics-ii/Train/train.csv')
test   = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics-ii/test.csv')
sample = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics-ii/sample_submission.csv')
print(train.shape)
print(test.shape)
(train.isna().sum() / train.shape[0])*100
(test.isna().sum() / test.shape[0])*100
plt.figure(figsize = (15,7))

sns.countplot(train['Stay'])
from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()
train['Stay'] = le.fit_transform(train['Stay'])
df = train.append(test)
df.dtypes
df.head()
cols = ['Hospital_code','City_Code_Hospital','Hospital_region_code','Severity of Illness','Visitors with Patient','Age']
plt.figure(figsize = (15,7))
for i in cols:
    sns.countplot(train[i])
    plt.show()
df[df['Bed Grade'].isna() == False].nunique()
df.drop(['case_id','patientid'],axis = 1,inplace = True)
df.isna().sum()
plt.figure(figsize = (15,7))

sns.countplot(df['Bed Grade'])
df['Bed Grade'] = np.where(df['Bed Grade'].isna(),2.0,df['Bed Grade'])
plt.figure(figsize = (15,7))

sns.countplot(df['City_Code_Patient'])
df['City_Code_Patient'] = np.where(df['City_Code_Patient'].isna(),8.0,df['City_Code_Patient'])
df = pd.get_dummies(data=df,columns=['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type',
       'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness',
       'Age'])
df.isna().sum()
df['Bed Grade'] = df['Bed Grade'].astype(int)
df['City_Code_Patient'] = df['City_Code_Patient'].astype(int)
df['Admission_Deposit'] = df['Admission_Deposit'].astype(int)
train_df = df[df['Stay'].isna() == False] 
test_df  = df[df['Stay'].isna() == True]
cat=train_df.select_dtypes(['object']).columns
print(train_df.shape)
print(test_df.shape)
test_df.drop('Stay',axis = 1,inplace = True)
train_df['Stay'] = train_df['Stay'].astype(int)
df_train, df_eval = train_test_split(train_df, test_size=0.20, random_state=3600, shuffle=True, stratify= train_df['Stay'])
feature_cols = train_df.columns.tolist()
feature_cols.remove('Stay')

label_col = 'Stay'
print(feature_cols)
cat_cols = feature_cols
cat_cols.remove('Admission_Deposit')
cat_cols
params = {}
params['learning_rate'] = 0.025
params['max_depth'] = 100
params['n_estimators'] = 1000
params['objective'] = 'multiclass'
params['boosting_type'] = 'gbdt'
params['subsample'] = 0.8
params['random_state'] = 3600
params['colsample_bytree']=0.7
params['min_data_in_leaf'] = 100
params['reg_alpha'] = 1.6
params['reg_lambda'] = 1.1
clf = lgb.LGBMClassifier(**params)
    
clf.fit(df_train[feature_cols], df_train[label_col], early_stopping_rounds=100, eval_set=[(df_train[feature_cols], df_train[label_col]), (df_eval[feature_cols], df_eval[label_col])], eval_metric='multi_error', verbose=True, categorical_feature=cat_cols)

eval_score = accuracy_score(df_eval[label_col], clf.predict(df_eval[feature_cols]))

print('Eval ACC: {}'.format(eval_score))
best_iter = clf.best_iteration_
params['n_estimators'] = best_iter
print(params)
clf = lgb.LGBMClassifier(**params)

clf.fit(train_df[feature_cols], train_df[label_col], eval_metric='multi_error', verbose=False, categorical_feature=cat_cols)

eval_score_acc = accuracy_score(train_df[label_col], clf.predict(train_df[feature_cols]))

print('ACC: {}'.format(eval_score_acc))
preds = clf.predict(test_df[feature_cols])
preds

plt.rcParams['figure.figsize'] = (12,50)
lgb.plot_importance(clf)
plt.show()
preds = le.inverse_transform(preds)
sample['Stay'] = preds
sample.to_csv('Submission_revised.csv',index = False)