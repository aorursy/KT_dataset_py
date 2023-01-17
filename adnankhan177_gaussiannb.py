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
train_df=pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test_df=pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
train_df.head()
train_df.shape
train_df.isnull().sum().sum()
test_df.isnull().sum().sum()
train_df.describe()
train_df.dtypes
count_class_0, count_class_1 = train_df.target.value_counts()
df_class_0 = train_df[train_df['target'] == 0]
df_class_1 = train_df[train_df['target'] == 1]
df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under.target.value_counts())

df_test_under.target.value_counts().plot(kind='bar', title='Count (target)');
df_test_under.head()
X_train=df_test_under.iloc[:,2:]
y_train=df_test_under['target'].values
pd.DataFrame(X_train[y_train==0]).plot.kde(ind=100, legend=False)
print('KDE for -ve classes')
pd.DataFrame(X_train[y_train==1]).plot.kde(ind=100, legend=False)
print('KDE for +ve classes')
from sklearn.preprocessing import StandardScaler
scaled=pd.DataFrame(StandardScaler().fit_transform(X_train))
scaled[y_train==0].plot.kde(ind=100, legend=False)
print('KDE for -ve classes after normalization')
scaled[y_train==1].plot.kde(ind=100, legend=False)
print('KDE for +ve classes after normalization')
from sklearn.preprocessing import QuantileTransformer
quantile_transformed=pd.DataFrame(QuantileTransformer(output_distribution='normal').fit_transform(X_train))
quantile_transformed[y_train==0].plot.kde(ind=100, legend=False)
print('KDE for -ve classes after quantile_transformed')
quantile_transformed[y_train==1].plot.kde(ind=100, legend=False)
print('KDE for -ve classes after quantile_transformed')
X_train.head()
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB

pipeline=make_pipeline(QuantileTransformer(output_distribution='normal'), GaussianNB())

pipeline.fit(X_train,y_train)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr,tpr,thr=roc_curve(y_train,pipeline.predict_proba(X_train)[:,1])
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Plot')
auc(fpr, tpr)
from sklearn.model_selection import cross_val_score

cross_val_score(pipeline,X_train,y_train,scoring='roc_auc',cv=10).mean()
X_test=test_df.iloc[:,1:].values.astype('float64')
X_test.shape

submission=pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
submission['target']=pipeline.predict_proba(X_test)[:,1]
submission.to_csv('submission.csv',index=False)
submission.head(50)
