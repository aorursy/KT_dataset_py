from time import time
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
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
train_df = pd.read_csv(r'/kaggle/input/network-intrusion-detection/Train_data.csv')
test_df = pd.read_csv(r'/kaggle/input/network-intrusion-detection/Test_data.csv')
print("Train dataset shape - ",train_df.shape)
print("Test dataset shape - ",test_df.shape)
pd.set_option('display.max_columns', None)
train_df.head()
test_df.head()
train_df.info()
t0 = time()
print(train_df.groupby('protocol_type')['protocol_type'].count())
time() - t0
t0 = time()
pd.set_option('display.max_row', None)
print(train_df.groupby('class')['class'].count())
time() - t0
t0 = time()
print(train_df.groupby('flag')['flag'].count())
time() - t0
col_names = train_df.columns
type(col_names)
num_cols = col_names.drop(['protocol_type', 'flag', 'service'])
corr_df = train_df[num_cols].corr()
sns.heatmap(corr_df)
train_df['num_outbound_cmds'].unique()
train_df.drop('num_outbound_cmds', axis = 1, inplace = True)
highly_correlated_df = (corr_df.abs() > 0.9) & (corr_df.abs() < 1.0) 
corr_var_index = (highly_correlated_df == True).any()
corr_var_names = corr_var_index[corr_var_index == True].index

de_duplicate = []
corr_pairs = []

for i in corr_var_index.index:
    row = highly_correlated_df[i]
    de_duplicate.append(i)
    for j in corr_var_names:
        if j not in de_duplicate and row[j] == True:
            print(i,j,": ", corr_df.loc[i,j])
            corr_pairs.append((i,j))


train_df.drop(['num_root', 'srv_serror_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
              'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'], axis = 1, inplace = True)
train_df.info()
categorical_columns = ['protocol_type', 'service', 'flag']
train_df[categorical_columns].head()
label_encoder = preprocessing.LabelEncoder()
train_df['protocol_type'] = label_encoder.fit_transform(train_df['protocol_type'])
train_df['service'] = label_encoder.fit_transform(train_df['service'])
train_df['flag'] = label_encoder.fit_transform(train_df['flag'])
train_df['class'] = label_encoder.fit_transform(train_df['class'])
train_df[categorical_columns].head()
#p value is 0.0 which is less than significant value. Hence service and class are not independent
chi2_contingency(pd.crosstab(train_df['service'], train_df['class']))
#P value is 0.0 which is less than significant value. Hence flag and class are not independent
chi2_contingency(pd.crosstab(train_df['flag'], train_df['class']))
#p value is 0.0 which is less than significant value. Hence service and class features are not independent.
chi2_contingency(pd.crosstab(train_df['service'], train_df['class']))
Y = train_df['class']
train_df.drop('class', axis=1, inplace = True)

X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(train_df, Y, test_size = 0.3)
model = LogisticRegression()
model = model.fit(X_train, Y_train)
pred = model.predict(X_valid)
metrics.confusion_matrix(Y_valid, pred)
metrics.f1_score(Y_valid, pred)
test_df.drop('num_outbound_cmds', axis = 1, inplace = True)
test_df.drop(['num_root', 'srv_serror_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
              'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'], axis = 1, inplace = True)
test_df['protocol_type'] = label_encoder.fit_transform(test_df['protocol_type'])
test_df['service'] = label_encoder.fit_transform(test_df['service'])
test_df['flag'] = label_encoder.fit_transform(test_df['flag'])


test_df.head()
model.predict(test_df)