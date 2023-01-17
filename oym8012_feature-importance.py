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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv("../input/lish-moa/train_features.csv")
train_targets = pd.read_csv("../input/lish-moa/train_targets_scored.csv")
def preprocess(df):
    df = df.copy()
    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
    df.loc[:, 'cp_time'] = df.loc[:, 'cp_time']/72
    del df['sig_id']
    return df
train = preprocess(train)
del train_targets['sig_id']
#SelectFromModel_by_RandomForest
cols = train_targets.columns
selected_features = pd.DataFrame(columns = train.columns,index=['vote'])
selected_features.fillna(0, inplace=True)
clf = RandomForestClassifier(n_estimators=100,random_state=8012)
X_train, X_test, y_train, y_test = train_test_split(train, train_targets,shuffle=True,random_state=8012)
for c, column in enumerate(cols,1):
    print('Model:',c)
    y_tr = y_train.iloc[:,c-1]
    y_te = y_test.iloc[:,c-1]
    select=SelectFromModel(clf, threshold='2*median')
    select.fit(X_train,y_tr)
    mask = select.get_support()
    selected_features[selected_features.columns[mask]] += 1  #if feature importance ranking is 1/4 or more, give 1 point
#FeatureImportance_sorted
pd.set_option('display.max_columns', None)
selected_features.sort_values(by='vote',axis=1,ascending=False)
plt.figure(figsize=(100, 50))
sns.barplot(data=selected_features)
plt.figure(figsize=(100, 50))
sns.barplot(data=selected_features.sort_values(by='vote',axis=1,ascending=False))
