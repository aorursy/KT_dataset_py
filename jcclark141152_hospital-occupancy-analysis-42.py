# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('max_columns', 500)

df_train = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/train_data.csv')

df_test = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/test_data.csv')
X_train, X_test, y_train, y_test = train_test_split(df_train.drop(columns = 'Stay', axis=1), df_train['Stay'], test_size=0.2, random_state=0)
X_train.head()
X_train.info()
X_train.describe()
X_train.isnull().sum()
sns.countplot(x='Bed Grade', data=X_train)
X_train['Bed Grade'].fillna(X_train['Bed Grade'].mode()[0], inplace=True)

X_test['Bed Grade'].fillna(X_train['Bed Grade'].mode()[0], inplace=True)

df_test['Bed Grade'].fillna(df_train['Bed Grade'].mode()[0], inplace = True)

df_train['Bed Grade'].fillna(df_train['Bed Grade'].mode()[0], inplace = True)
X_train.columns
X_train['repeat'] = X_train.groupby('patientid')['case_id'].transform('count')

X_test['repeat'] = X_test.groupby('patientid')['case_id'].transform('count')

df_test['repeat'] = df_test.groupby('patientid')['case_id'].transform('count')

df_train['repeat'] = df_train.groupby('patientid')['case_id'].transform('count')
X_train.isnull().sum()
X_train.drop(columns=['patientid','case_id'], inplace = True)

X_test.drop(columns=['patientid','case_id'], inplace = True)

df_test.drop(columns=['patientid','case_id'], inplace=True)

df_train.drop(columns=['patientid','case_id'], inplace=True)
X_train.isnull().sum()
X_train['City_Code_Patient'].nunique()
plt.figure(figsize=(13,5))

sns.countplot(x='City_Code_Patient', data=X_train)

plt.xticks(rotation = 90)

plt.show()
X_train['City_Code_Patient'].fillna(X_train['City_Code_Patient'].mode()[0], inplace=True)

X_test['City_Code_Patient'].fillna(X_train['City_Code_Patient'].mode()[0], inplace=True)

df_test['City_Code_Patient'].fillna(df_train['City_Code_Patient'].mode()[0], inplace=True)

df_train['City_Code_Patient'].fillna(df_train['City_Code_Patient'].mode()[0], inplace=True)
X_train.isnull().sum()
cat_vars = [c for c in X_train.columns if X_train[c].dtypes=='object']

num_vars = [c for c in X_train.columns if X_test[c].dtypes!='object']
for var in cat_vars:

    map_feats = dict(zip(X_train[var].unique(), range(len(X_train[var]))))

    X_train.replace(map_feats,inplace=True)

    X_test.replace(map_feats, inplace=True)

    df_test.replace(map_feats, inplace=True) 

    df_train.replace(map_feats, inplace=True)
stay_map = dict(zip(y_train.unique(), range(len(y_train))))

y_train.replace(stay_map,inplace = True)

y_test.replace(stay_map, inplace=True)

df_train['Stay'].replace(stay_map, inplace = True)
from catboost import CatBoostClassifier, Pool



eval_set = Pool(X_test, y_test)



cbc = CatBoostClassifier(iterations = 100, learning_rate=.5, depth = 5, loss_function='MultiClass', eval_metric='Accuracy')



cbc.fit(X_train, y_train, eval_set = eval_set, verbose = False)

yhat = cbc.predict(Pool(X_test))

cbc.get_best_score()
from catboost.utils import get_confusion_matrix



conf_matrix = get_confusion_matrix(cbc, eval_set)

np.set_printoptions(suppress=True)

print(conf_matrix)

sns.heatmap(conf_matrix, cmap='coolwarm', linewidth=1)
inv_map = {v: k for k, v in stay_map.items()}
yhat = map(lambda x: x[0], yhat)

yhat = pd.Series(yhat)

t = yhat.replace(inv_map)
plt.figure(figsize=(15,5))

plt.title('Predicted Hospital Stay Times')

plt.xlabel("Number of Days")

sns.countplot(t, order=t.value_counts().index)
plt.figure(figsize=(15,5))

p = y_test.replace(inv_map)

plt.title('Test Set Hospital Stay Times')

plt.xlabel("Number of Days")

sns.countplot(p, order=p.value_counts().index)