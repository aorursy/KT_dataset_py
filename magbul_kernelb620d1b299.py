# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline
train_data = pd.read_csv('../input/train.csv')
train_data.shape
train_data.select_dtypes(include=[np.int64]).nunique().value_counts().sort_index().plot.bar(color='blue');

plt.xlabel('Number of Unique Values')

plt.ylabel('Count')

plt.title('Count of Unique values in Integer Columns')
train_data.select_dtypes(include = ['object']).head()

    
def ProcessObjectTypeColumns(df):

    replacements = {'yes':1,'no':0}

    df['dependency'].replace(replacements,inplace=True)

    df['edjefe'].replace(replacements,inplace=True)

    df['edjefa'].replace(replacements,inplace=True)

    #Now all the 'object' columns have 'float' data. So, convert these columns to 'float' datatype

    df['dependency'] = pd.to_numeric(df['dependency'])

    df['edjefe'] = pd.to_numeric(df['edjefe'])

    df['edjefa'] = pd.to_numeric(df['edjefa'])

    return df
train_data = ProcessObjectTypeColumns(train_data)
train_data['Target'].value_counts().sort_index().plot.bar(color='blue')
all_equal = train_data.groupby('idhogar')['Target'].apply(lambda x:x.nunique() == 1)

not_equal = all_equal[all_equal != True]

print('There are {} households where the poverty level is not same for all members'.format(len(not_equal)))
head_of_household  = train_data.groupby('idhogar')['parentesco1'].sum()

households_no_head = train_data.loc[train_data['idhogar'].isin(head_of_household[head_of_household == 0].index),:]

print('There are {} households without a head of household'.format(households_no_head['idhogar'].nunique()))
for household in not_equal.index:

    actual_target = int(train_data[(train_data['idhogar'] == household)&(train_data['parentesco1']) == 1.0]['Target'])

    train_data.loc[train_data['idhogar']==household,'Target'] = actual_target
missing_values = train_data.isnull().sum().sort_values(ascending = False)

missing_values[missing_values != 0]
def ReplaceMissingDatawithMean(df):

    df['rez_esc'].fillna((df['rez_esc'].mean()),inplace=True)

    df['v18q1'].fillna((df['v18q1'].mean()),inplace=True)

    df['v2a1'].fillna((df['v2a1'].mean()),inplace=True)

    df['meaneduc'].fillna((df['meaneduc'].mean()),inplace=True)

    df['SQBmeaned'].fillna((df['SQBmeaned'].mean()),inplace=True)

    return df
train_data = ReplaceMissingDatawithMean(train_data)
X = train_data[train_data.columns.difference(['Id','idhogar','Target'])].copy()

y = train_data['Target']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)
from sklearn.decomposition import PCA

pca = PCA().fit(X)

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)')

plt.title('PCA')

plt.show()
pca = PCA(n_components = 80)

X = pca.fit_transform(X)
df = pd.DataFrame(data = X)
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(df,y,test_size=0.3)
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

gbclf = GradientBoostingClassifier()

gbclf.fit(X_train,y_train)

y_gbclf = gbclf.predict(X_val)

cm_gbclf = confusion_matrix(y_val,y_gbclf)

cm_gbclf
score_gbclf = accuracy_score(y_val,y_gbclf)

score_gbclf
from sklearn.ensemble import RandomForestClassifier

rfclf = RandomForestClassifier()

rfclf.fit(X_train,y_train)

y_rfclf = rfclf.predict(X_val)

cm_rfclf = confusion_matrix(y_val,y_rfclf)

cm_rfclf
score_rfclf = accuracy_score(y_val,y_rfclf)

score_rfclf
from sklearn.neighbors import KNeighborsClassifier

knclf = KNeighborsClassifier()

knclf.fit(X_train,y_train)

y_knclf = knclf.predict(X_val)

cm_knclf = confusion_matrix(y_val,y_knclf)

cm_knclf
score_knclf = accuracy_score(y_val,y_knclf)

score_knclf
from sklearn.ensemble import ExtraTreesClassifier

etclf = ExtraTreesClassifier()

etclf.fit(X_train,y_train)

y_etclf = etclf.predict(X_val)

cm_etclf = confusion_matrix(y_val,y_etclf)

cm_etclf
score_etclf = accuracy_score(y_val,y_etclf)

score_etclf
from xgboost import XGBClassifier

xgclf = XGBClassifier()

xgclf.fit(X_train,y_train)

y_xgclf = xgclf.predict(X_val)

cm_xgclf = confusion_matrix(y_val,y_xgclf)

cm_xgclf
score_xgclf = accuracy_score(y_val,y_xgclf)

score_xgclf
import lightgbm as lgb

d_train = lgb.Dataset(X_train,label=y_train)

params={}

lgbclf = lgb.train(params,d_train,100)

y_lgbclf = lgbclf.predict(X_val)
test_data = pd.read_csv('../input/test.csv')
test_data = ProcessObjectTypeColumns(test_data)
test_data = ReplaceMissingDatawithMean(test_data)
X_test = test_data[test_data.columns.difference(['Id','idhogar'])].copy()

X_test = scaler.fit_transform(X_test)

X_test = pca.fit_transform(X_test)

X_test = pd.DataFrame(data = X_test)
y_test_knclf = knclf.predict(X_test)
test_data['Target'] = pd.Series(y_test_knclf)
test_data.head()