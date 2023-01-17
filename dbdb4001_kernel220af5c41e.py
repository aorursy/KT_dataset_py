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
data=pd.read_csv('../input/train.csv')
data.info()
data.columns[data.dtypes==object]
data['dependency'].unique()
data[(data['dependency']=='no') & (data['SQBdependency']!=0)]
data[(data['dependency']=='yes') & (data['SQBdependency']!=1)]
data[(data['dependency']=='3') & (data['SQBdependency']!=9)]
data['dependency']=np.sqrt(data['SQBdependency'])
data['edjefe'].unique()
data['edjefa'].unique()
data['SQBedjefe'].unique()
data[['edjefe', 'edjefa', 'SQBedjefe']][:20]
data[['edjefe', 'edjefa', 'SQBedjefe']][data['edjefe']=='yes']
data[(data['edjefe']=='yes') & (data['edjefa']!='no')]
data[(data['edjefa']=='yes') & (data['parentesco1']==1)][['edjefe', 'edjefa', 'parentesco1', 'escolari']]
data[data['edjefe']=='yes'][['edjefe', 'edjefa','age', 'escolari', 'parentesco1','male', 'female', 'idhogar']]
data[(data['edjefe']=='no') & (data['edjefa']=='no')][['edjefe', 'edjefa', 'age', 'escolari', 'female', 'male', 'Id', 'parentesco1', 'idhogar']]
data[(data['edjefe']=='yes') & data['parentesco1']==1][['escolari']]
conditions = [

    (data['edjefe']=='no') & (data['edjefa']=='no'), #both no

    (data['edjefe']=='yes') & (data['edjefa']=='no'), # yes and no

    (data['edjefe']=='no') & (data['edjefa']=='yes'), #no and yes 

    (data['edjefe']!='no') & (data['edjefe']!='yes') & (data['edjefa']=='no'), # number and no

    (data['edjefe']=='no') & (data['edjefa']!='no') # no and number

]

choices = [0, 1, 1, data['edjefe'], data['edjefa']]

data['edjefx']=np.select(conditions, choices)

data['edjefx']=data['edjefx'].astype(int)

data[['edjefe', 'edjefa', 'edjefx']][:15]
data.describe()
data.columns[data.isna().sum()!=0]
data[data['meaneduc'].isnull()]
data[data['meaneduc'].isnull()][['Id','idhogar','edjefe','edjefa', 'hogar_adul', 'hogar_mayor', 'hogar_nin', 'age', 'escolari']]
print(len(data[data['idhogar']==data.iloc[1291]['idhogar']]))

print(len(data[data['idhogar']==data.iloc[1840]['idhogar']]))

print(len(data[data['idhogar']==data.iloc[2049]['idhogar']]))
meaneduc_nan=data[data['meaneduc'].isnull()][['Id','idhogar','escolari']]
me=meaneduc_nan.groupby('idhogar')['escolari'].mean().reset_index()
me
for row in meaneduc_nan.iterrows():

    idx=row[0]

    idhogar=row[1]['idhogar']

    m=me[me['idhogar']==idhogar]['escolari'].tolist()[0]

    data.at[idx, 'meaneduc']=m

    data.at[idx, 'SQBmeaned']=m*m

    
data['v2a1'].isnull().sum()
norent=data[data['v2a1'].isnull()]

print("Owns his house:", norent[norent['tipovivi1']==1]['Id'].count())

print("Owns his house paying installments", norent[norent['tipovivi2']==1]['Id'].count())

print("Rented ", norent[norent['tipovivi3']==1]['Id'].count())

print("Precarious ", norent[norent['tipovivi4']==1]['Id'].count())

print("Other ", norent[norent['tipovivi5']==1]['Id'].count())

print("Total ", 6860)
data['v2a1']=data['v2a1'].fillna(0)
data['v18q1'].isna().sum()
tabletnan=data[data['v18q1'].isnull()]

tabletnan[tabletnan['v18q']==0]['Id'].count()
data['v18q1'].unique()
data['v18q1']=data['v18q1'].fillna(0)
data['rez_esc'].isnull().sum()
data['rez_esc'].describe()
data['rez_esc'].unique()
data[data['rez_esc']>1][['age', 'escolari', 'rez_esc']][:20]
rez_esc_nan=data[data['rez_esc'].isnull()]

rez_esc_nan[(rez_esc_nan['age']<18) & rez_esc_nan['escolari']>0][['age', 'escolari']]
data['rez_esc']=data['rez_esc'].fillna(0)
d={}

weird=[]

for row in data.iterrows():

    idhogar=row[1]['idhogar']

    target=row[1]['Target']

    if idhogar in d:

        if d[idhogar]!=target:

            weird.append(idhogar)

    else:

        d[idhogar]=target
len(set(weird))
data[data['idhogar']==weird[2]][['idhogar','parentesco1', 'Target']]
for i in set(weird):

    hhold=data[data['idhogar']==i][['idhogar', 'parentesco1', 'Target']]

    target=hhold[hhold['parentesco1']==1]['Target'].tolist()[0]

    for row in hhold.iterrows():

        idx=row[0]

        if row[1]['parentesco1']!=1:

            data.at[idx, 'Target']=target
data[data['idhogar']==weird[1]][['idhogar','parentesco1', 'Target']]
def data_cleaning(data):

    data['dependency']=np.sqrt(data['SQBdependency'])

    data['rez_esc']=data['rez_esc'].fillna(0)

    data['v18q1']=data['v18q1'].fillna(0)

    data['v2a1']=data['v2a1'].fillna(0)

    

    conditions = [

    (data['edjefe']=='no') & (data['edjefa']=='no'), #both no

    (data['edjefe']=='yes') & (data['edjefa']=='no'), # yes and no

    (data['edjefe']=='no') & (data['edjefa']=='yes'), #no and yes 

    (data['edjefe']!='no') & (data['edjefe']!='yes') & (data['edjefa']=='no'), # number and no

    (data['edjefe']=='no') & (data['edjefa']!='no') # no and number

    ]

    choices = [0, 1, 1, data['edjefe'], data['edjefa']]

    data['edjefx']=np.select(conditions, choices)

    data['edjefx']=data['edjefx'].astype(int)

    data.drop(['edjefe', 'edjefa'], axis=1, inplace=True)

    

    meaneduc_nan=data[data['meaneduc'].isnull()][['Id','idhogar','escolari']]

    me=meaneduc_nan.groupby('idhogar')['escolari'].mean().reset_index()

    for row in meaneduc_nan.iterrows():

        idx=row[0]

        idhogar=row[1]['idhogar']

        m=me[me['idhogar']==idhogar]['escolari'].tolist()[0]

        data.at[idx, 'meaneduc']=m

        data.at[idx, 'SQBmeaned']=m*m

        

    return data
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data['Target'].hist()
data_undersampled=data.drop(data.query('Target == 4').sample(frac=.75).index)
data_undersampled['Target'].hist()
X=data_undersampled.drop(['Id', 'idhogar', 'Target', 'edjefe', 'edjefa'], axis=1)

y=data_undersampled['Target']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train.shape
y_train.shape
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
clf = RandomForestClassifier()

params={'n_estimators': list(range(40,61, 1))}

gs = GridSearchCV(clf, params, cv=5)
gs.fit(X_train, y_train)
preds=gs.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, preds))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, preds))
print(gs.best_params_)

print(gs.best_score_)

print(gs.best_estimator_)
cvres = gs.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(mean_score), params)
test_data=pd.read_csv('../input/test.csv')
test_data=data_cleaning(test_data)
ids=test_data['Id']

test_data.drop(['Id', 'idhogar'], axis=1, inplace=True)
test_predictions=gs.predict(test_data)
test_predictions[:5]
submit=pd.DataFrame({'Id': ids, 'Target': test_predictions})
submit.to_csv('submit.csv', index=False)