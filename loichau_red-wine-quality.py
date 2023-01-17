# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report

def test_model(X,y):
    rfc = RandomForestClassifier(n_estimators = 200)
    rfc.fit(X,y)
    print('Model score is : ' + str(rfc.score(X,y)))
    
    print('-'*40)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    rfc = RandomForestClassifier(n_estimators = 200)
    rfc.fit(X_train, y_train)
    pred_rfc = rfc.predict(X_test)
    print(classification_report(y_test, pred_rfc))
def check_high_cut (df1, on = 'alpha'):
    df=df1.copy()
    le=LabelEncoder()
    columns = df.columns.tolist()
    columns.remove(on)
    result = {}
    for i in columns:
        data = df[[i,on]]
        for x in range (2,10):
            data[i+' cut '+str(x)] =pd.cut(data[i],x)
            data[i+' cut '+str(x)] = le.fit_transform(data[i+' cut '+str(x)])
            try: 
                data[i+' qcut '+str(x)]= pd.qcut(data[i],x)
                data[i+' qcut '+str(x)] = le.fit_transform(data[i+' qcut '+str(x)])
            except:
                pass
        origin = data.corr()[on].sort_values(ascending=False).reset_index()
        fetch = pd.concat( 
            [origin,
            origin['index'].str.extract(r'(?P<cut>cut|qcut)'),
            origin['index'].str.extract(r'(?P<num>\d)')], axis =  1)
        fetch['quality'] = np.abs(fetch['quality'])
        fetch=fetch[fetch.quality != 1][~fetch['cut'].isnull()]
        max_count = fetch[fetch.quality != 1][~fetch['cut'].isnull()].sort_values(by='num')[fetch['quality'] == fetch['quality'].max()]
        if len(max_count) > 1:
            max_count= max_count[max_count['num'] == max_count['num'].min()].reset_index(drop = True).to_dict()
        else:
            max_count = max_count.reset_index(drop=True).to_dict()
        result[i]= [max_count['cut'][0],max_count['num'][0]]
        if result[i][0] == 'cut':
            df[i] = pd.cut(df[i],int(result[i][1]))
            df[i] = le.fit_transform(df[i])
        elif result[i][0] =='qcut':
            df[i] = pd.qcut(df[i],int(result[i][1]))
            df[i] = le.fit_transform(df[i])
        print(str(i) +' '+  str(result[i]))
    return df
import warnings 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
warnings.filterwarnings('ignore')
import seaborn as sns
import json

df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.describe()
fig, axes = plt.subplots(3,4, figsize = (20,15))
quality = df['quality'].unique().tolist()
quality.sort()
columns = df.columns.tolist()
columns.remove('quality')
for i in range (12):
    for u in quality:
        try :
            filtered_df = df[df['quality']==u][columns[i]]
            if i <4:
                sns.distplot(filtered_df,hist=False,kde=True,label =u, ax = axes[0,i])
            elif i >=4 and i < 8:
                sns.distplot(filtered_df,hist=False,kde=True,label =u, ax = axes[1,i-4])
            elif i >= 8:
                sns.distplot(filtered_df,hist=False,kde=True,label =u, ax = axes[2,i-8])
        except:
            pass
plt.figure(figsize= (10,5))
sns.heatmap(df.corr(), annot = True, cbar = False)
z=df[['free sulfur dioxide','total sulfur dioxide','quality']]
z['percent of free']  = z['free sulfur dioxide'] / z['total sulfur dioxide']
z['log'] = np.log(z['percent of free'])
check_high_cut(z,on = 'quality')

#update to DataFrame
df['% of free sulfur'] = df['free sulfur dioxide'] / df['total sulfur dioxide']
df['% of free sulfur'] = pd.qcut(df['% of free sulfur'],5)
df['% of free sulfur'] = le.fit_transform(df['% of free sulfur'])
df.drop(['free sulfur dioxide', 'total sulfur dioxide'], axis = 1, inplace = True)
df[:5]
test = df[['quality','alcohol','residual sugar', 'density']]
test['new_column'] = 0
test.loc[df['alcohol'] <=10.265, 'new_column']=0
test.loc[(df['alcohol'] > 10.265) & (df['alcohol'] <= 11.4659), 'new_column'] =1
test.loc[df['alcohol'] >11.4659, 'new_column'] =2
test['a'] = (np.log(test['residual sugar']) + (test['new_column']))/test['density']
check_high_cut(test[['a','quality']], on = 'quality')
test['a'] = pd.cut(test['a'],4)
test['a'] = le.fit_transform(test['a'])
test[['a','quality']].corr()

#update to df
df.loc[df['alcohol'] <=10.265, 'alcohol']=0
df.loc[(df['alcohol'] > 10.265) & (df['alcohol'] <= 11.4659), 'alcohol'] =1
df.loc[df['alcohol'] >11.4659, 'alcohol'] =2
df['sugar_alcohol/density'] = (df['alcohol'] + np.log(df['residual sugar']))/ df['density']
df['sugar_alcohol/density']=pd.cut(df['sugar_alcohol/density'], 4)
df['sugar_alcohol/density'] = le.fit_transform(df['sugar_alcohol/density'])
df.drop(['alcohol', 'residual sugar', 'density'], axis = 1, inplace = True)
df[:5]

test = df[['fixed acidity','citric acid', 'pH','quality']]
test['test fix'] = np.log(test['fixed acidity']) / test['pH']
test['test citric'] = (test['citric acid']) / test['pH']
test['fix_citric/pH'] = test['test fix'] + test['test citric']
check_high_cut(test[['fix_citric/pH', 'quality']], on = 'quality')

#update to df
df['fix_citric/pH'] = (np.log(df['fixed acidity']) /df['pH']) + (df['citric acid'] / df['pH'])
df['fix_citric/pH'] = pd.cut(df['fix_citric/pH'],2)
df['fix_citric/pH'] = le.fit_transform(df['fix_citric/pH'])
df.drop(['fixed acidity','citric acid', 'pH'], axis = 1, inplace =  True)
df[:5]
test = df[['chlorides','sulphates','quality']] 
check_high_cut(test,on='quality')
test['chlorides'] = pd.cut(test['chlorides'], 3)
test['chlorides'] = le.fit_transform(test['chlorides'])
test['sulphates'] = pd.qcut(test['sulphates'],6)
test['sulphates'] = le.fit_transform(test['sulphates'])
test['sum'] = test['chlorides'] + test['sulphates']
check_high_cut(test[['sum','quality']], on = 'quality')
test['sum']= pd.qcut(test['sum'],3)
test['sum'] = le.fit_transform(test['sum'])
test.corr()

#Update df
df['chlorides'] = pd.cut(df['chlorides'], 3)
df['chlorides'] = le.fit_transform(df['chlorides'])
df['sulphates'] = pd.qcut(df['sulphates'],6)
df['sulphates'] = le.fit_transform(df['sulphates'])
df['sum'] = df['chlorides'] + df['sulphates']
df['chlo_sulph'] = pd.qcut(df['sum'],3)
df['chlo_sulph'] = le.fit_transform(df['chlo_sulph'])
df.drop(['chlorides','sulphates', 'sum'], axis = 1, inplace = True)
df[:5]
df['volatile acidity'] = pd.cut(df['volatile acidity'], 2)
df['volatile acidity'] = le.fit_transform(df['volatile acidity'])
X=df.drop('quality', axis =1)
y=df['quality']
test_model(X,y)
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
new_df = check_high_cut(df, on='quality')
X=new_df.drop('quality', axis =1)
y=new_df['quality']
test_model(X,y)
fig, axes = plt.subplots(1,2, figsize= (15,5))
plt.figure(figsize= (10,5))
sns.heatmap(df.corr(),annot=True, cbar = False, ax = axes[0])
sns.heatmap(new_df.corr(),annot=True, cbar = False, ax = axes[1])
df1 = df.copy()
df1['free sulfur over total'] = df1['free sulfur dioxide'] / df1['total sulfur dioxide']
df1.drop(['free sulfur dioxide', 'total sulfur dioxide'], axis = 1, inplace = True)
df1 = check_high_cut(df1, on ='quality')
X=df1.drop('quality', axis =1)
y=df1['quality']
test_model(X,y)