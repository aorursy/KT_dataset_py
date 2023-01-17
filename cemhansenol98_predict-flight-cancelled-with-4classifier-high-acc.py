# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_2019 = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2019_ontime.csv')

df_2020 = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2020_ontime.csv')
df_2020.head()
df_2020.tail()
print(df_2020.shape)

print(df_2020['Unnamed: 21'].isnull().sum())
def bar_plot(variable):

    var = df_2020[variable] # get feature

    varValue = var.value_counts() # count number of categorical variable(value/sample)

    

    plt.figure(figsize = (9,6))

    plt.bar(varValue.index,varValue)

    plt.xticks(varValue.index,varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{} \n {}".format(variable,varValue))
bar_plot('CANCELLED')
print(df_2020.columns)

print(df_2020.shape[1])
df_2020.info()
column_names = df_2020.columns

j=0

for i in df_2020.columns:

    print("  {} has got {} Null Sample " .format(df_2020.columns[j],df_2020[i].isnull().sum()))

    j=j+1
import missingno as msno

plt.figure(figsize=(4,4))

msno.bar(df_2020)
msno.heatmap(df_2020) 
#Data Preprocessing

df_2020 = df_2020.drop(['Unnamed: 21'],axis=1)

df_2020.shape
#Drop NaN TAIL_NUM rows

df_2020 = df_2020.dropna(subset=['TAIL_NUM'])

print(df_2020['TAIL_NUM'].isna().sum())

print(df_2020.shape)
df_2020['DEP_DEL15'] = df_2020['DEP_DEL15'].replace(np.NaN,0)

df_2020['DEP_DEL15'].isnull().sum()
df_2020['ARR_DEL15'] = df_2020['ARR_DEL15'].replace(np.NaN,0)

df_2020['ARR_DEL15'].isnull().sum()
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')

#DEP_TIME



df_2020['DEP_TIME'] = imp_mean.fit_transform(df_2020[['DEP_TIME']])

#ARR_TIME



df_2020['ARR_TIME'] = imp_mean.fit_transform(df_2020[['ARR_TIME']])
column_names = df_2020.columns

j=0

for i in df_2020.columns:

    print("  {} has got {} NaN Sample " .format(df_2020.columns[j],df_2020[i].isnull().sum()))

    j=j+1
df_2020.shape
import seaborn as sns

f,ax= plt.subplots(figsize=(15,15))

sns.heatmap(df_2020.corr(),linewidths=.5,annot=True,fmt='.4f',ax=ax)

plt.show()
df_2020 = df_2020.drop(['DEST_AIRPORT_SEQ_ID'],axis=1)

df_2020 = df_2020.drop(['ORIGIN_AIRPORT_SEQ_ID'],axis=1)

print(df_2020.shape)
bar_plot('CANCELLED')
y = df_2020.CANCELLED

df_2020 = df_2020.drop('CANCELLED',axis=1)

X = df_2020
categorical_columns = ['OP_CARRIER','OP_UNIQUE_CARRIER','TAIL_NUM','ORIGIN','DEST','DEP_TIME_BLK']

for col in categorical_columns:

    X_encoded = pd.get_dummies(X[col],prefix_sep = '_')

    df_2020 = df_2020.drop([col],axis=1)



df_2020 = pd.concat([df_2020, X_encoded], axis=1)
X = df_2020


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,shuffle=True,random_state=42)
from sklearn.tree import DecisionTreeClassifier

clf_dt = DecisionTreeClassifier(random_state = 0)

model_dt = clf_dt.fit(X_train, y_train) 
from sklearn import tree

tree.plot_tree(model_dt) 
from sklearn import metrics

y_pred = model_dt.predict(X_test)

print(metrics.classification_report(y_test,y_pred))
y_test.value_counts()
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(max_depth=50)

model_rf = clf_rf.fit(X_train, y_train)
from sklearn import metrics

y_pred = model_rf.predict(X_test)

print(metrics.classification_report(y_test,y_pred))
from sklearn.ensemble import AdaBoostClassifier

clf_ab = RandomForestClassifier()

model_ab = clf_ab.fit(X_train, y_train)
from sklearn import metrics

y_pred = model_ab.predict(X_test)

print(metrics.classification_report(y_test,y_pred))
import xgboost as xgb

clf_xgb = xgb.XGBClassifier()

model_xgb = clf_xgb.fit(X_train, y_train)
from sklearn import metrics

y_pred = model_xgb.predict(X_test)

print(metrics.classification_report(y_test,y_pred))