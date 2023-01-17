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
data = pd.read_csv('/kaggle/input/telecom-company-churn/Churn.csv')

data.head()
data.shape
def levels(df):

    return (pd.DataFrame({'dtype':df.dtypes, 

                         'levels':df.nunique(), 

                         'levels':[df[x].unique() for x in df.columns],

                         'null_values':df.isna().sum(),

                         'unique':df.nunique()}))

levels(data)
data.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns

#plt.figure(figsize = (8,8))

b = sns.countplot(data.Churn)

b.set_title('Churn',fontsize = 16)

b.set_xlabel("Churn",fontsize=14)

plt.show()
data.groupby(['MultipleLines','Churn']).agg({'Churn':'count'}).plot(kind='bar')
data.groupby(['InternetService','Churn']).agg({'Churn':'count'}).plot(kind='bar')
data.groupby(['MultipleLines','InternetService','Churn']).agg({'Churn':'count'}).plot(kind='bar')
data.groupby(['StreamingTV','Churn']).agg({'Churn':'count'}).plot(kind='bar')
data.groupby(['StreamingTV','StreamingMovies','Churn']).agg({'Churn':'count'}).plot(kind='bar')
sns.boxplot(data.Churn,data.MonthlyCharges,orient='v')
bins_list = [0,19,39,59,79,99,120]

data['month_bin']=pd.Series()

data['month_bin']=pd.cut(data['MonthlyCharges'],bins_list)
data['TotalCharges'].dtypes
data['TotalCharges'][(pd.to_numeric(data['TotalCharges'], errors='coerce').isnull())]=0
data[(pd.to_numeric(data['TotalCharges'], errors='coerce').isnull())]
data['TotalCharges'] = data['TotalCharges'].astype('float')
med = data['TotalCharges'][(data['TotalCharges']!=0)&(data['Churn']=='No')].median()



med
data['TotalCharges'][(data['TotalCharges']==0)] = med
sns.boxplot(data.Churn,data.TotalCharges,orient='v')
bins_list = [0,1999,3999,5999,7999,9999]

data['tc_bin']=pd.Series()

data['tc_bin']=pd.cut(data['TotalCharges'],bins_list)
data['TotalCharges'][data['Churn']=='Yes'].max()
data['TotalCharges'][data['Churn']=='Yes'].min()
data['TotalCharges'][data['Churn']=='Yes'].describe()
np.quantile(data['TotalCharges'],0.95)
data['TotalCharges'][(data['Churn']=='Yes')&(data['TotalCharges']>=7000)] = 7000
data.set_index('customerID',inplace=True)
data['Churn'] = pd.Series(map(lambda x: dict(Yes=1, No=0)[x],

              data['Churn'].values.tolist()), data.index)
x = data.copy().drop('Churn',axis=1)

y = data['Churn']
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.2, random_state = 200)
x_train.columns
num_cols = ['MonthlyCharges', 'TotalCharges']

cat_cols = x_train.columns.difference(num_cols)

cat_cols
for col in cat_cols:

    x_train[col] = x_train[col].astype('category')

    x_val[col] = x_val[col].astype('category')
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

scale.fit(x_train[num_cols])

x_train[num_cols] = scale.transform(x_train[num_cols])

x_val[num_cols] = scale.transform(x_val[num_cols])
x_train.dtypes
x_train = pd.get_dummies(x_train,columns=cat_cols,drop_first=False)

x_val = pd.get_dummies(x_val,columns=cat_cols,drop_first=False)
x_train.shape,x_val.shape,y_train.shape,y_val.shape
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=0.01,penalty='l2',solver='saga',random_state=200,max_iter=1000)

lr.fit(x_train,y_train)

train_pred_lr = lr.predict(x_train)

val_pred_lr = lr.predict(x_val)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_val, val_pred_lr)

print("Accuracy on train is:",accuracy_score(y_train,train_pred_lr))

print("Accuracy on val is:",accuracy_score(y_val,val_pred_lr))
cm
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

parameters={'max_depth':range(3,21),

           'max_features':range(20,91,10)}

dt = GridSearchCV(DecisionTreeClassifier(),param_grid=parameters,n_jobs=-1,cv=10,scoring='accuracy')

dt.fit(x_train,y_train)
dt.best_params_
grid_model = dt.best_estimator_
grid_model.fit(x_train,y_train)
train_pred_gm = grid_model.predict(x_train)

val_pred_gm = grid_model.predict(x_val)



print("Accuracy on train is:",accuracy_score(y_train,train_pred_gm))

print("Accuracy on val is:",accuracy_score(y_val,val_pred_gm))
confusion_matrix(y_val, val_pred_gm)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc



parameters={'n_estimators':range(100,401,200),

           'max_features':range(10,51,10),

           'max_depth':[5,9,13],

           'bootstrap':[True,False]}

rf = GridSearchCV(rfc,param_grid=parameters,n_jobs=-1,cv=10,scoring='accuracy')

rf.fit(x_train,y_train)
rf.best_params_
rfgrid = rf.best_estimator_

rfgrid.fit(x_train,y_train)
train_pred_rf = grid_model.predict(x_train)

val_pred_rf = grid_model.predict(x_val)



print("Accuracy on train is:",accuracy_score(y_train,train_pred_rf))

print("Accuracy on val is:",accuracy_score(y_val,val_pred_rf))