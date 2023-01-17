# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.io as pio

import missingno as msno

from scipy import stats

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

df.head()
df.info()
n = msno.bar(df,color='coral')
plt.style.use('classic')

plt.rcParams['figure.figsize'] = (8,6)

sns.countplot(x = 'anaemia',data=df,hue = 'DEATH_EVENT',color='yellow')

plt.show()
plt.rcParams['figure.figsize'] = (8,6)

sns.countplot(x = 'high_blood_pressure',data=df,hue = 'DEATH_EVENT',color='yellow')

plt.show()
plt.figure(figsize=(8,6))

fig,ax = plt.subplots(2,2,figsize=(8,6))

sns.distplot(df['age'], fit = stats.norm,color='coral',ax=ax[0][0])

sns.distplot(df['ejection_fraction'], fit = stats.norm,color='coral',ax=ax[0][1])

sns.distplot(df['serum_sodium'],fit = stats.norm,color='coral',ax=ax[1][0])

sns.distplot(df['platelets'], fit = stats.norm,color='coral',ax=ax[1][1])

plt.tight_layout()

plt.xticks(rotation=90)

plt.show()


facet = sns.FacetGrid(df,hue="DEATH_EVENT",aspect = 4)

facet.map(sns.kdeplot,"age",shade = True)

facet.set(xlim = (0,df["age"].max()))

facet.add_legend()

plt.show()
plt.rcParams['figure.figsize']=(10,8)

plt.style.use("classic")

labels=['Survived','Not Survived']

color = ['yellowgreen','gold','lightskyblue','coral']

plat_gr = df.groupby("DEATH_EVENT")['platelets'].sum().reset_index()

plat_gr.plot.pie(y = 'platelets',colors=color,explode=(0,0.02),shadow=True,autopct = '%0.1f%%')

plt.legend(labels,loc='best')

plt.axis('on');
plt.rcParams['figure.figsize'] =(9,8)

sns.catplot(x="sex", hue="smoking", col="anaemia",

                data=df, kind="count",

                height=6,aspect=.7,palette='Set3')

plt.show()
plt.figure(figsize=(8,6))

sns.jointplot(x = 'age', y = 'platelets',data= df,kind = 'kde',color='coral')

plt.show()
plt.figure(figsize=(8,6))

fig,ax = plt.subplots(2,3,figsize=(10,8))

sns.regplot(x = 'age', y = 'platelets',data= df,color='coral',ax=ax[0][0])

sns.regplot(x = 'age', y = 'serum_sodium',data= df,color='coral',ax=ax[0][1])

sns.regplot(x = 'age', y = 'creatinine_phosphokinase',data= df,color='coral',ax=ax[0][2])

sns.countplot(x='sex',hue = 'DEATH_EVENT',color='gold', data= df,ax=ax[1][0])

sns.countplot(x='smoking',hue = 'DEATH_EVENT',color='gold',data= df,ax=ax[1][1])

sns.countplot(x='diabetes',hue = 'DEATH_EVENT',color='gold', data= df,ax=ax[1][2])

plt.tight_layout()

plt.show()
x = df.drop('DEATH_EVENT',axis=1)

y = df['DEATH_EVENT']
from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier()

clf.fit(x,y)

print(clf.feature_importances_)
plt.rcParams['figure.figsize'] = (8,5)

feature_importance = pd.Series(clf.feature_importances_,index = x.columns)

feature_importance.nlargest(6).plot(kind='barh',color='gold')
x_sel = df[['time','ejection_fraction','serum_creatinine','age','serum_sodium','platelets']]

y_sel = df['DEATH_EVENT']
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.svm import SVC
x_train1,x_test1,y_train1,y_test1 = train_test_split(x_sel,y_sel,test_size=0.2,random_state=2)
xgb_sel = XGBClassifier(max_depth = 2,n_estimators=100,random_state=0)

xgb_sel.fit(x_train1,y_train1)
from sklearn.metrics import accuracy_score

y_pred_sel =xgb_sel.predict(x_test1)

accuracy_score(y_test1,y_pred_sel)
svm_sel = SVC(kernel = 'linear')

svm_sel.fit(x_train1,y_train1)
y_pred_sel = svm_sel.predict(x_test1)

accuracy_score(y_test1,y_pred_sel)
dt_sel = DecisionTreeClassifier(criterion='entropy',max_depth=12,random_state=18)#11,random_state=18

dt_sel.fit(x_train1,y_train1)
y_pred_sel = dt_sel.predict(x_test1)

accuracy_score(y_test1,y_pred_sel)
lgb_sel = LGBMClassifier(max_depth = 2,random_state=2)

lgb_sel.fit(x_train1,y_train1)
y_pred_sel = lgb_sel.predict(x_test1)

accuracy_score(y_test1,y_pred_sel)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
xgb = XGBClassifier(max_depth = 2,n_estimators=100,random_state=0)

xgb.fit(x_train,y_train)
y_pred_xg =xgb.predict(x_test)
accuracy_score(y_test,y_pred_xg)
lgb = LGBMClassifier(max_depth = 2,random_state=2)

lgb.fit(x_train,y_train)
y_pred_lgb =lgb.predict(x_test)

accuracy_score(y_test,y_pred_lgb)
dt = DecisionTreeClassifier(criterion='entropy',max_depth=12,random_state=0)

dt.fit(x_train,y_train)
y_pred_dt = dt.predict(x_test)

accuracy_score(y_test,y_pred_dt)
result = pd.DataFrame({"Feature Selection" : [0.85,0.91666,0.9], "Without Feature Selection" : [0.9,0.9,0.91666]})

result.set_index([pd.Index(['XGBoost','DecisionTree','LightGBM'])],inplace=True)

result.style.background_gradient(cmap = "PuRd")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense,Dropout,BatchNormalization

from tensorflow.keras.optimizers import Adam
model = Sequential()

model.add(Dense(x.shape[1],activation='relu',input_dim=x.shape[1]))

model.add(Dense(32,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(1,activation ='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=30,validation_split=0.2,verbose=1)