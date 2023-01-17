import os 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import os 

%matplotlib inline

sns.set(style="darkgrid")
data_path='../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv'

df=pd.read_csv(data_path)

df.shape

df.head()
num_cols = df.select_dtypes('float64').columns

str_cols=df.select_dtypes('object').columns



unique_deg=df['degree_t'].unique()

null_cols=df.isnull().sum() 

sns.heatmap(df.isnull(),cmap="YlGnBu",cbar='False')

plt.show()
df_sal_notplaced=df[df['status']=='Not Placed']

df_sal_notplaced['salary'].unique()



df['salary']=df['salary'].fillna(0)

sns.heatmap(df.isnull(),cmap="YlGnBu",cbar='False')

plt.show()
sns.countplot(df['gender'],hue=df['status'],palette="Set3") 



sns.countplot(df['degree_t'],hue=df['status']) #science and commerce has higher placements  





comm_placed=df[(df['degree_t']=='Comm&Mgmt') & (df['status']=='Placed')]

total_comm=df[df['degree_t']=='Comm&Mgmt']

print('% commerce placed',len(comm_placed)/len(total_comm)) 





sci_placed=df[(df['degree_t']=='Sci&Tech') & (df['status']=='Placed')]

total_sci=df[df['degree_t']=='Sci&Tech']

print('% science placed',len(sci_placed)/len(total_sci)) 



sns.countplot(df['workex'],hue=df['status'])  # work exp plays a role in placements 
a=pd.cut(df['hsc_p'],bins=[1,70,100],labels=[0,1])

b=pd.cut(df['degree_p'],bins=[1,60,100],labels=[0,1])

c=pd.cut(df['ssc_p'],bins=[1,60,100],labels=[0,1])

d=pd.cut(df['mba_p'],bins=[1,62,100],labels=[0,1])

e=pd.cut(df['etest_p'],bins=[1,60,100],labels=[0,1])

li=[a,b,c,d,e]





e_test=df[(df['etest_p']>70) & (df['status']=='Placed')]

e_test_total=df['etest_p']

print('students abv 70 are',len(e_test))

print('% > 70 e test',len(e_test)/len(e_test_total))

sns.countplot(e,hue=df['status'])

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in str_cols :

    df[i]=le.fit_transform(df[i])

    
cor=df.corr()

cor_values =cor['status'].sort_values(ascending=False)

cor_values
df['percent']=(df['ssc_p']+df['hsc_p']+df['degree_p'])/3

cor=df.corr()

cor_values =cor['status'].sort_values(ascending=False)

cor_values
sns.catplot(x="status", y="etest_p", data=df)



sns.catplot(x="status", y="ssc_p", data=df)



sns.catplot(x="status", y="percent", data=df)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score,train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.tree import DecisionTreeClassifier







lr=LogisticRegression()

dtc=DecisionTreeClassifier()

mms=MinMaxScaler()

imp_feature1=['percent','workex','gender']

X_new=df[imp_feature1]

Y=df['status']









x_train,x_test,y_train,y_test=train_test_split(X_new,Y,test_size=0.3,random_state=101)

lr.fit(x_train,y_train)



pred=lr.predict(x_test)



print(accuracy_score(y_test,pred))

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))




