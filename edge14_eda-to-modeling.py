



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt 



sns.set_style('white')







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#Load data 

df=pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')

df.head()

df_shape=df.shape

df_info=df.info()

df_dtyps=df.dtypes





infos=[df_shape,df_info,df_dtyps]



for dfs in infos :

    dfs



num_cols=df.select_dtypes(['int64']).columns

str_cols=df.select_dtypes(['object']).columns

str_cols
#null values in the data 



null_vals=df.isnull().sum()

null_vals



sns.heatmap(df.isnull())

sns.countplot(df['Survived']) #highly imbalanced Data
# age feature 



age_desc=df['Age'].describe()

sns.distplot(df['Age'],bins=10,hist=True)

age_desc
df['Age_cut']=pd.cut(df['Age'],bins=[-1,30,60,90],labels=[0,1,2])

df['Age_cut'].head()
df.isnull().sum()
sns.countplot(df['Age_cut'],hue=df['Survived'])
for cat in str_cols:

    print(cat ,'\n',df[cat].nunique())
df.loc[

    df["Country"].value_counts()[df["Country"]].values<20,"Country"] = "RARE"
df['Country'].value_counts()
#Handle Categorical Data -> Sex,Class





sex_map={'M':1,'F':0}

cat_map={'P':0,'C':1}





df['Sex_Class']=df['Sex'].map(sex_map)

df['Category_Class']=df['Category'].map(cat_map)





new_feats=['Sex_Class','Category_Class']





for nf in new_feats:

    df[nf].head()

X=pd.get_dummies(df['Sex'],drop_first=True)

df=pd.concat([df,X],axis=1)





X=pd.get_dummies(df['Category'],drop_first=True)

df=pd.concat([df,X],axis=1)

df.head()
#Handle Country Feature 





from sklearn.preprocessing import LabelEncoder,OneHotEncoder





X=pd.get_dummies(df['Country'])

df=pd.concat([df,X],axis=1)

df.head()

df.drop('PassengerId',axis=1,inplace=True)
df.corr()['Survived'].sort_values(ascending=False)
imp_cols=['Age_cut','M','Sweden','RARE','Estonia','Survived']

data=df[imp_cols]
X=data.drop('Survived',axis=1)

Y=data['Survived']
from sklearn.ensemble import RandomForestClassifier

import xgboost

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,classification_report

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression





lreg=LogisticRegression()

lreg.fit(X,Y)







print(cross_val_score(lreg,X,Y,cv=10).mean())













from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=101,test_size=0.3)



logreg=LogisticRegression()

logreg.fit(x_train,y_train)



pred=logreg.predict(x_test)





print(accuracy_score(y_test,pred))

print(classification_report(y_test,pred))

print(confusion_matrix(y_test,pred))












