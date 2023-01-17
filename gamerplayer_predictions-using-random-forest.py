# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
files={}
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        files[filename] = pd.read_csv(os.path.join(dirname, filename))
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=files['train.csv']
train_len=df.shape[0]
df=pd.concat([df,files['test.csv']],axis=0).reset_index()
columns=df.columns
print(columns)
df.isnull().sum()
#First Lets See embarked
df[df['Embarked'].isnull()]
df['Embarked'].unique()
df['Embarked'].fillna('Not',inplace=True)
df['Embarked']=df['Embarked'].astype('category').cat.as_ordered()
df['Embarked'].dtype
df['Embarked'].isnull().sum()
df['Cabin'].unique()
df['Null Cabin'] = False
df['Null Cabin'].loc[df['Cabin'].isnull()]=True
df['Cabin'].fillna('No Cabin',inplace=True)
df['Block']=df['Cabin'].str[0]
df['floor']=df['Cabin'].str[1]
df['floor'].fillna(0,inplace=True)

df
df.isnull().sum()
bins=[0,15,25,35,45,55,65,100]
pd.cut(df['Age'],bins,labels=['Child',"Teen","Young","early thirties","old experienced","old","really old"]).isnull().sum()
df['Age_Bin'] = pd.cut(df['Age'],bins,labels=['Child',"Teen","Young","early thirties","old experienced","old","really old"]).cat.codes
df['Age_Bin']
fare_bin=[0,30,60,100,200,300,600]
df['Fare'] = pd.cut(df['Fare'],bins)
df.info()
cat_cols=list(df.columns[6:-1])
cat_cols.append('Sex')
cat_cols.append('Pclass')
cat_cols.append('Age_Bin')
cat_cols
X=df[cat_cols]
X.info()
X=X.drop(['Ticket','Age'],axis=1)

cat_cols=X.columns
cat_cols
for c in cat_cols:
    X[c]=X[c].astype('category').cat.as_ordered().cat.codes
X.info()
from sklearn.ensemble import RandomForestClassifier
X_train=X.head(891)
Y=df['Survived']
Y_train=Y.head(files['train.csv'].shape[0])
X_train.corrwith(Y_train)

X_pred=X.tail(files['test.csv'].shape[0])

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_train,Y_train,test_size=0.33
                                              )
model=RandomForestClassifier(n_jobs=-1)
model.fit(X_train,Y_train)
model.score(X_train,Y_train)
model.score(X_test,Y_test)
preds=model.predict(X_pred)
preds
d={
    'PassengerId':df.tail(files['test.csv'].shape[0])['PassengerId'],
    'Survived': preds
}
pred_df=pd.DataFrame(d,columns=['PassengerId','Survived'])
pred_df['Survived']=pred_df['Survived'].astype(int)
pred_df

pred_df.to_csv('output.csv',index=False)
