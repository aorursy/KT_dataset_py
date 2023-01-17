# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.model_selection import train_test_split

import xgboost as xgb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


heart=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

corr=heart.corr()

#plt.scatter(x='thal',y='target',c='red',marker='*',data=heart)



sns.countplot('sex',hue='target',data=heart)

sns.countplot('cp',hue='target',data=heart)

sns.countplot('age',hue='target',data=heart)

sns.countplot('chol',hue='target',data=heart)

sns.countplot('fbs',hue='target',data=heart)

sns.countplot('restecg',hue='target',data=heart)

sns.countplot('exang',hue='target',data=heart)

sns.countplot('slope',hue='target',data=heart)

sns.countplot('ca',hue='target',data=heart)

sns.countplot('thal',hue='target',data=heart)

x,y=heart[['sex','cp','trestbps','chol','fbs','restecg','oldpeak','ca','thal']],heart['target']

dataDmatrix=xgb.DMatrix(data=x,label=y)

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

x_cls=xgb.XGBClassifier(objective='binary:logistic',colsample_bytree=0.3,max_depth=5,learning_rate=0.1,alphe=10,n_estimators=100)

x_cls.fit(X_train,y_train)

preds=x_cls.predict(X_test)

acc=accuracy_score(y_test,preds)

print("Accuracy is %f"%(acc*100))

result=confusion_matrix(y_test,preds)

heart_trgt=pd.DataFrame({'target':preds})

heart_trgt.to_csv('/kaggle/working/target.csv')