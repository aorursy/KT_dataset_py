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
import seaborn as sns
%matplotlib inline
train_df=pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")
train_df.info()
sns.countplot(train_df["Sex"]);
sns.countplot(train_df["Sex"],hue=train_df["Pclass"]);
sns.countplot(train_df["Pclass"],hue=train_df["Sex"]);
train_df["Age"].hist(bins=70);
train_df["Age"].mean()
sns.countplot(train_df["Embarked"],hue=train_df["Pclass"]);
sns.factorplot("Embarked","Survived",data=train_df);
sns.factorplot('Pclass','Survived',data=train_df, order=[1,2,3]);
sns.factorplot("Pclass","Survived",hue="Sex",data=train_df,order=[1,2,3])
sns.lmplot("Age","Survived",data=train_df);
sns.lmplot("Survived","Age",hue="Pclass",data=train_df);
sns.lmplot("Age","Survived",hue="Sex",data=train_df);
train_df["Family"]=train_df["SibSp"]+train_df["Parch"]+1
sns.countplot("Family",data=train_df);
sns.factorplot("Survived","Family",data=train_df);
train_df["Embarked"]=train_df["Embarked"].replace("S",0).replace("Q",2).replace("C",1)
train_df["Sex"]=train_df["Sex"].replace("male",0).replace("female",1)
train_df["Age"].fillna(train_df.Age.median(),inplace=True)
train_df["Embarked"].fillna(train_df.Embarked.median(),inplace=True)
train_df["Family"]=train_df["SibSp"]+train_df["Parch"]+1
del train_df["SibSp"]
del train_df["Parch"]
test_df["Embarked"]=test_df["Embarked"].replace("S",0).replace("Q",2).replace("C",1)
test_df["Sex"]=test_df["Sex"].replace("male",0).replace("female",1)
test_df["Age"].fillna(test_df.Age.median(),inplace=True)
test_df["Fare"].fillna(test_df.Fare.median(),inplace=True)
test_df["Family"]=test_df["SibSp"]+test_df["Parch"]+1
del test_df["SibSp"]
del test_df["Parch"]
del train_df["Cabin"]
del test_df["Cabin"]
del train_df["Ticket"]
del test_df["Ticket"]
del train_df["Name"]
del test_df["Name"]
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scalerall_train_df=scaler.fit_transform(train_df[["Pclass","Sex","Age","Fare","Embarked","Family"]])
scalerall_train_df=pd.DataFrame(scalerall_train_df)
scalerall_train_df=scalerall_train_df.rename(columns={0:"Pclass",1:"Sex",2:"Age",3:"Fare",4:"Embarked",5:"Family"})

scalerall_test_df=scaler.transform(test_df[["Pclass","Sex","Age","Fare","Embarked","Family"]])
scalerall_test_df=pd.DataFrame(scalerall_test_df)
scalerall_test_df=scalerall_test_df.rename(columns={0:"Pclass",1:"Sex",2:"Age",3:"Fare",4:"Embarked",5:"Family"})
X_train=scalerall_train_df
y_train=train_df.iloc[:,1]
X_test=scalerall_test_df
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
X_train.shape,y_train.shape,X_test.shape
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

gbc_model = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=50,
              presort='auto', random_state=0, subsample=1.0, verbose=0,
              warm_start=False)

gbc_model.fit(X_train, y_train);
kfold = KFold(n_splits=10,random_state=0)
scores = cross_val_score(gbc_model, X_train, y_train,cv=kfold,scoring="accuracy")
print(gbc_model.score(X_train,y_train)),print(scores.mean());
pred = gbc_model.predict(X_test)
submit = pd.DataFrame()
imageid = []
for i in range(len(pred)):
    imageid.append(i+892)
submit["PassengerId"] = imageid
submit["Survived"] = pred
submit.to_csv("result25.csv", index=False)