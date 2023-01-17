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
train_base=pd.read_csv('../input/train.csv')
test_base=pd.read_csv('../input/test.csv')
train=train_base.loc[:,["Survived","Pclass","Name","Sex","Age","SibSp","Parch"]].copy()
test=test_base.loc[:,["Pclass","Name","Sex","Age","SibSp","Parch"]].copy()
def clean_data(df,families_zeros):
    df["Title"]=df["Name"].str.split(",").str.get(1).str.split(" ").str.get(1).str.strip()
    df["Has_Title"]=(~df["Title"].isin(["Mr.","Mrs.","Miss."]))*1

    df=pd.concat([df,pd.get_dummies(df["Sex"]).iloc[:,1:].copy()],axis=1)
        
    df=(pd.concat([df,families_zeros],axis=1))
    for index,row in df.iterrows():
        if row["Name"].split(",")[0] in families_zeros.columns.unique():
            df.set_value(index,row["Name"].split(",")[0],1)
    
    df=df.drop(columns=['Name', 'Title',"Sex"])
    if 'Embarked' in df.columns:
        df=df.drop(columns=["Embarked"])

    from sklearn.impute import SimpleImputer
    my_imputer = SimpleImputer()
    df = pd.DataFrame(my_imputer.fit_transform(df),columns=df.columns)

    return df
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



models=pd.DataFrame()
train_clean=clean_data(train,families_zeros=pd.DataFrame(np.zeros((train.shape[0],
                                                    train["Name"].str.split(",").str.get(0).unique().shape[0])
                                                    ),columns=train["Name"].str.split(",").str.get(0).unique()))
X=train_clean.iloc[:,1:].values
Y=train_clean.iloc[:,0].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)

model=LogisticRegression()
model.fit(X,Y)
models["LogisticRegression"]=pd.Series([round(100*accuracy_score(model.predict(X_train),Y_train),1),
                                       round(100*accuracy_score(model.predict(X_test),Y_test),1)],
                                      index=["TRAIN","TEST"])

model=SVC(C=5000)
model.fit(X,Y)
models["SVC"]=pd.Series([round(100*accuracy_score(model.predict(X_train),Y_train),1),
                                       round(100*accuracy_score(model.predict(X_test),Y_test),1)],
                                      index=["TRAIN","TEST"])

model=LinearSVC()
model.fit(X,Y)
models["LinearSVC"]=pd.Series([round(100*accuracy_score(model.predict(X_train),Y_train),1),
                                       round(100*accuracy_score(model.predict(X_test),Y_test),1)],
                                      index=["TRAIN","TEST"])

model=RandomForestClassifier(n_estimators=100)
model.fit(X,Y)
models["RandomForestClassifier"]=pd.Series([round(100*accuracy_score(model.predict(X_train),Y_train),1),
                                       round(100*accuracy_score(model.predict(X_test),Y_test),1)],
                                      index=["TRAIN","TEST"])
models
model=RandomForestClassifier(n_estimators=100)
model.fit(X,Y)
X_valdt=clean_data(test,families_zeros=pd.DataFrame(np.zeros((test.shape[0],
                                                    train["Name"].str.split(",").str.get(0).unique().shape[0])
                                                    ),columns=train["Name"].str.split(",").str.get(0).unique()))
predictions=pd.DataFrame({"PassengerId":test_base["PassengerId"],
                          "Survived":model.predict(X_valdt)})

# train
# predictions
predictions.to_csv('csv_to_submit_v2.csv',index = False)

