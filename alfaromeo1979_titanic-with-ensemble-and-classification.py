# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 500)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
#train_df.head(60)
#Fill missing Age Values by using Titles
for dataset in combine:  
    dataset['Sex'].replace('female',0,inplace=True)
    dataset['Sex'].replace('male',1,inplace=True)
    df=dataset[dataset["Age"].isnull()]
    df.loc[df.Name.str.contains("Mr\."),'Age']=dataset[dataset.Name.str.contains("Mr\.")]["Age"].mean()
    df.loc[df.Name.str.contains("Miss\."),'Age']=dataset[dataset.Name.str.contains("Miss\.")]["Age"].mean()
    df.loc[df.Name.str.contains("Ms\."),'Age']=dataset[dataset.Name.str.contains("Miss\.")]["Age"].mean()
    df.loc[df.Name.str.contains("Mrs\."),'Age']=dataset[dataset.Name.str.contains("Mrs\.")]["Age"].mean()
    df.loc[df.Name.str.contains("Mstr\."),'Age']=dataset[dataset.Name.str.contains("Mstr\.")]["Age"].mean()
    df.loc[df.Name.str.contains("Master\."),'Age']=dataset[dataset.Name.str.contains("Master\.")]["Age"].mean()
    df.loc[df.Name.str.contains("Dr\."),'Age']= dataset[dataset.Name.str.contains("Mr\.")]["Age"].mean()
    dataset.loc[dataset.PassengerId.isin(df.PassengerId), ['Age']] = df[['Age']].values
    dataset["Age"]=dataset.Age.round()
    dataset['Fare'].fillna(dataset.Fare.mean(), inplace=True)
    #Drop fields , these are irrelevant
    dataset.drop(['Ticket','Cabin','Name','Embarked'], axis=1,inplace=True)
    
#One-Hot encoding
train_df=pd.get_dummies(train_df, columns=["Sex"])
test_df=pd.get_dummies(test_df, columns=["Sex"])
train_df.info()
import seaborn as sns
sns.pairplot(train_df , diag_kind = "kde", hue="Survived")
X_train = train_df.drop(["Survived","PassengerId"], axis=1)
Y_train = train_df.pop("Survived")
X_test  = test_df.drop(["PassengerId"], axis=1)
X_train.shape, Y_train.shape, X_test.shape
dcclassifier = tree.DecisionTreeClassifier(max_depth=6)
dcclassifier.fit(X_train , Y_train)

Y_pred = dcclassifier.predict(X_test)
acc_dc = round(dcclassifier.score(X_train, Y_train) * 100, 2)
acc_dc
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.head()
submission.to_csv('submission.csv', index=False)