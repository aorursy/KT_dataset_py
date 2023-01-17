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
import matplotlib.pyplot as plt
import seaborn as sns
data_titanic = pd.read_csv('/kaggle/input/titanic/train.csv')
data_train_titanic = pd.read_csv('/kaggle/input/titanic/train.csv')
data_test_titanic = pd.read_csv('/kaggle/input/titanic/test.csv')
gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
data_train_titanic.head()
data_test_titanic.head()
gender_submission.head()
data_train_titanic.info()
data_test_titanic.info()
data_train_titanic.describe()
data_train_titanic.isnull().sum()
data_test_titanic.isnull().sum()
data_train_titanic["familySize"] = data_train_titanic["SibSp"]+data_train_titanic["Parch"]+1
check_survive = data_train_titanic.drop(["Ticket","Name","Cabin","SibSp","Parch"],axis = 1)
check_survive
pd.crosstab(check_survive["Survived"],check_survive["Sex"])
check_survive.groupby('Sex').Survived.mean()
check_survive.groupby('Pclass').Survived.mean()
check_survive.groupby(['Pclass','Sex']).mean()
check_survive.groupby(['Pclass','Sex']).mean()["Survived"].plot.bar()
def bar_chart(features):
    survived = check_survive[check_survive['Survived']==1][features].value_counts()
    Dead = check_survive[check_survive['Survived']==0][features].value_counts()
    df = pd.DataFrame([survived,Dead])
    df.index = ["survived","Dead"]
    df.plot(kind="bar",stacked=True,figsize=(10,5))
bar_chart("Sex")
bar_chart("Pclass")
bar_chart("Embarked")
bar_chart("familySize")
check_survive.describe(include='all')
check_survive.isnull().sum()
##Replacing age null by age mean
check_survive["Age"].fillna(29.69,inplace=True)
##Replacing embarked null by embarked top
check_survive['Embarked'] = check_survive['Embarked'].fillna('S')
check_survive.isnull().sum()
## Split Feature Sex, Embarked, Pclass
PC1 = pd.get_dummies(check_survive['Sex'],drop_first=True)
PC2 = pd.get_dummies(check_survive['Embarked'],drop_first=True)
PC3 = pd.get_dummies(check_survive['Pclass'],drop_first=True)

## Add feature columns in dataset
check_survive  = pd.concat([check_survive,PC1,PC2,PC3],axis=1)
check_survive  = check_survive.drop(["Sex","Embarked","Pclass"],axis=1)
check_survive
## train test
X_train = check_survive.drop(["Survived"],axis=1)
Y_train = check_survive["Survived"]
X_train
Y_train
data_test_titanic["familySize"] = data_test_titanic["SibSp"]+data_test_titanic["Parch"]+1
check_survive_test = data_test_titanic.drop(["Ticket","Name","Cabin","SibSp","Parch"],axis = 1)
check_survive_test
check_survive_test.info()
check_survive_test.describe()
check_survive_test.isnull().sum()
##Replacing null values by mean
check_survive_test["Age"].fillna(30.27,inplace=True)
check_survive_test["Fare"].fillna(35.62,inplace=True)
check_survive_test.isnull().sum()
## Split Feature Sex, Embarked, Pclass in check_survive_test
PC4 = pd.get_dummies(check_survive_test['Sex'],drop_first=True)
PC5 = pd.get_dummies(check_survive_test['Embarked'],drop_first=True)
PC6 = pd.get_dummies(check_survive_test['Pclass'],drop_first=True)

##Add feature columns in test dataset
check_survive_test  = pd.concat([check_survive_test,PC4,PC5,PC6],axis=1)
check_survive_test  = check_survive_test.drop(["Sex","Embarked","Pclass"],axis=1)
check_survive_test
X_test = check_survive_test
X_test
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
k_fold = KFold(n_splits=5,shuffle=True,random_state=True)
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.naive_bayes import GaussianNB # Naïve Bayes
import tensorflow as tf # Neural Network
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'recall' : make_scorer(recall_score),
           'precision' : make_scorer(precision_score),
           'f1_score' : make_scorer(f1_score)}
clf = DecisionTreeClassifier()

for k, (train_idx, val_idx) in enumerate(k_fold.split(X_train,Y_train)):
    print("Fold : ", k+1)
    k_x_train = X_train.iloc[train_idx]
    k_y_train = Y_train.iloc[train_idx]
    k_x_val = X_train.iloc[val_idx]
    k_y_val = Y_train.iloc[val_idx]
    
    clf.fit(k_x_train, k_y_train)

    pred = clf.predict(k_x_val) #.reshape((X_val.shape[0],1))
#     print(pred)

    print(classification_report(k_y_val, pred))
    
# preds = cross_val_predict(clf, X_train, Y_train, cv=k_fold)
results = cross_validate(clf, X_train, Y_train, cv=k_fold, scoring=scoring)
# print(preds)
print("Average F1-Score : ", results['test_f1_score'].mean())
clf = GaussianNB()

for k, (train_idx, val_idx) in enumerate(k_fold.split(X_train,Y_train)):
    print("Fold : ", k+1)
    k_x_train = X_train.iloc[train_idx]
    k_y_train = Y_train.iloc[train_idx]
    k_x_val = X_train.iloc[val_idx]
    k_y_val = Y_train.iloc[val_idx]
    
    clf.fit(k_x_train, k_y_train)

    pred = clf.predict(k_x_val) #.reshape((X_val.shape[0],1))
#     print(pred)

    print(classification_report(k_y_val, pred))
    
# preds = cross_val_predict(clf, X_train, Y_train, cv=k_fold)
results = cross_validate(clf, X_train, Y_train, cv=k_fold, scoring=scoring)
# print(preds)
print("Average F1-Score : ", results['test_f1_score'].mean())
X_train.shape[1]
in_shape = (X_train.shape[1],)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, input_shape=in_shape, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.01))
model.summary()
f1 = []
for k, (train_idx, val_idx) in enumerate(k_fold.split(X_train,Y_train)):
    print("Fold : ", k+1)
    k_x_train = X_train.iloc[train_idx]
    k_y_train = Y_train.iloc[train_idx]
    k_x_val = X_train.iloc[val_idx]
    k_y_val = Y_train.iloc[val_idx]
    
    model.fit(k_x_train, k_y_train, epochs=100, verbose=0)

    pred = model.predict(k_x_val) #.reshape((X_val.shape[0],1))
    pred = np.round(pred)
#     print(pred)

    print(classification_report(k_y_val, pred))
    f1.append(f1_score(k_y_val, pred))
    
# preds = cross_val_predict(clf, X_train, Y_train, cv=k_fold)
# results = cross_validate(model, X_train, Y_train, cv=k_fold, scoring=scoring)
# print(preds)
print("Average F1-Score : ", sum(f1)/ 5)
