import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report



train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()



train_df.describe()
test_df.describe()
def get_person(x):
    return 'child' if x.Age < 16 else x.Sex
    
train_df['Person'] = train_df.apply(get_person,axis=1)
test_df['Person']    = test_df.apply(get_person,axis=1)
             
person_dummies_titanic = pd.get_dummies(train_df['Person'].values)
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(test_df['Person'].values)
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

cols = person_dummies_titanic.columns
train_df[cols] = person_dummies_titanic
test_df[cols]    = person_dummies_test

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

seaborn.countplot(x='Person', data=train_df, ax=axis1)

# average of survived for each Person(male, female, or child)
person_perc = train_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
seaborn.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])

train_df.drop(['Person'],axis=1,inplace=True)
test_df.drop(['Person'],axis=1,inplace=True)
# Embarked get dummies

train_df["Embarked"] = train_df["Embarked"].fillna("S")

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

seaborn.countplot(x='Embarked', data=train_df, ax=axis1)
seaborn.countplot(x='Survived', hue="Embarked", data=train_df, order=[1,0], ax=axis2)

embark_perc = train_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
seaborn.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

embark_dummies_train  = pd.get_dummies(train_df['Embarked'].values)
embark_dummies_test  = pd.get_dummies(test_df['Embarked'].values)

cols = embark_dummies_train.columns
train_df[cols] = embark_dummies_train
test_df[cols]  = embark_dummies_test

train_df["Survived"].value_counts()
train_df["Pclass"].value_counts()
train_df["Sex"].value_counts()
train_df["Embarked"].value_counts()

train_df["Age"] = train_df["Age"].fillna(np.mean(train_df["Age"]))
test_df["Age"] = test_df["Age"].fillna(np.mean(test_df["Age"]))

train_df = pd.get_dummies(train_df, prefix="G", columns=["Sex"])
test_df = pd.get_dummies(test_df, prefix="G", columns=["Sex"])

X = train_df.iloc[:, [2, 4, 5, 6, 11, 12]].values
y = train_df.iloc[:,1:2].values

X_test_df = test_df.iloc[:, [1, 3, 4, 5, 10, 11]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_test_df = sc.transform(X_test_df)


classifier_lr = LogisticRegression()
classifier_lr.fit(X_train, y_train)

y_pred_lr = classifier_lr.predict(X_test)

cm_lr = confusion_matrix(y_test, y_pred_lr)
ac_lr = accuracy_score(y_test, y_pred_lr)
print(classification_report(y_test, y_pred_lr))