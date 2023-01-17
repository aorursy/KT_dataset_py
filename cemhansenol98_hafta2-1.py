import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestClassifier

from pdpbox import pdp

from plotnine import *

from sklearn.metrics import confusion_matrix





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#read data

data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

data_test = pd.read_csv("/kaggle/input/titanic/test.csv")
data_train.sample(5)
data_test.sample(5)

data_train.info()

print("---------------------------------")

data_test.info()
data_train.columns
data_test.columns

data_train.describe(include="all")
data_test.describe(include="all")
#missing values

print(pd.isnull(data_train).sum())

print("-------------------------")

print(pd.isnull(data_test).sum())
# train survived count

survived = data_train.Survived

plt.figure(figsize=(7,5))

sns.countplot(survived)

plt.title("Survived",color='blue',fontsize=15)

plt.show()
passanger_class = data_train.Pclass

plt.figure(figsize=(7,5))

sns.countplot(passanger_class)

plt.title("data_train Passanger Class",color = 'blue',fontsize=15)

plt.show()
passanger_class = data_test.Pclass

plt.figure(figsize=(7,5))

sns.countplot(passanger_class)

plt.title("data_test Passanger Class",color = 'blue',fontsize=15)

plt.show()
data_train['Title'] = data_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



data_train['Title'] = data_train['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

data_train['Title'] = data_train['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

data_train['Title'] = data_train['Title'].replace('Mlle', 'Miss')

data_train['Title'] = data_train['Title'].replace('Ms', 'Miss')

data_train['Title'] = data_train['Title'].replace('Mme', 'Mrs')



passanger_name = data_train.Title

plt.figure(figsize=(10,7))

sns.countplot(passanger_name)

plt.title("data_train Passanger Name",color = 'blue',fontsize=15)

plt.show()
data_test['Title'] = data_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



data_test['Title'] = data_test['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

data_test['Title'] = data_test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

data_test['Title'] = data_test['Title'].replace('Mlle', 'Miss')

data_test['Title'] = data_test['Title'].replace('Ms', 'Miss')

data_test['Title'] = data_test['Title'].replace('Mme', 'Mrs')



passanger_name = data_test.Title

plt.figure(figsize=(10,7))

sns.countplot(passanger_name)

plt.title("data_test Passanger Name",color = 'blue',fontsize=15)

plt.show()

gender = data_train.Sex

plt.figure(figsize=(7,5))

sns.countplot(gender)

plt.title("data_train Gender",color = 'blue',fontsize=15)

plt.show()
gender = data_test.Sex

plt.figure(figsize=(7,5))

sns.countplot(gender)

plt.title("data_test Gender",color = 'blue',fontsize=15)

plt.show()

data_train['AgeGroup'] = ["Baby" if (i>=0 and i<5) else "Child" if (i>=5 and i<12) else "Teenager" if (i>=12 and i<18) 

                          else "Student" if(i>=18 and i<24) else "Young Adult" if(i>=24 and i<35) 

                          else "Adult" if(i>=35 and i<60) else "Senior" if(i>=60) else "Unknown" 

                          for i in data_train.Age ]



passanger_ageGroup = data_train.AgeGroup

plt.figure(figsize=(10,7))

sns.countplot(passanger_ageGroup)

plt.title("data_train Passanger AgeGroup",color = 'blue',fontsize=15)

plt.show()
data_test['AgeGroup'] = ["Baby" if (i>=0 and i<5) else "Child" if (i>=5 and i<12) else "Teenager" if (i>=12 and i<18) 

                          else "Student" if(i>=18 and i<24) else "Young Adult" if(i>=24 and i<35) 

                          else "Adult" if(i>=35 and i<60) else "Senior" if(i>=60) else "Unknown" 

                          for i in data_test.Age ]



passanger_ageGroup = data_test.AgeGroup

plt.figure(figsize=(10,7))

sns.countplot(passanger_ageGroup)

plt.title("data_test Passanger AgeGroup",color = 'blue',fontsize=15)

plt.show()
passanger_sibsp = data_train.SibSp

plt.figure(figsize=(10,7))

sns.countplot(passanger_sibsp)

plt.title("data_train Passanger SibSp")

plt.show()
passanger_sibsp = data_test.SibSp

plt.figure(figsize=(10,7))

sns.countplot(passanger_sibsp)

plt.title("data_test Passanger SibSp")

plt.show()
passanger_parch = data_train.Parch

plt.figure(figsize=(10,7))

sns.countplot(passanger_parch)

plt.title("data_train Passanger Parch")

plt.show()
passanger_parch = data_test.Parch

plt.figure(figsize=(10,7))

sns.countplot(passanger_parch)

plt.title("data_test Passanger Parch")

plt.show()
data_train.Fare.describe()
passanger_fare = ['above100$' if i>=100 else '32between100$' if (i<100 and i>=32) else 'Free' if i==0 else 'below32$' for i in data_train.Fare]

plt.figure(figsize=(10,7))

sns.countplot(passanger_fare)

plt.title("data_train Passanger Fare",color = 'blue',fontsize=15)

plt.show()
data_test.Fare.describe()
passanger_fare_test = ['above100$' if i>=100 else '35between100$' if (i<100 and i>=35) else 'Free' if i==0 else 'below35$' for i in data_test.Fare]

plt.figure(figsize=(10,7))

sns.countplot(passanger_fare_test)

plt.title("data_test Passanger Fare",color = 'blue',fontsize=15)

plt.show()
passanger_embarked = data_train.Embarked

plt.figure(figsize=(10,7))

sns.countplot(passanger_embarked)

plt.title("data_train Passanger Embarked",color = 'blue',fontsize=15)

plt.show()
passanger_embarked = data_test.Embarked

plt.figure(figsize=(10,7))

sns.countplot(passanger_embarked)

plt.title("data_test Passanger Embarked",color = 'blue',fontsize=15)

plt.show()
data_train.head()
data_test.head()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

data_train['Title'] = data_train['Title'].map(title_mapping)

data_train['Title'] = data_train['Title'].fillna(0)



data_test['Title'] = data_test['Title'].map(title_mapping)

data_test['Title'] = data_test['Title'].fillna(0)



#data_test.Title.head()

#data_train.Title.head()
data_train.Sex = [0 if i=="male" else 1 for i in data_train.Sex]

data_test.Sex = [0 if i=="male" else 1 for i in data_test.Sex]

data_test.Sex.head()

data_train.Sex.head()
data_train['Age'] = data_train['Age'].fillna(0)

data_test['Age'] = data_test['Age'].fillna(0)

print("Missing train age value count:",pd.isnull(data_test.Age).sum())

print("Missing test age value count:",pd.isnull(data_train.Age).sum())
title_mapping_age = {"Baby":1, "Child":2, "Teenager":3, "Student":4, "Young Adult":5, "Adult":6, "Senior":7, "Unknow":0}

data_train['AgeGroup'] = data_train['AgeGroup'].map(title_mapping_age)

data_train['AgeGroup'] = data_train['AgeGroup'].fillna(0)

data_test['AgeGroup'] = data_test['AgeGroup'].map(title_mapping_age)

data_test['AgeGroup'] = data_test['AgeGroup'].fillna(0)

#data_test.AgeGroup.head()

#data_train.AgeGroup.head()
#train

data_train['FamilySize'] = data_train['SibSp'] + data_train['Parch']

data_train['IsAlone'] = [0 if i==0 else 1 for i in data_train['FamilySize']]# 0 equals alone 1 equals family

data_train["CabinBool"] = (data_train["Cabin"].notnull().astype('int'))

data_train['FareBand'] = [4 if i=='above100$' else 3 if i=='32between100$' else 2 if i=='Free' else 1 for i in passanger_fare]

data_train.Embarked = [0 if i=="S" else 1 if i=="C" else 2 if i=="Q" else 0 for i in data_train.Embarked]

data_train['Embarked'] = data_train['Embarked'].fillna(0)

print(pd.isnull(data_train.Embarked).sum())



#test

data_test['FamilySize'] = data_test['SibSp'] + data_test['Parch']

data_test['IsAlone'] = [0 if i==0 else 1 for i in data_test['FamilySize']]# 0 equals alone 1 equals family

data_test["CabinBool"] = (data_test["Cabin"].notnull().astype('int'))

data_test['FareBand'] = [4 if i=='above100$' else 3 if i=='35between100$' else 2 if i=='Free' else 1 for i in passanger_fare_test]



data_test.Embarked = [0 if i=="S" else 1 if i=="C" else 2 if i=="Q" else 0 for i in data_test.Embarked]

print(pd.isnull(data_test.Embarked).sum())



data_train.head()
data_train_x = data_train.drop(['PassengerId','Survived','Name','Cabin','SibSp','Parch','Age','Fare','Ticket'],axis=1)

data_train_y = data_train.Survived

data_train_x.head()
data_test_x = data_test.drop(['PassengerId','Name','Cabin','SibSp','Parch','Age','Fare','Ticket'],axis=1)

data_test_x.head()
#normalization

data_train_x = (data_train_x - np.min(data_train_x))/(np.max(data_train_x)-np.min(data_train_x)).values

data_train_x.head()
data_test_x = (data_test_x - np.min(data_test_x))/(np.max(data_test_x)-np.min(data_test_x)).values

data_test_x.head()
X_train, X_test, y_train, y_test = train_test_split(data_train_x,data_train_y,test_size=0.2,random_state=42)
rf = RandomForestClassifier(n_estimators = 200,random_state = 42)

rf.fit(X_train,y_train)

print(rf.score(X_test,y_test))
print("Train Score : ", rf.score(X_train,y_train))

print("Test Score  : ", rf.score(X_test,y_test))
#Estimated number of survivors

y_pred = rf.predict(X_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)



#cm visualization

f, ax = plt.subplots(figsize =(10,10))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
def plot_pdp(feat, clusters=None, feat_name=None):

    feat_name = feat_name or feat

    p = pdp.pdp_isolate(rf, X_train, X_train.columns, feat)

    return pdp.pdp_plot(p, feat_name, plot_lines=True,

                        cluster=clusters is not None,

                        n_cluster_centers=clusters)
plot_pdp('Embarked')
plot_pdp('FamilySize')
plot_pdp('Pclass')
df_ext = X_train.copy()

df_ext['is_valid'] = 1
df_ext.is_valid[:600]=0
df_ext.head()
df_ext.is_valid.value_counts()
X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(df_ext, y_train, test_size=0.33)
num_feats = ['Pclass', 'Sex', 'Embarked', 'Title', 'AgeGroup', 'FamilySize',

       'IsAlone', 'CabinBool', 'FareBand']
X_df_ext = df_ext[num_feats]

y_df_ext = df_ext["is_valid"]

X_df_ext.head()
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_df_ext, y_df_ext);

m.oob_score_
def rf_feat_importance(m, df):

    return pd.DataFrame({'columns':df.columns, 'importance':m.feature_importances_}

                       ).sort_values('importance', ascending=False)
fi= rf_feat_importance(m,X_df_ext); fi[:4]
X_train_1.drop(['AgeGroup'], axis=1, inplace=True)

X_val_1.drop(['AgeGroup'], axis=1, inplace=True)
t = RandomForestClassifier(n_estimators=100, n_jobs=-1,max_depth=15,bootstrap=True,random_state=42)

t.fit(X_train_1, y_train_1)

y_val_pred = t.predict(X_val_1)
print(t.score(X_train_1,y_train_1))

print(t.score(X_val_1,y_val_1))