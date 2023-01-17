%matplotlib inline

import os

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split



from sklearn.metrics import accuracy_score
#data_path = r"C:\Users\Reljod\Desktop\Study Materials\kaggle\dataset\titanic"
train_path = "../input/train.csv"

test_path = "../input/test.csv"

submission_path = "../input/gender_submission.csv"
train_raw = pd.read_csv(train_path)

test_raw = pd.read_csv(test_path)
train_raw.head()
test_raw.head()
train_raw.shape
test_raw.shape
# Make a copy, it is always advisable to save a copy of the raw train data

df_train = train_raw.copy()

df_test = test_raw.copy()
dfs = [df_train, df_test]
df_train.columns
for df in dfs:    

    df.set_index('PassengerId', inplace=True)
df_train.head()
df_test.head()
# Setting the plotting style into seaborn style

sns.set()
target = df_train["Survived"]

x_surv = target.value_counts().keys()

y_surv = target.value_counts()

_ = plt.bar(x_surv, y_surv, tick_label=["Dead","Alive"])

_ = plt.title("Passengers' state", {'fontsize': 15})

_ = plt.ylabel("Count", {'fontsize': 15}, labelpad=20)

_ = plt.show()
pclass = df_train["Pclass"]

x_pclass = pclass.value_counts().keys()

y_pclass = pclass.value_counts()

_ = plt.bar(x_pclass, y_pclass, tick_label=['3: Lower', '1: Upper', '2: Middle',])

_ = plt.title("Ticket Class", {'fontsize': 15})

_ = plt.ylabel("Count", {'fontsize': 15}, labelpad=20)

_ = plt.show()
df_train.corr()
df_train.mean()
df_train.mode(numeric_only=True).transpose()
df_train.std()
df_train.var()
datainfo = df_train.info()

print(datainfo)
for df in dfs:

    df.drop(["Cabin"], axis=1, inplace=True)
## Checking if the cabin column is gone...

df_train.head()
print(df_train.Age.describe())

### Looking at the description, we can see that the there's a variety of age groups looking at the std, there's an infant, 

### a senior but most of them lies between 20 to 38 years old age group. The median and the mean also doesn't differ much

### at 28.5.
for df in dfs:

    df["Age"].fillna(df.Age.median(), inplace=True)
df_train.info()
print(df_train["Embarked"].value_counts())

print(df_train["Embarked"].unique())
dfs[1][pd.isnull(dfs[1]).any(axis=1)]
### Removing the rows with nan-values

for i, df in enumerate(dfs):    

    if i == 0:

        df.dropna(inplace=True)

    if i == 1:

        df.fillna(df["Fare"].median(), inplace=True)
dfs[1].info()
## Checking again, it seems there is no null values anymore!

df_train.info()
df_train_obj = df_train.select_dtypes(['object']).copy()

df_test_obj = df_test.select_dtypes(['object']).copy()
df_objects = [df_train_obj, df_test_obj]
for df_obj in df_objects:    

    df_obj.drop(["Name", "Ticket"], axis=1, inplace=True)
df_train_obj.tail()
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder
df_ohes = []

sex_cols = []
ohe = OneHotEncoder(categories='auto')

for df_obj in df_objects:

    df_ohes.append(ohe.fit_transform(df_obj["Sex"].values.reshape(-1,1)))

    sex_cols.append(list(ohe.categories_[0]))
df_train_ohe_sex = pd.DataFrame(df_ohes[0].toarray(), columns=sex_cols[0], dtype=np.int)

df_test_ohe_sex = pd.DataFrame(df_ohes[1].toarray(), columns=sex_cols[1], dtype=np.int)
df_train_ohe_sex.set_index(df_train.index, inplace=True)

df_test_ohe_sex.set_index(df_test.index, inplace=True)
df_train_ohe_sex.head()
df_train_ohe_sex.shape
df_test_ohe_sex.head()
ohe_emb = []

emb_cols = []
for df_obj in df_objects:

    ohe_emb.append(ohe.fit_transform(df_obj["Embarked"].values.reshape(-1,1)))

    emb_cols.append(list(ohe.categories_[0]))
df_train_ohe_emb = pd.DataFrame(ohe_emb[0].toarray(), columns=emb_cols[0], dtype=np.int)

df_test_ohe_emb = pd.DataFrame(ohe_emb[1].toarray(), columns=emb_cols[1], dtype=np.int)
df_train_ohe_emb.set_index(df_train.index, inplace=True)

df_test_ohe_emb.set_index(df_test.index, inplace=True)
df_train_ohe_emb.head()
df_test_ohe_emb.head()
### Just creating a copy

df1 = df_train.copy()

df_test1 = df_test.copy()

dfs1 = [df1, df_test1]
target = df1.pop("Survived")
df1.head()
### Dropping some columns

for dfx in dfs1:

    dfx.drop(["Name", "Sex", "Ticket", "Embarked"], axis=1, inplace=True)
### Adding the one hot encoded columns

df_train_ohe = pd.concat([df1, df_train_ohe_sex, df_train_ohe_emb], axis=1)

df_test_ohe = pd.concat([df_test1, df_test_ohe_sex, df_test_ohe_emb], axis=1)
df_ohe = [df_train_ohe, df_test_ohe]
df_train_ohe.head(10)
df_train_ohe.info()
df_test_ohe.head(10)
df_train_ohe.Pclass.value_counts()
df_test_ohe.Pclass.value_counts()
ohe_pclass = []

pclass_cols = []
ohe = OneHotEncoder(categories="auto")

for df_x in df_ohe:

    ohe_pclass.append(ohe.fit_transform(df_x.Pclass.values.reshape(-1,1)))

    #pclass_cols.append(list(ohe.categories_[0]))
pclass_cols = list(ohe.categories_[0])

#print(pclass_cols)

pclass_names = ["upper", "middle", "lower"]

pclass_dict = dict(zip(pclass_names, pclass_cols))

print(pclass_dict)

print(pclass_dict.keys())
df_train_pclass_ohe = pd.DataFrame(ohe_pclass[0].toarray(), index=df_train.index, columns=pclass_dict.keys(), dtype=np.int)

df_test_pclass_ohe = pd.DataFrame(ohe_pclass[1].toarray(), index=df_test.index, columns=pclass_dict.keys(), dtype=np.int)

df_train_pclass_ohe.head()
df_test_pclass_ohe.head()
df_train_ohe.SibSp.value_counts()
df_train_ohe.Parch.value_counts()
df_train_ohe.Parch.head()
new_SibSp = []

new_Parch = []
for df_x in df_ohe:

    new_SibSp.append(df_x.SibSp.apply(lambda x: 'NoSibSp' if x==0 else 'HasSibSp'))
for df_x in df_ohe:

    new_Parch.append(df_x.Parch.apply(lambda x: 'NoParch' if x==0 else 'HasParch'))
print(new_SibSp[0].value_counts())

print(new_SibSp[1].value_counts())
print(new_Parch[0].value_counts())

print(new_Parch[1].value_counts())
ohe_train_sibsp = ohe.fit_transform(new_SibSp[0].values.reshape(-1,1))

ohe_test_sibsp = ohe.fit_transform(new_SibSp[1].values.reshape(-1,1))
sibsp_cols = list(ohe.categories_[0])

print(sibsp_cols)
ohe_train_parch = ohe.fit_transform(new_Parch[0].values.reshape(-1,1))

ohe_test_parch = ohe.fit_transform(new_Parch[1].values.reshape(-1,1))

ohe.categories_
parch_cols = list(ohe.categories_[0])

print(parch_cols)
df_train_sibsp_ohe = pd.DataFrame(ohe_train_sibsp.toarray(), columns=sibsp_cols, index=df_train.index, dtype=np.int)

df_test_sibsp_ohe = pd.DataFrame(ohe_test_sibsp.toarray(), columns=sibsp_cols, index=df_test.index, dtype=np.int)

df_train_sibsp_ohe.head(10)
df_test_sibsp_ohe.head(10)
df_train_parch_ohe = pd.DataFrame(ohe_train_parch.toarray(), columns=parch_cols, index=df_train.index, dtype=np.int)

df_test_parch_ohe = pd.DataFrame(ohe_test_parch.toarray(), columns=parch_cols, index=df_test.index, dtype=np.int)

df_train_parch_ohe.head(10)
df_test_parch_ohe.head(10)
#for df_x in df_ohe:

#    df_x.drop(["Pclass"], axis=1, inplace=True)
df_train_num = pd.concat([df_ohe[0], df_train_pclass_ohe, df_train_sibsp_ohe, df_train_parch_ohe], axis=1)

df_test_num = pd.concat([df_ohe[1], df_test_pclass_ohe, df_test_sibsp_ohe, df_test_parch_ohe], axis=1)
df_train_num.values
df_test = df_test_num.copy()
X_train, X_test, y_train, y_test = train_test_split(df_train_num, target, test_size=0.3, random_state=10)
X_train.head()
print("Train data: ", X_train.shape)

print("Train label: ", y_train.shape)

print("Test data: ", X_test.shape)

print("Test label: ", y_test.shape)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200,

                            max_depth=4,

                            random_state=10, min_samples_leaf=1, n_jobs=-1)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
y_pred_train = rfc.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_train)

print("-------------------------------------------------------------")

print("The Accuracy of the model in the Training Data is {:.3f}%".format(acc_train*100))

print("-------------------------------------------------------------")
acc_test = accuracy_score(y_test, y_pred)

print("-------------------------------------------------------------")

print("The Accuracy of the model in the Training Data is {:.3f}%".format(acc_test*100))

print("-------------------------------------------------------------")
test_prediction = rfc.predict(df_test)
test_prediction.shape
df_submission = pd.read_csv(submission_path, index_col="PassengerId")
print(len(df_submission))

print(len(df_submission)==len(test_prediction)) #equality
acc = accuracy_score(df_submission["Survived"], test_prediction)

print("The Accuracy of the model in the Unseen Test Data is {:.3f}%".format(acc*100))
rfc_submission = pd.DataFrame({'Survived':test_prediction})
rfc_submission.set_index(df_test.index, inplace=True)
rfc_submission.shape
rfc_submission.to_csv("RandomForestClassifer_submission.csv")