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
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

display(df_train.head())
import re
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


#print(f"mean: {mean}, std : {std}, min:{min_age}")
#df_train["Fare"] = np.log1p(df_train["Fare"])


display(df_train[df_train["Age"] < 1])
display(df_test[df_test["Age"] < 1])

def find_salutation(name):
    pattern = re.compile("([\w']+[.,!?;])\W")
    name_parts = pattern.findall(name)
    if name_parts != None:
        for s in name_parts:
            if s.endswith("."):
                return s
        
df_train["Salutation"] = df_train.Name.apply(find_salutation)
df_test["Salutation"] = df_test.Name.apply(find_salutation)
print(df_train["Salutation"].unique())
print(df_test["Salutation"].unique())
print(df_train["Ticket"].value_counts().sort_values(ascending=False)[:10])

display(df_train[(df_train["Age"]> 40) & (df_train["Sex"] == 'male') & (df_train["Survived"] == 1)])
display(df_test[(df_test["Fare"].isna())])
#df_train[df_train.Age < 17][df.Cabin.isna()]

# drop the rows if the passenger has not embarked as the test set does not have that combination
#df_train[df_train.Embarked.isna()]
#df_test[df_test.Embarked.isna()]

df_train_copy = df_train.copy()
df_test_copy = df_test.copy()

avg_fare_by_class = df_train_copy.groupby(["Pclass"])[["Pclass", "Fare"]].median().to_dict()["Fare"]
for pclass in avg_fare_by_class:
    df_train_copy.loc[df_train_copy.Fare.isna(), "Fare"] = avg_fare_by_class[pclass]

avg_fare_by_class = df_test_copy.groupby(["Pclass"])[["Pclass", "Fare"]].median().to_dict()["Fare"]
for pclass in avg_fare_by_class:
    df_test_copy.loc[df_test_copy.Fare.isna(), "Fare"] = avg_fare_by_class[pclass]

df_train_copy.loc[df_train_copy.Salutation == "Ms.", "Salutation"] = "Mrs."

print("salutations", df_train_copy.Salutation.unique(),df_test_copy.Salutation.unique() )

salutations_null_age = df_train_copy[df_train_copy["Age"].isna()]["Salutation"].unique()
print("train", salutations_null_age)
avg_age_by_salutation = df_train_copy[df_train_copy["Age"].notna()].groupby(["Salutation"])[["Salutation", "Age"]].median().to_dict()["Age"]
print("train",avg_age_by_salutation)
for null_age in salutations_null_age:
    df_train_copy.loc[(df_train_copy.Age.isna()) & (df_train_copy.Salutation == null_age), "Age"] = avg_age_by_salutation[null_age]

df_test_copy.loc[df_test_copy.Salutation == "Ms.", "Salutation"] = "Mrs."
salutations_null_age = df_test_copy[df_test_copy["Age"].isna()]["Salutation"].unique()
print("test",salutations_null_age)

avg_age_by_salutation = df_test_copy[df_test_copy["Age"].notna()].groupby(["Salutation"])[["Salutation", "Age"]].median().to_dict()["Age"]
print(avg_age_by_salutation)
for null_age in salutations_null_age:
    df_test_copy.loc[(df_test_copy.Age.isna()) & (df_test_copy.Salutation == null_age), "Age"] = avg_age_by_salutation[null_age]
    
salutations_null_age = df_test_copy[df_test_copy["Age"].isna()]["Salutation"].unique()
print("test",salutations_null_age)

salutation_idx = {}
for i, s in enumerate(set(df_train_copy.Salutation).union(set(df_test_copy.Salutation))):
    salutation_idx[s] = i+1
    
def process_salutations(s):
    try:
        return salutation_idx[s]
    except Exception as e:
        print(f"key failed :{s} {e.__str__()}")

df_train_copy["Salutation"] = df_train_copy["Salutation"].apply(process_salutations)
df_train_copy["Salutation"] = df_train_copy["Salutation"].astype("int", copy=False)

df_test_copy["Salutation"] = df_test_copy["Salutation"].apply(process_salutations)
df_test_copy["Salutation"] = df_test_copy["Salutation"].astype("int", copy=False)


df_train_copy.loc[df_train_copy.Sex == "male", "Sex"] = 1
df_train_copy.loc[df_train_copy.Sex == "female", "Sex"] = 2
df_train_copy["Sex"] = df_train_copy["Sex"].astype("int", copy=False)

df_test_copy.loc[df_test_copy.Sex == "male", "Sex"] = 1
df_test_copy.loc[df_test_copy.Sex == "female", "Sex"] = 2
df_test_copy["Sex"] = df_test_copy["Sex"].astype("int", copy=False)

def preprocess_cabin(x):
    if x == "nan":
        return ord("X")
    else:
        return ord(x[0])

df_train_copy["Cabin"] = df_train_copy["Cabin"].astype("str", copy=False)
df_train_copy["Cabin"] = df_train_copy["Cabin"].apply(preprocess_cabin)

df_test_copy["Cabin"] = df_test_copy["Cabin"].astype("str", copy=False)
df_test_copy["Cabin"] = df_test_copy["Cabin"].apply(preprocess_cabin)

df_train_copy.loc[df_train_copy["Embarked"].notna(), "Embarked"] = 1
df_train_copy.loc[df_train_copy["Embarked"].isna(), "Embarked"] = 0
df_train_copy["Embarked"] = df_train_copy["Embarked"].astype("float", copy=False)

df_test_copy.loc[df_test_copy["Embarked"].notna(), "Embarked"] = 1
df_test_copy.loc[df_test_copy["Embarked"].isna(), "Embarked"] = 0
df_test_copy["Embarked"] = df_test_copy["Embarked"].astype("float", copy=False)

print(df_train_copy["Cabin"].unique(), df_test_copy["Cabin"].unique())

df_train_copy["together"] = df_train_copy["SibSp"] + df_train_copy["Parch"]
df_test_copy["together"] = df_test_copy["SibSp"] + df_train_copy["Parch"]

df_train_copy["Fare"] = np.log1p(df_train_copy["Fare"])
df_test_copy["Fare"] = np.log1p(df_test_copy["Fare"])

df_train_copy["Ticket"] = df_train_copy["Ticket"].apply(lambda x: ord(x[0]))
df_test_copy["Ticket"] = df_test_copy["Ticket"].apply(lambda x: ord(x[0]))


df_train_copy["Age"] = df_train_copy["Age"].astype("float", copy=False).rank()
df_test_copy["Age"] = df_test_copy["Age"].astype("float", copy=False).rank()

df_train_copy.Age.unique()
#from sklearn.preprocessing import MinMaxScaler
#scaling = MinMaxScaler()
#df_train["Age"] = df_train["Age"].values.reshape(-1, 1)))
#df_test_copy["Age"] = df_test_copy["Age"].values.reshape(-1, 1)))

fig, ax = plt.subplots(1, 2)
sns.distplot(df_train["Age"], ax=ax[0])
sns.distplot(np.log1p(df_train["Age"]), ax=ax[1])
#df_train.drop(df_train[df_train.Embarked.isna()].index, inplace=True)

X_train = pd.get_dummies(df_train_copy.drop(["Name", "Survived", "SibSp", "Parch"], axis=1))
y_train = df_train_copy["Survived"]

X_test = pd.get_dummies(df_test_copy.drop(["Name","SibSp", "Parch"], axis=1))
print(list(X_train.columns[:]))
print(list(X_test.columns[:]))
X_train.describe()
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def random_forest(X_train_train, y_train_train,X_test_train, y_test_train):
    model = RandomForestClassifier(n_estimators=250, random_state= 0, max_depth=5)
    default_random_forest = clone(model)
    default_random_forest.fit(X_train_train, y_train_train)
    y_pred = default_random_forest.predict(X_test_train)
    accuracy = accuracy_score(y_pred, y_test_train)
    recall = recall_score(y_pred, y_test_train)
    f1 = f1_score(y_pred, y_test_train)
    return accuracy, recall, f1

    #print(classification_report(y_pred, y_test_train))
    #print("------------------------------------------")
    
def logistic(X_train_train, y_train_train,X_test_train, y_test_train):

    model = LogisticRegression(random_state= 0, max_iter=10000, solver="liblinear", penalty="l1")
    default_logistic = clone(model)
    default_logistic.fit(X_train_train, y_train_train)
    y_pred = default_logistic.predict(X_test_train)
    accuracy = accuracy_score(y_pred, y_test_train)
    recall = recall_score(y_pred, y_test_train)
    f1 = f1_score(y_pred, y_test_train)
    return accuracy, recall, f1
    #print(classification_report(y_pred, y_test_train))
    #print("------------------------------------------")
#    return score

def svc(X_train_train, y_train_train,X_test_train, y_test_train):

    model = SVC(kernel='linear', C=.1, random_state=0, gamma="auto", tol=.001)

    default_svc = clone(model)
    default_svc.fit(X_train_train, y_train_train)
    y_pred = default_svc.predict(X_test_train)
    accuracy = accuracy_score(y_pred, y_test_train)
    recall = recall_score(y_pred, y_test_train)
    f1 = f1_score(y_pred, y_test_train)
    return accuracy, recall, f1
    
    #print(classification_report(y_pred, y_test_train))
    #print("------------------------------------------")
    #return score
    

    
def tree(X_train_train, y_train_train,X_test_train, y_test_train):

    model = DecisionTreeClassifier(random_state=0, max_depth=3, ccp_alpha=0.05)

    default_tree = clone(model)
    default_tree.fit(X_train_train, y_train_train)
    y_pred = default_tree.predict(X_test_train)
    accuracy = accuracy_score(y_pred, y_test_train)
    recall = recall_score(y_pred, y_test_train)
    f1 = f1_score(y_pred, y_test_train)
    return accuracy, recall, f1
    
    #print(classification_report(y_pred, y_test_train))
    #print("------------------------------------------")
    #return score


choice = ["random", "logistic", "tree"]
for i in range(10):
    accuracy = [0, 0, 0]
    recall = [0, 0, 0]
    f1 = [0, 0, 0]
    X_train_train, X_test_train, y_train_train, y_test_train = train_test_split(X_train, y_train, test_size=.45)
    accuracy[0], recall[0], f1[0] = random_forest(X_train_train, y_train_train,X_test_train, y_test_train)
    accuracy[1], recall[1], f1[1] = logistic(X_train_train, y_train_train,X_test_train, y_test_train)
    accuracy[2], recall[2], f1[2] = tree(X_train_train, y_train_train,X_test_train, y_test_train)
    #accuracy[2], recall[2], f1[2] = svc(X_train_train, y_train_train,X_test_train, y_test_train)
    
    print(i, choice[np.argmax(accuracy)], np.max(accuracy),  choice[np.argmax(recall)], np.max(recall), choice[np.argmax(f1)], np.max(f1))
model = RandomForestClassifier(n_estimators=600, random_state= 0, max_depth=3, criterion="entropy")
model.fit(X_train, y_train)
predictions_random = model.predict(X_test)
print(list(zip(X_train.columns, model.feature_importances_)))

model = LogisticRegression(random_state= 0, max_iter=1000)
model.fit(X_train, y_train)
predictions_logistic = model.predict(X_test)

model = DecisionTreeClassifier(random_state=0, max_depth=3, ccp_alpha=0.05)
model.fit(X_train, y_train)
predictions_tree = model.predict(X_test)


df_pred = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
df_pred.columns
df_pred.set_index("PassengerId", inplace=True)
#submission.to_csv("Titanic_Predictions1.csv",index=False)
submission_random = pd.DataFrame({'PassengerId':X_test['PassengerId'],'Survived':predictions_random})
print(accuracy_score(predictions_random, df_pred["Survived"]))

submission_random.set_index("PassengerId", inplace=True)
joined = df_pred.join(submission_random, "PassengerId", lsuffix="left")

display(X_test[X_test["PassengerId"].isin(joined[joined["Survivedleft"] != joined["Survived"]].index.to_list())])

print(accuracy_score(predictions_tree, df_pred["Survived"]))
submission_tree = pd.DataFrame({'PassengerId':X_test['PassengerId'],'Survived':predictions_tree})
submission_tree.to_csv("Titanic_Predictions2.csv",index=False)

submission_tree.set_index("PassengerId", inplace=True)
joined = df_pred.join(submission_tree, "PassengerId", lsuffix="left")

display(X_test[X_test["PassengerId"].isin(joined[joined["Survivedleft"] != joined["Survived"]].index.to_list())])


