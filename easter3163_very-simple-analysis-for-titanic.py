import pandas as pd
train = pd.read_csv("../input/train.csv", index_col = "PassengerId")
test = pd.read_csv("../input/test.csv", index_col="PassengerId")
# Check the row and column of train dataframe
train.shape
train.head()
test.shape
test.head()
%matplotlib inline

import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(data=train, x="Sex", hue='Survived')
pd.pivot_table(train, index="Sex", values="Survived")
sns.countplot(data=train, x="Pclass", hue="Survived")
pd.pivot_table(train, index="Pclass", values="Survived")
sns.countplot(data=train, x="Embarked", hue="Survived")
pd.pivot_table(train, index="Embarked", values="Survived")
# If you don't want Regression, make fit_reg False
sns.lmplot(data=train, x="Age", y="Fare", hue="Survived", fit_reg=False)
low_low_fare = train[train["Fare"] < 100]
sns.lmplot(data=low_low_fare, x="Age", y="Fare", hue="Survived", fit_reg=False)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
train[["FamilySize"]].head()
sns.countplot(data=train, x="FamilySize", hue="Survived")
train.loc[train["FamilySize"]==1, "FamilyType"] = "Single"
train.loc[(train["FamilySize"] > 1) & (train["FamilySize"] < 5), "FamilyType"] = "Nuclear"
train.loc[train["FamilySize"] >=5, "FamilyType"] = "Big"
train[["FamilySize", "FamilyType"]].head()
sns.countplot(data=train, x="FamilyType", hue="Survived")
pd.pivot_table(data=train, index="FamilyType", values="Survived")
train["Name"].head()
def get_title(name):
    return name.split(", ")[1].split(". ")[0]
train["Name"].apply(get_title).unique()
train.loc[train["Name"].str.contains("Mr"), "Title"] = "Mr"
train.loc[train["Name"].str.contains("Miss"), "Title"] = "Miss"
train.loc[train["Name"].str.contains("Mrs"), "Title"] = "Mrs"
train.loc[train["Name"].str.contains("Master"), "Title"] = "Master"

train[["Name", "Title"]].head()
sns.countplot(data=train, x="Title", hue="Survived")
pd.pivot_table(train, index="Title", values="Survived")
train.loc[train["Sex"] == "male", "Sex_encode"] = 0
train.loc[train["Sex"] == "female", "Sex_encode"] = 1

train[["Sex", "Sex_encode"]].head()
test.loc[test["Sex"] == "male", "Sex_encode"] = 0
test.loc[test["Sex"] == "female", "Sex_encode"] = 1

test[["Sex", "Sex_encode"]].head()
train[train["Fare"].isnull()]
test[test["Fare"].isnull()]
train["Fare_fillin"] = train["Fare"]
test["Fare_fillin"] = test["Fare"]
test.loc[test["Fare"].isnull(), "Fare_fillin"] = 0
train["Fare_fillin"] = train["Fare_fillin"] / 10.0
test["Fare_fillin"] = test["Fare_fillin"] / 10.0
train["Embarked"].fillna("S")
train["Embarked_C"] = False
train.loc[train["Embarked"]=='C', "Embarked_C"] = True
train["Embarked_S"] = False
train.loc[train["Embarked"]=='S', "Embarked_S"] = True
train["Embarked_Q"] = False
train.loc[train["Embarked"]=='Q', "Embarked_Q"] = True
train[["Embarked", "Embarked_C", "Embarked_S", "Embarked_Q"]].head()
test["Embarked_C"] = False
test.loc[test["Embarked"]=='C', "Embarked_C"] = True
test["Embarked_S"] = False
test.loc[test["Embarked"]=='S', "Embarked_S"] = True
test["Embarked_Q"] = False
test.loc[test["Embarked"]=='Q', "Embarked_Q"] = True
test[["Embarked", "Embarked_C", "Embarked_S", "Embarked_Q"]].head()
train["Age"].fillna(train["Age"].mean(), inplace=True)
test["Age"].fillna(test["Age"].mean(), inplace=True)
train["Child"] = False
train.loc[train["Age"] < 15, "Child"] = True
train[["Age", "Child"]].head(10)
test["Child"] = False
test.loc[test["Age"] < 15, "Child"] = True
test[["Age", "Child"]].head(10)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
test[["FamilySize"]].head()
train["Single"] = False
train.loc[train["FamilySize"]==1, "Single"] = True
train["Nuclear"] = False
train.loc[(train["FamilySize"]>1)&(train["FamilySize"]<5), "Nuclear"] = True
train["Big"] = False
train.loc[train["FamilySize"] >=5, "Big"] = True
train[["FamilySize", "Single", "Nuclear", "Big"]].head(10)
test["Single"] = False
test.loc[test["FamilySize"]==1, "Single"] = True
test["Nuclear"] = False
test.loc[(test["FamilySize"]>1)&(test["FamilySize"]<5), "Nuclear"] = True
test["Big"] = False
test.loc[test["FamilySize"] >=5, "Big"] = True
test[["FamilySize", "Single", "Nuclear", "Big"]].head(10)
train["Master"] = False
train.loc[train["Name"].str.contains("Master"), "Master"] = True
train[["Name", "Master"]].head(10)
test["Master"] = False
test.loc[test["Name"].str.contains("Master"), "Master"] = True
test[["Name", "Master"]].head(10)
feature_names = ["Pclass", "Sex_encode", "Fare_fillin", "Embarked_C", "Embarked_S", "Embarked_Q", "Child", "Single", "Nuclear", "Big", "Master"]
feature_names
label_name = "Survived"
label_name
from sklearn.model_selection import train_test_split
random_seed=0
Y = train[label_name]
X = train[feature_names]
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=random_seed)
#X_train = X_train[feature_names]
#y_train = y_train[label_name]
X_test = test[feature_names]
# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
rforest_model = RandomForestClassifier(n_estimators=100)
svc = SVC()
knn = KNeighborsClassifier(n_neighbors = 3)
gbmodel = GradientBoostingClassifier(max_depth=12)
gnb = GaussianNB()
LR = LogisticRegression()
mlp = MLPClassifier(solver='lbfgs', random_state=0)
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
x_train = X_train.values
x_val = X_val.values
test_nn = test[feature_names].values
model = Sequential()

batch_size = 32
epochs = 200

model.add(Dense(32, activation="relu", input_dim=11))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train,y_train, epochs=epochs, batch_size=batch_size)
test_loss, test_acc = model.evaluate(x_val, y_val)
print('Test Score:{}'.format(test_acc))
nn_predict = model.predict(test_nn)
rforest_model.fit(X_train, y_train)
svc.fit(X_train, y_train)
knn.fit(X_train, y_train)
gbmodel.fit(X_train, y_train)
gnb.fit(X_train, y_train)
LR.fit(X_train, y_train)
mlp.fit(X_train, y_train)
score = []
score.append(rforest_model.score(X_val, y_val))
score.append(svc.score(X_val, y_val))
score.append(knn.score(X_val, y_val))
score.append(gbmodel.score(X_val, y_val))
score.append(gnb.score(X_val, y_val))
score.append(LR.score(X_val, y_val))
score.append(mlp.score(X_val,y_val))
score
score = []
score.append(rforest_model.score(X_train, y_train))
score.append(svc.score(X_train, y_train))
score.append(knn.score(X_train, y_train))
score.append(gbmodel.score(X_train, y_train))
score.append(gnb.score(X_train, y_train))
score.append(LR.score(X_train, y_train))
score.append(mlp.score(X_train,y_train))
score
x = X.values
y = Y.values
model.fit(x,y, epochs=epochs, batch_size=batch_size)
gbmodel.fit(X, Y)
predictions = gbmodel.predict(X_test)
#nn_predict = model.predict(test_nn)
#predictions = nn_predict
#predictions = [0 if pred<0.5 else 1 for pred in predictions ]
submission = pd.read_csv('../input/gender_submission.csv')
submission['Survived'] = predictions
submission.head()
submission.to_csv('./simpletitanic.csv', index=False)