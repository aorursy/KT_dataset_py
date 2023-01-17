import pandas as pd



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print (train.head())
print(train.describe())

train.shape
#train["Survived"].value_counts()

train["Survived"].value_counts(normalize = True)
train["Survived"][train["Sex"]=='female'].value_counts(normalize=True)
#create new column child and assign 1 to less than 18, 0 to 18 or older

train["Child"] = float('NaN')

train["Child"][train["Age"]<18] = 1

train["Child"][train["Age"]>=18] = 0

train["Child"]



#survival rates for children less than 18yo

(train["Survived"][train["Child"] ==1]).value_counts(normalize=True)



import numpy as np

from sklearn import tree
#Encode sex male=0; female=1

train["Sex"][train["Sex"]=="female"] = 1

train["Sex"][train["Sex"]=="male"]=0

test["Sex"][test["Sex"]=="female"] = 1

test["Sex"][test["Sex"]=="male"]=0



#fillna for Embark

train["Embarked"]= train["Embarked"].fillna('S')



# Convert the Embarked classes to integer form

#train["Embarked"][train["Embarked"] == "S"] = 0

#train["Embarked"][train["Embarked"]=="C"] = 1

#train["Embarked"][train["Embarked"]=="Q"] = 2



#Print the Sex and Embarked columns

#print(train.Sex)

print(train.Embarked.head())





train["Age"] = train["Age"].fillna(train["Age"].median())

test["Age"] = test["Age"].fillna(test["Age"].median())
#create target

y_train = train['Survived'].values

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values



#Fit decision tree

model_one = tree.DecisionTreeClassifier()

model_one = model_one.fit(features_one, y_train)



print(model_one.feature_importances_)

print(model_one.score(features_one, y_train))
test.Fare[152]=test.Fare.median()



test_features = test[["Pclass", "Sex", "Age", "Fare"]].values



preds_one=model_one.predict(test_features)



#Create df with 2 columns: ids & survived

PassengerId = np.array(test["PassengerId"]).astype(int)

output1 = pd.DataFrame(preds_one, PassengerId, columns=["Survived"])

print(output1)



print(output1.shape)



output1.to_csv("output1.csv", index_label=["PassengerId"])
features_two = train [["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values



model_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state=1)

model_two = model_two.fit(features_two, y_train)



print(model_two.score(features_two, y_train))
train_two = train.copy()

train_two["family_size"] = train_two["SibSp"] + train_two["Parch"]+1



features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values



#Define tree classifier

model_three = tree.DecisionTreeClassifier()

model_three = model_three.fit(features_three, y_train)



print(model_three.score(features_three, y_train))
from sklearn.ensemble import RandomForestClassifier



features_forest = train[["Pclass","Age", "Sex", "Fare", "SibSp", "Parch"]].values



forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators =100, random_state=1)

model_forest = forest.fit(features_forest, y_train)



print (model_forest.score(features_forest, y_train))
test_features = test[["Pclass","Age","Sex","Fare","SibSp","Parch" ]].values

pred_forest = model_forest.predict(test_features)

print(len(pred_forest))
print(model_two.feature_importances_)

print(model_two.score(features_two,y_train))