# Load in libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

from sklearn.preprocessing import Imputer

# Load in the train and test datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
train['Has_Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
train = train.drop(["PassengerId","Ticket","Cabin"], axis=1)
test = test.drop(["Ticket","Cabin"], axis=1)
full_data = [train, test]


for data in full_data: 
    data["Sex"] = data["Sex"].map({'female':0,'male':1}).astype(int)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data["Embarked"] = data["Embarked"].fillna('S')
    age_avg = data['Age'].mean()
    age_std = data['Age'].std()
    age_null_count = data['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    data['Age'][np.isnan(data['Age'])] = age_null_random_list
    data['Age'] = data['Age'].astype(int)
    data['Fare'] = data['Fare'].fillna(train['Fare'].median())
for data in full_data:
    data['Title'] = data['Name'].apply(get_title)
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
for data in full_data:
    data["Embarked"] = data["Embarked"].map({'S':0, 'C':1, 'Q':2}).astype(int)
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
    
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)
    
for data in full_data:
    data.loc[data["Age"] <= 16, 'Age'] = 0
    data.loc[(data["Age"] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data["Age"] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data["Age"] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[(data["Age"] > 64), 'Age'] = 4
    
    data.loc[data["Fare"] <= 7.91, 'Fare'] = 0
    data.loc[(data["Fare"] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data["Fare"] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
    data.loc[(data["Fare"] > 31), 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)
train = train.drop(["Name"], axis=1)
test = test.drop(["Name"], axis=1)
train.head()
plt.figure(figsize=(14,12))
sns.heatmap(
    train.astype(float).corr(),
    cmap=plt.cm.RdBu,
    vmax=1.0,
    linewidths=0.1,
    linecolor='white',
    square=True,
    annot=True
)
g = sns.pairplot(train[[u"Survived",u"Pclass",u"Sex",u"Age",u"SibSp",u"Parch",u"Fare",u"Embarked",u"Has_Cabin",u"FamilySize",u"IsAlone"]],
                 hue="Survived", palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10))
g.set(xticklabels=[])
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

predictors = ["Sex","Pclass","Age","Fare","Embarked","Has_Cabin","FamilySize", "Name_length", "Title"]
X = train[predictors]
y = train.Survived

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(train_X, train_y)

predicted_survive = decision_tree_model.predict(val_X)
for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    survive = [1 if p >= i else 0 for p in predicted_survive]
    print("Threshold: " + str(i))
    print("Decision Tree Model: " + str(mean_absolute_error(val_y, survive)))

random_forest_model = RandomForestRegressor()
random_forest_model.fit(train_X, train_y)

predicted_survive = random_forest_model.predict(val_X)
for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    survive = [1 if p >= i else 0 for p in predicted_survive]
    print("Threshold: " + str(i))
    print("Random Forest Model: " + str(mean_absolute_error(val_y, survive)))
def get_decision_tree_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    maes = []
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        val = [1 if p >= i else 0 for p in preds_val]
        mae = mean_absolute_error(targ_val, val)
        maes += [mae]
    return(maes)

def get_random_forest_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    maes = []
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        val = [1 if p >= i else 0 for p in preds_val]
        mae = mean_absolute_error(targ_val, val)
        maes += [mae]
    return(maes)

for max_leaf_nodes in [5, 50, 500, 5000,]:
    decision_tree_maes = get_decision_tree_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes(decision tree): %d  \t\t Mean Absolute Error:  %s" %(max_leaf_nodes, decision_tree_maes))
    
print()
for max_leaf_nodes in [5, 50, 500, 5000,]:
    random_forest_maes = get_random_forest_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes(random forest): %d  \t\t Mean Absolute Error:  %s" %(max_leaf_nodes, random_forest_maes))
    
model = RandomForestRegressor(max_leaf_nodes=5000, random_state=0)
model.fit(X,y)

test_X = test[predictors]
predict_survive = model.predict(test_X)
survive = [1 if p >= 0.6 else 0 for p in predict_survive]

#Submit
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': survive})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)