import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

import seaborn as sns
from matplotlib import pyplot as plt
import pylab as plot
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.head()
test_data.head()
train_data.describe()
test_data.describe()
params = {
    'axes.labelsize': "large",
    'xtick.labelsize': 'medium',
    'legend.fontsize': 'medium',
    'legend.loc': "best",

}
plot.rcParams.update(params)

train_data['Died'] = 1 - train_data['Survived']
train_data.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()
train_data.groupby('Pclass').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()
train_data.groupby('SibSp').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()
train_data.groupby('Parch').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()
plt.hist([train_data[train_data['Survived'] == 1]['Fare'], train_data[train_data['Survived'] == 0]['Fare']], bins = 30, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()
plt.hist([train_data[train_data['Survived'] == 1]['Age'], train_data[train_data['Survived'] == 0]['Age']], bins = 8, label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()
sing_titles = list()
for name in train_data["Name"]:
    title = name.split(',')[1].split('.')[0].strip()
    if title not in sing_titles: sing_titles.append(title)
print(sing_titles)
sing_test = list()
for name in test_data["Name"]:
    title = name.split(',')[1].split('.')[0].strip()
    if title not in sing_test: sing_test.append(title)
print(sing_test)
# Function that, given a title string, checks it and replaces it with the correct title
def title_corr(t):
    newt = t
    if t == 'Mrs' or t == 'Mr' or t == 'Miss':
        return newt
    elif t == 'Capt' or t == 'Col' or t == 'Major' or t == 'Dr' or t == 'Rev':
        newt = 'Crew'
    elif t == 'Jonkheer' or t == 'Sir' or t == 'the Countess' or t == 'Lady' or t == 'Master':
        newt = 'Noble'
    elif t == 'Don':
        newt = 'Mr'
    elif t == 'Dona' or t == 'Ms' or t == 'Mme':
        newt = 'Mrs'
    elif t == 'Mlle':
        newt = 'Miss'
    else: print("Title not included:", t)
    return newt

# Extract the titles from the name and put them in a list, then correct them
# Train data
titles = list()
for name in train_data["Name"]:
    titles.append(name.split(',')[1].split('.')[0].strip())
for i in range(len(titles)):
    titles[i] = title_corr(titles[i])
train_data["Titles"] = titles

# Plotting
plt.hist([train_data[train_data['Survived'] == 1]['Titles'], train_data[train_data['Survived'] == 0]['Titles']], label = ['Survived','Dead'])
plt.xlabel('Title')
plt.ylabel('Number of passengers')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

# Test data
test_titles = list()
for name in test_data["Name"]:
    test_titles.append(name.split(',')[1].split('.')[0].strip())
for i in range(len(test_titles)):
    test_titles[i] = title_corr(test_titles[i])
test_data["Titles"] = test_titles
title_mapping = {"Mrs": 4, "Miss": 3, "Mr": 0, "Noble": 2,"Crew": 1}
train_data['Title Map'] = train_data['Titles'].map(title_mapping)
test_data['Title Map'] = test_data['Titles'].map(title_mapping)
train_data["Fare"] = train_data["Fare"].fillna(train_data["Fare"].median())
test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].median())
train_data['FareGroup'] = pd.cut(train_data['Fare'],3)
print(train_data[['FareGroup', 'Survived']].groupby('FareGroup', as_index=False).mean().sort_values('Survived', ascending=False))
def group_fare(fare):
    if fare <= 170: return 0
    if fare > 170 and fare <= 340: return 1
    if fare > 340: return 2
    
# Loops over the df and fill the Fare Group column
for i, row in train_data.iterrows():
    train_data.at[i,'Fare Group'] = group_fare(row["Fare"])
# Same for test data
for i, row in test_data.iterrows():
    test_data.at[i,'Fare Group'] = group_fare(row["Fare"])
# Function that returns the median age for passengers from a certain class, sex and title
def calc_age(df, cl, sx, tl):
    a = df.groupby(["Pclass", "Sex", "Titles"])["Age"].median()
    return a[cl][sx][tl]

# Getting the full dataset (more accurate for median calculation)
age_train = train_data.copy()
age_train.drop('PassengerId', axis=1, inplace=True)
age_train.drop('Survived',axis=1, inplace=True)
age_test = test_data.copy()
age_test.drop('PassengerId', axis=1, inplace=True)
df = pd.concat([age_train, age_test], sort=False).reset_index(drop=True)

# Fill up missing ages
for i, row in train_data.iterrows():
    if pd.isna(row['Age']) :
        newage = (calc_age(df, row["Pclass"], row["Sex"], row["Titles"]))
        train_data.at[i,'Age'] = newage
    else: continue
# Same for test data
for i, row in test_data.iterrows():
    if pd.isna(row['Age']) :
        newage = (calc_age(df, row["Pclass"], row["Sex"], row["Titles"]))
        test_data.at[i,'Age'] = newage
    else: continue
train_data['AgeGroup'] = pd.cut(train_data['Age'],5)
print(train_data[['AgeGroup', 'Survived']].groupby('AgeGroup', as_index=False).mean().sort_values('Survived', ascending=False))
def group_age(age):
    if age <= 16: return 4
    if age > 16 and age <= 32: return 1
    if age > 32 and age <= 48: return 2
    if age > 48 and age <= 64: return 3
    if age > 64: return 0

# Loops over the df and fill the Age Group column
for i, row in train_data.iterrows():
    train_data.at[i,'Age Group'] = group_age(row["Age"])
    # Same for test data
for i, row in test_data.iterrows():
    test_data.at[i,'Age Group'] = group_age(row["Age"])
train_data["Family"] = train_data["SibSp"] + train_data["Parch"]
test_data["Family"] = test_data["SibSp"] + test_data["Parch"]
train_data["Embarked"] = train_data["Embarked"].fillna('S')
print(train_data[['Embarked', 'Survived']].groupby('Embarked', as_index=False).mean().sort_values('Survived', ascending=False))
def embarked_rate(embarked_port):
    if embarked_port == 'C': return 2
    if embarked_port == 'Q': return 1
    if embarked_port == 'S': return 0

for i, row in train_data.iterrows():
    train_data.at[i,'Emb Rate'] = embarked_rate(row["Embarked"])
for i, row in test_data.iterrows():
    test_data.at[i,'Emb Rate'] = embarked_rate(row["Embarked"])
sex_mapping = {"male": 0, "female": 1}
train_data['Sex Map'] = train_data['Sex'].map(sex_mapping)
test_data['Sex Map'] = test_data['Sex'].map(sex_mapping)
# Drops some columns
cols_to_drop = ["SibSp", "Parch", "Name", "Age", "Fare",  "Embarked", "Cabin", "Ticket", "Sex", "Titles"]
new_train = train_data.drop(cols_to_drop, axis=1)
new_test = test_data.drop(cols_to_drop, axis=1)

y = train_data["Survived"]
features = ["Pclass", "Sex Map", "Family", "Title Map", "Age Group", "Fare Group", "Emb Rate"]
X = pd.get_dummies(new_train[features])
X_test = pd.get_dummies(new_test[features])
# X = new_train.drop("Survived", axis=1)
# X_test = new_test
model1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model1.fit(X, y)
y1_test = model1.predict(X_test)

model2 = XGBClassifier(max_depth=3, n_estimators=1000, learning_rate=0.05)
model2.fit(X, y)
y2_test = model2.predict(X_test)

model3 = SVC(random_state=1)
model3.fit(X,y)
y3_test = model3.predict(X_test)

model4 = GradientBoostingClassifier(random_state=42)
model4.fit(X, y)
y4_test = model4.predict(X_test)
model1_preds = cross_val_predict(model1, X, y, cv=10)
model1_acc = accuracy_score(y, model1_preds)
model2_preds = cross_val_predict(model2, X, y, cv=10)
model2_acc = accuracy_score(y, model2_preds)
model3_preds = cross_val_predict(model3, X, y, cv=10)
model3_acc = accuracy_score(y, model3_preds)
model4_preds = cross_val_predict(model4, X, y, cv=10)
model4_acc = accuracy_score(y, model4_preds)

print("Random Forest Accuracy:", model1_acc)
print("XGBoost Accuracy:", model2_acc)
print("SVC Accuracy:", model3_acc)
print("GB Accuracy:", model4_acc)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y2_test})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")