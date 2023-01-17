import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import statistics
import seaborn as sns
import sklearn.metrics as metrics

from numpy import inf
from scipy import stats
from statistics import median
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # For Random Forest Classification
from sklearn.svm import SVC # For SVC algorithm
from sklearn import model_selection # For Power Tuning
from sklearn.model_selection import cross_val_score
train_dataset = pd.read_csv("../input/train.csv")
train_dataset.head()
print ("Shape:", train_dataset.shape)
train_dataset.isnull().sum()
print("The", round(train_dataset.Age.isnull().sum()/891,2)*100, "% of the observations have a missing Age attribute")
print("The", round(train_dataset.Cabin.isnull().sum()/891,2)*100, "% of the observations have a missing Cabin attribute")
train_dataset = train_dataset.drop(labels=['Cabin'], axis=1)
# id of row where age is null
PassengerID_toDrop_AgeNull = train_dataset.PassengerId[train_dataset.Age.isnull()].tolist()
PassengerID_toDrop_EmbarkedNull = train_dataset.PassengerId[train_dataset.Embarked.isnull()].tolist()
dataset_toDelete = []
for i in range(train_dataset.shape[0]):
    for j in range(pd.DataFrame(PassengerID_toDrop_AgeNull).shape[0]):
        if train_dataset.loc[i, 'PassengerId'] == PassengerID_toDrop_AgeNull[j]:
            dataset_toDelete.append(train_dataset.iloc[i,:])
            
dataset_toDelete = pd.DataFrame(dataset_toDelete)

# Histogram of the client with Age null
%matplotlib inline 

count = dataset_toDelete['Survived'].value_counts()

fig = plt.figure(figsize=(5,5)) # define plot area
ax = fig.gca() # define axis    

count.plot.bar(ax = ax, color = 'b')
# Draw a nested barplot to show Age for class and sex
sns.set(style="whitegrid")

g = sns.FacetGrid(train_dataset, row="Pclass", col="Sex")
g.map(plt.hist, "Age", bins=20)
for sur in train_dataset['Survived'].unique():
    for sex in train_dataset['Sex'].unique():
        for pclass in sorted(train_dataset['Pclass'].unique()):
            median_age = train_dataset[(train_dataset['Survived'] == sur) & (train_dataset['Sex'] == sex) & (train_dataset['Pclass'] == pclass)]['Age'].median()
            if sur == 0:  
                print("Median age for Not Survived", sex, "of the", pclass, "°class =", median_age)
            else:
                print("Median age for Survived", sex, "of the", pclass, "°class =", median_age)
            train_dataset.loc[(train_dataset['Survived'] == sur) & (train_dataset['Sex'] == sex) & (train_dataset['Pclass'] == pclass) & (train_dataset['Age'].isnull()), 'Age'] = median_age 
train_dataset.isnull().sum()
# Code for isolating the client with Embarked Null 

dataset_toDelete = []
for i in range(train_dataset.shape[0]):
    for j in range(pd.DataFrame(PassengerID_toDrop_EmbarkedNull).shape[0]):
        if train_dataset.loc[i, 'PassengerId'] == PassengerID_toDrop_EmbarkedNull[j]:
            dataset_toDelete.append(train_dataset.iloc[i,:])
            
dataset_toDelete = pd.DataFrame(dataset_toDelete)
Embarked_null = dataset_toDelete
Embarked_null
# Bar Plot of Embarked
df = train_dataset

# Grouped boxplot
sns.set(font_scale = 1.50)
sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(7, 7))
graph = sns.countplot(y="Embarked", data=df, ax = ax, color="b")
#graph.set_xticklabels(graph.get_xticklabels(), rotation='vertical')
graph.set_title('Bar Chart of Embarked')

# Draw a nested barplot to show embarked for class and sex
sns.set(style="whitegrid")

g = sns.catplot(x="Embarked", hue="Pclass", col="Sex", kind="count", data=train_dataset, palette="muted")
FirstClass_Women_S = train_dataset.loc[ (train_dataset.Sex == "female") & (train_dataset.Embarked == "S") & (train_dataset.Pclass == 1), :]
print("% Survived Women in First Class from Southampton:", round((FirstClass_Women_S['Survived'].sum()/FirstClass_Women_S['Survived'].count())*100,2))
FirstClass_Women_C = train_dataset.loc[ (train_dataset.Sex == "female") & (train_dataset.Embarked == "C") & (train_dataset.Pclass == 1), :]
print("% Survived Women in First Class from Cherbourg:", round((FirstClass_Women_C['Survived'].sum()/FirstClass_Women_C['Survived'].count())*100,2))
# Fill na in Embarked with "C"
train_dataset.Embarked = train_dataset.Embarked.fillna('C')
train_dataset.isnull().sum()
train_dataset.head()
g = sns.FacetGrid(train_dataset, row="Survived", col="Pclass")

g.map(plt.hist, "Age", bins = 20)
g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train_dataset, kind = "bar")
g.set_ylabels("Survival Probability")
g.fig.suptitle("Survival Probability by Sex and Passengers Class")
# Distribution plot of Fares
g = sns.distplot(train_dataset["Fare"], bins = 20, kde=False)
g.set_title("Distribution plot of Fares")
train_dataset.Fare.describe()
train_dataset.loc[ train_dataset.Fare > train_dataset.Fare.median(), "Fare_Bound" ] = 1 # High Fare type
train_dataset.loc[ train_dataset.Fare <= train_dataset.Fare.median(), "Fare_Bound" ] = 2 # Low Fare type
g = sns.catplot(x="Fare_Bound", hue="Pclass", col="Survived", kind="count", data=train_dataset, palette="muted")
g = sns.catplot(x="Survived", col="Fare_Bound", kind="count", data=train_dataset, palette="muted")
print("Survived people that had paid more than median Fare:", train_dataset.loc[ (train_dataset.Fare_Bound == 1) & (train_dataset.Survived == 1), "Survived" ].sum())
print("Survived people that had paid equal or less than median Fare:", train_dataset.loc[ (train_dataset.Fare_Bound == 2) & (train_dataset.Survived == 1), "Survived" ].sum())
print("People that had paid more than median Fare:", train_dataset.loc[ (train_dataset.Fare_Bound == 1), "Survived" ].count())
print("People that had paid equal or less than median Fare:", train_dataset.loc[ (train_dataset.Fare_Bound == 2), "Survived" ].count())
print(" % Survived among people that had paid more than median:", round(train_dataset.loc[ (train_dataset.Fare_Bound == 1) & (train_dataset.Survived == 1), "Survived" ].sum()/train_dataset.loc[ (train_dataset.Fare_Bound == 1), "Survived" ].count()*100,2),"%")
print(" % Survived among people that had paid equal or less than median:", round(train_dataset.loc[ (train_dataset.Fare_Bound == 2) & (train_dataset.Survived == 1), "Survived" ].sum()/train_dataset.loc[ (train_dataset.Fare_Bound == 2), "Survived" ].count()*100,2), "%")

train_dataset.head()
SibSp_Pos = train_dataset.loc[ train_dataset.SibSp > 0, :]
SibSp_Null = train_dataset.loc[ train_dataset.SibSp == 0, :]
Parch_Pos = train_dataset.loc[ train_dataset.Parch > 0, :]
Parch_Null = train_dataset.loc[ train_dataset.Parch == 0, :]
print("% Survived with Siblings/Spouses number positive:", round(SibSp_Pos.Survived.sum()/SibSp_Pos.Survived.count()*100), "%")
print("% Survived with Siblings/Spouses number null:", round(SibSp_Null.Survived.sum()/SibSp_Null.Survived.count()*100), "%")
print("% Survived with Parents / Children number positive:", round(Parch_Pos.Survived.sum()/Parch_Pos.Survived.count()*100), "%")
print("% Survived with Parents / Children number null:", round(Parch_Null.Survived.sum()/Parch_Null.Survived.count()*100), "%")
g = sns.catplot(x="SibSp", y="Survived", kind="bar", data=train_dataset)
g.set_ylabels("Survival Probability")
g = sns.catplot(x="Parch", y="Survived", kind="bar", data=train_dataset)
g.set_ylabels("Survival Probability")
train_dataset = pd.get_dummies(train_dataset, columns=['Sex'])
train_dataset.head()
dataset_to_train = train_dataset.drop(labels=['Name', 'Ticket'], axis = 1)
dataset_to_train = pd.get_dummies(dataset_to_train, ['Embarked'])
label_to_train = dataset_to_train.loc[:, ['Survived']]
dataset_to_train = dataset_to_train.drop(labels=['Survived'], axis = 1)
dataset_to_train.head()
data_train, data_test, label_train, label_test = train_test_split(dataset_to_train, label_to_train, test_size = 0.3, random_state=7)
data_train = data_train.drop(labels=['PassengerId'], axis = 1)
data_test = data_test.drop(labels=['PassengerId'], axis = 1)
x = []
for i in range(100):
    x.append(i+1)
model = RandomForestClassifier(oob_score = True)
parameters = {'n_estimators':x}
power_tuning = model_selection.GridSearchCV(model, parameters)
model.fit(data_train, label_train)
model_tuned = power_tuning.fit(data_train, label_train.Survived)
model_tuned.best_estimator_
print("The best parameter for 'n_estimator' is:", model_tuned.best_estimator_.n_estimators)
model_Power_tuned = RandomForestClassifier(n_estimators = model_tuned.best_estimator_.n_estimators, oob_score = True)
model_Power_tuned.fit(data_train, label_train.Survived)
print("Score of the training dataset obtained using an out-of-bag estimate:" , round(model_Power_tuned.oob_score_,6))
prediction = model_Power_tuned.predict(data_test)
confusion_mat = metrics.confusion_matrix(label_test, prediction)
columns = ['Not-Survived', 'Survived']
plt.imshow(confusion_mat, cmap=plt.cm.Blues, interpolation='nearest')
plt.title('Confusion Matrix')
plt.xticks([0,1], columns, rotation='vertical')
plt.yticks([0,1], columns)
plt.colorbar()
plt.show()
print("Accuracy:", round(metrics.accuracy_score(label_test, prediction),6))
print("Recall:", metrics.recall_score(label_test, prediction))
print("Precision:", round(metrics.precision_score(label_test, prediction),6))
score_Cross_val = cross_val_score(model_Power_tuned, data_train, label_train.Survived, cv=10).mean()
print ("Cross Validation Score:", round(score_Cross_val,6))
print("Accuracy:", round(metrics.accuracy_score(label_test, prediction),6))
print("Recall:", metrics.recall_score(label_test, prediction))
print("Precision:", round(metrics.precision_score(label_test, prediction),6))
score_Cross_val = cross_val_score(model, data_train, label_train.Survived, cv=10).mean()
print ("Cross Validation Score:", score_Cross_val)
test_dataset = pd.read_csv("../input/test.csv")
test_dataset.head()
Passenger_id_test = test_dataset.loc[:, 'PassengerId']
test_dataset.isnull().sum()
test_dataset.loc[ (test_dataset.Pclass == 1) & (test_dataset.Sex == "female") & (test_dataset.Age.isnull()), "Age" ] = test_dataset.loc[ (test_dataset.Pclass == 1) & (test_dataset.Sex == "female") & (test_dataset.Age.isnull()), "Age" ].fillna(test_dataset.loc[ (test_dataset.Pclass == 1) & (test_dataset.Sex == "female") ]["Age"].median())
test_dataset.loc[ (test_dataset.Pclass == 2) & (test_dataset.Sex == "female") & (test_dataset.Age.isnull()), "Age" ] = test_dataset.loc[ (test_dataset.Pclass == 2) & (test_dataset.Sex == "female") & (test_dataset.Age.isnull()), "Age" ].fillna(test_dataset.loc[ (test_dataset.Pclass == 2) & (test_dataset.Sex == "female") ]["Age"].median())
test_dataset.loc[ (test_dataset.Pclass == 3) & (test_dataset.Sex == "female") & (test_dataset.Age.isnull()), "Age" ] = test_dataset.loc[ (test_dataset.Pclass == 3) & (test_dataset.Sex == "female") & (test_dataset.Age.isnull()), "Age" ].fillna(test_dataset.loc[ (test_dataset.Pclass == 3) & (test_dataset.Sex == "female") ]["Age"].median())

test_dataset.loc[ (test_dataset.Pclass == 1) & (test_dataset.Sex == "male") & (test_dataset.Age.isnull()), "Age" ] = test_dataset.loc[ (test_dataset.Pclass == 1) & (test_dataset.Sex == "male") & (test_dataset.Age.isnull()), "Age" ].fillna(test_dataset.loc[ (test_dataset.Pclass == 1) & (test_dataset.Sex == "male") ]["Age"].median())
test_dataset.loc[ (test_dataset.Pclass == 2) & (test_dataset.Sex == "male") & (test_dataset.Age.isnull()), "Age" ] = test_dataset.loc[ (test_dataset.Pclass == 2) & (test_dataset.Sex == "male") & (test_dataset.Age.isnull()), "Age" ].fillna(test_dataset.loc[ (test_dataset.Pclass == 2) & (test_dataset.Sex == "male") ]["Age"].median())
test_dataset.loc[ (test_dataset.Pclass == 3) & (test_dataset.Sex == "male") & (test_dataset.Age.isnull()), "Age" ] = test_dataset.loc[ (test_dataset.Pclass == 3) & (test_dataset.Sex == "male") & (test_dataset.Age.isnull()), "Age" ].fillna(test_dataset.loc[ (test_dataset.Pclass == 3) & (test_dataset.Sex == "male") ]["Age"].median())

test_dataset.isnull().sum()
test_dataset.loc[test_dataset.Fare.isnull(), :]
train_dataset.loc[ (train_dataset.Pclass == 3) & (train_dataset.Sex_male == 1) & (train_dataset.Age > 60), :]
train_dataset.loc[ (train_dataset.Pclass == 3) & (train_dataset.Sex_male == 1) & (train_dataset.Age > 60), ['Fare']].median()
test_dataset.loc[test_dataset.Fare.isnull(), :] = test_dataset.loc[test_dataset.Fare.isnull(), :].fillna(train_dataset.loc[ (train_dataset.Pclass == 3) & (train_dataset.Sex_male == 1) & (train_dataset.Age > 60), ['Fare']].median())
test_dataset.isnull().sum()
test_dataset = test_dataset.drop(labels=['Cabin', 'PassengerId', 'Name', 'Ticket'], axis = 1)
test_dataset.head()
test_dataset.loc[ test_dataset.Fare > test_dataset.Fare.median(), "Fare_Bound" ] = 1 # High Fare type
test_dataset.loc[ test_dataset.Fare <= test_dataset.Fare.median(), "Fare_Bound" ] = 2 # Low Fare type
test_dataset = pd.get_dummies(test_dataset, columns=['Sex', 'Embarked'])
test_dataset.head()
prediction = model.predict(test_dataset)
my_submission = pd.DataFrame({'PassengerId': Passenger_id_test, 'Survived': prediction})
my_submission.to_csv('submission.csv', index=False)