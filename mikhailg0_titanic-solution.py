import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import LabelEncoder


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data_train = pd.read_csv('/kaggle/input/titanic/train.csv')
data_test = pd.read_csv('/kaggle/input/titanic/test.csv')
data = pd.concat([data_train, data_test], ignore_index=True, sort=False)
print("Colums: ", data_train.columns.values)
print("Shape: ", data.shape)

# Missing values
plt.ylabel("Missing values:")
plt.plot(pd.DataFrame(data.isnull().sum()))
plt.show()
print(data.isnull().sum())

sns.heatmap(data.isnull(), cbar=False).set_title("Missing values heatmap")
plt.show()

pClass_1 = round(
    (data_train[data_train.Pclass == 1].Survived == 1).value_counts()[1] /
    len(data_train[data_train.Pclass == 1]) * 100, 2)
pClass_2 = round(
    (data_train[data_train.Pclass == 2].Survived == 1).value_counts()[1] /
    len(data_train[data_train.Pclass == 2]) * 100, 2)
pClass_3 = round(
    (data_train[data_train.Pclass == 3].Survived == 1).value_counts()[1] /
    len(data_train[data_train.Pclass == 3]) * 100, 2)

pClassDf = pd.DataFrame(
    {"Survived": {"Class 1": pClass_1,
                  "Class 2": pClass_2,
                  "Class 3": pClass_3},
     "Not survived": {"Class 1": 100 - pClass_1,
                      "Class 2": 100 - pClass_2,
                      "Class 3": 100 - pClass_3}})
pClassDf.plot.bar().set_title("Survived ~ Class")
plt.show()

data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sex_1 = round(
    (data_train[data_train.Sex == 'male'].Survived == 1).value_counts()[1] /
    len(data_train[data_train.Sex == 'male']) * 100, 2)
sex_2 = round(
    (data_train[data_train.Sex == 'female'].Survived == 1).value_counts()[1] /
    len(data_train[data_train.Sex == 'female']) * 100, 2)

pClassDf = pd.DataFrame(
    {"Survived": {"Male": sex_1,
                  "Female": sex_2},
     "Not survived": {"Male": 100 - sex_1,
                      "Female": 100 - sex_2}})
pClassDf.plot.bar().set_title("Survived ~ Sex")
plt.show()

data_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
ss = pd.DataFrame()
ss['survived'] = data_train.Survived
ss['sibling_spouse'] = pd.cut(data_train.SibSp, [0, 1, 2, 3, 4, 5, 6, 7, 8], include_lowest=True)

x = sns.countplot(x="sibling_spouse", hue="survived", data=ss, palette=["C1", "C0"]).legend(
    labels=["Not survived", "Survived"])
x.set_title("Survival ~ Number of siblings or spouses")
plt.show()

data_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pc = pd.DataFrame()
pc['survived'] = data_train.Survived
pc['parents_children'] = pd.cut(data_train.Parch, [0, 1, 2, 3, 4, 5, 6], include_lowest=True)
x = sns.countplot(x="parents_children", hue="survived", data=pc, palette=["C1", "C0"]).legend(
    labels=["Not survived", "Survived"])

x.set_title("Survival ~ Parents/Children")
plt.show()

data_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
data['AgeRange'] = pd.cut(data.Age, [0, 10, 20, 30, 40, 50, 60, 70, 80])
sns.countplot(x="AgeRange", hue="Survived", data=data, palette=["C1", "C0"]).legend(labels=["Not survived", "Survived"])
plt.show()
data['FareCategory'] = pd.cut(data_train['Fare'], bins=[0, 7.90, 14.45, 31.28, 120], labels=['Low', 'Mid',
                                                                                                    'High_Mid', 'High'])
x = sns.countplot(x="FareCategory", hue="Survived", data=data, palette=["C1", "C0"]).legend(
    labels=["Not survived", "Survived"])
x.set_title("Survival ~ Fare")
p = sns.countplot(x="Embarked", hue="Survived", data=data_train, palette=["C1", "C0"])
p.set_xticklabels(["Southampton", "Cherbourg", "Queenstown"])
p.legend(labels=["Not survived", "Survived"])
p.set_title("Survival ~ Embarking.")
data['Family'] = data.Parch + data.SibSp
data['IsAlone'] = data.Family == 0
data['SmallFamily'] = data['Family'].map(lambda s: 1 if 1 <= s <= 3 else 0)
data['BigFamily'] = data['Family'].map(lambda s: 1 if 4 <= s else 0)
data['Salutation'] = data.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip()) 
print(data.Salutation.unique())
data.Salutation.nunique()
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in data["Name"]]
data['Title']=data.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
newtitles={
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"}

data['Title']=data.Title.map(newtitles)

data['Mother'] = (data['Title'] == 'Mrs') & (data['Parch'] > 0)
data['Mother'] = data['Mother'].astype(int)
grp = data.groupby(['Sex', 'Pclass', 'Title'])
data.Age = grp.Age.apply(lambda x_: x_.fillna(x_.median()))
data.Age.fillna(data.Age.median, inplace=True)

data['AgeRange'] = pd.cut(data['Age'].astype(int), 5)
data['Ticket_Lett'] = data['Ticket'].apply(lambda x: str(x)[0])
data['Ticket_Lett'] = data['Ticket_Lett'].apply(lambda x: str(x))
data['Ticket_Lett'] = np.where((data['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), data['Ticket_Lett'],
                                   np.where((data['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
data['Ticket_Len'] = data['Ticket'].apply(lambda x: len(x))
data.Fare.fillna(data.Fare.mean(), inplace = True)
# data['FareCategory'] = pd.qcut(data['Fare'], 4)
data.Embarked.fillna(data.Embarked.mode()[0], inplace = True)
data.Cabin.fillna('NA', inplace=True)

data['Cabin'] = data['Cabin'].map(lambda s: s[0])

data['Title'] = LabelEncoder().fit_transform(data['Title'])

data = pd.concat([data,
                pd.get_dummies(data.Cabin, prefix="Cabin"),
                pd.get_dummies(data.AgeRange, prefix="AgeRange"), 
                pd.get_dummies(data.Embarked, prefix="Embarked", drop_first = True),
                pd.get_dummies(data.Title, prefix="Title", drop_first = True),
                pd.get_dummies(data.FareCategory, prefix="Fare", drop_first = True), 
                pd.get_dummies(data.Ticket_Lett, prefix="Ticket", drop_first = True), 
                pd.get_dummies(data.Pclass, prefix="Class", drop_first = True)], axis=1)

data['IsAlone'] = LabelEncoder().fit_transform(data['IsAlone'])
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])

data.drop(['Pclass', 'Fare','Cabin', 'FareCategory','Name','Salutation', 'Ticket_Lett', 'Ticket','Embarked', 'AgeRange', 'SibSp', 'Parch', 'Age'], axis=1, inplace=True)
print("Colums: ", data.columns.values)
import sklearn


# Prediction

X_pred = data[data.Survived.isnull()].drop(['Survived'], axis=1)

# Training data

train_data = data.dropna()
X = train_data.drop(['Survived'], axis=1)
y = train_data['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler



clf = GradientBoostingClassifier(learning_rate=0.1,
                                 n_estimators=700,
                                 max_depth=2)

clf.fit(X_train, np.ravel(y_train))
print("RF Accuracy: " + repr(round(clf.score(X_test, y_test) * 100, 2)) + "%")

result_rf = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print('The cross validated score for Random forest is:', round(result_rf.mean() * 100, 2))
y_pred = cross_val_predict(clf, X_train, y_train, cv=5)
# Random forest
# entropy
# gini

clf = RandomForestClassifier(criterion='entropy',
                             n_estimators=700,
                             min_samples_split=5,
                             min_samples_leaf=1,
                             max_features = "auto",
                             oob_score=True,
                             random_state=0,
                             n_jobs=-1)

clf.fit(X_train, np.ravel(y_train))
print("RF Accuracy: " + repr(round(clf.score(X_test, y_test) * 100, 2)) + "%")

result_rf = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print('The cross validated score for Random forest is:', round(result_rf.mean() * 100, 2))
y_pred = cross_val_predict(clf, X_train, y_train, cv=5)

result = clf.predict(X_pred)
submission = pd.DataFrame({'PassengerId':X_pred.PassengerId,'Survived':result})
submission.Survived = submission.Survived.astype(int)
print(submission.shape)
filename = 'TitanicPredictions4.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)