import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
def process_cabin(df):
    df.Cabin = df.Cabin.fillna('n')
    df.Cabin = df.Cabin.apply(lambda x: x[0].lower())
    return df
import re

def process_ticket(df):
    df.Ticket = df.Ticket.fillna('n')
    df.Ticket = df.Ticket.apply(lambda x: re.sub("[!*?,.';:\/() ]", "", x.lower()))
    return df
def name_extractor(df):
    def prefix_extractor(row):
        i = row.Name.find('.')
        if i == -1:
            return 'N'
        else:
            cut = row.Name[:i]
            return cut[cut.rfind(' ')+1:]

    for index, row in df.iterrows():
        sec = re.compile("\((.*)\)").search(row.Name)
        if sec is None:
            df.loc[index, 'Altername'] = 'N'
        else:
            df.loc[index, 'Altername'] = sec.group(1).replace('"', '')

        name = re.findall(r"[\w']+", row.Name)
        df.loc[index, 'Secondname'] = name[0]

        pref = prefix_extractor(row)
        df.loc[index, 'Nameprefix'] = pref

        try:
            df.loc[index, 'Firstname'] = name[name.index(pref) + 1]
        except ValueError:
            df.loc[index, 'Firstname'] = name[1]
    return df
def process_ages(df):
    df['Age'].fillna(df['Age'].median(), inplace = True)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 100)
    group_names = ['unknown', 'baby', 'child', 'teenager', 'young', 'adult', 'mature', 'old']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df['CatAge'] = categories
    return df
def feature_engineering(df):
    df = process_ages(df)
    df = process_cabin(df)
    df = process_ticket(df)
    df = name_extractor(df)
    
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)
    df['FamilySize'] = df['SibSp'] + df['Parch']
    
    df['IsAlone'] = 1
    df['IsAlone'].loc[df['FamilySize'] > 1] = 0
    
    return df
train_e, test_e = feature_engineering(train), feature_engineering(test)
g = sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=train_e)
g = sns.catplot(x="Cabin", kind="count", palette="ch:.25", data=train_e[train_e['Cabin']!='n'])
g = sns.heatmap(train_e[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr(), fmt = ".2f", cmap = "coolwarm")
g = sns.catplot(x="CatAge", kind="count", palette="ch:.25", data=train_e)
g = sns.kdeplot(train_e["Age"][(train_e["Survived"] == 0) & (train_e["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train_e["Age"][(train_e["Survived"] == 1) & (train_e["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived", "Survived"])
g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train_e,
                   height=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
# Explore Pclass vs Embarked 
g = sns.catplot("Pclass", col="Embarked",  data=train_e,
                   height=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
def encode_features(df):
    features = [
        'Ticket', 'Fare', 'Cabin',
        'Embarked', 'FamilySize', 'IsAlone',
        'CatAge', 'Age', 'Sex', 'Firstname',
        'Secondname', 'Nameprefix', 'Altername'
    ]
    df = df.drop('Name', axis=1)
    for feature in features:
        le = preprocessing.LabelEncoder()
        #le = le.fit(df[feature])
        df[feature] = le.fit_transform(df[feature].astype(str))
    return df
train_ec = encode_features(train_e)

y_train = train_ec["Survived"]
X_train = train_ec.drop(labels = ["Survived"], axis = 1)

test_ec = encode_features(test_e)
print(X_train.shape, y_train.shape, test_ec.shape)
kfold = StratifiedKFold(n_splits=10)
rand_st = 2
classifiers = [
    KNeighborsClassifier(),
    SVC(random_state=rand_st),
    #LinearRegression(),
    LogisticRegression(random_state = rand_st),
    DecisionTreeClassifier(random_state=rand_st),
    RandomForestClassifier(random_state=rand_st),
    GradientBoostingClassifier(random_state=rand_st),
    AdaBoostClassifier(DecisionTreeClassifier(random_state=rand_st),random_state=rand_st,learning_rate=0.1)
]
cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []

for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":[
    "KNeighboors","SVC","LogisticRegression","DecisionTree","RandomForest","GradientBoosting","AdaBoost"
]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
ids = test_ec["PassengerId"]
cl = GradientBoostingClassifier(random_state=rand_st).fit(X_train, y_train)
survived = pd.Series(cl.predict(test_ec), name="Survived")
res = pd.concat([ids, survived], axis=1)

res.to_csv("predicted.csv", index=False)