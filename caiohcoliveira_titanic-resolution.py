import numpy as np

import pandas as pd

from sklearn.preprocessing import Imputer

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.offline

from plotly.offline import init_notebook_mode

import plotly.graph_objs as go
train = pd.read_csv('../input/titanic/train.csv')

train['set'] = 'train'

test = pd.read_csv('../input/titanic/test.csv')

test['Survived'] = 0

test['set'] = 'test'
train = train[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'set']]

test = test[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'set']]



df = train.append(test)

df.head()
df["Name"] = df["Name"].str.replace(r"\"", "")
cabin_dummies = df['Cabin'].str.extract(r'(?P<cabin_prefx>[a-zA-Z]*?)(?P<cabin_sufx>[0-9]+)$')

name_dummies = df['Name'].str.extract(r'^(?P<Family_Name>[a-zA-Z]+)\s*,\s*(?P<pronoun>[a-zA-Z]+)\s*\.\s*(?P<First_Name>[\s*\w()]*)')

df["prefix_ticket"] = df["Ticket"].str.extract(r"^([A-Za-z.-\/ ]+) ?")

df["prefix_ticket"] = df["prefix_ticket"].str.replace("[\/., ]", "")

df["prefix_ticket"] = df["prefix_ticket"].str.upper()



df = pd.concat([df,cabin_dummies], axis=1)

df = pd.concat([df,name_dummies], axis=1)
family_size = pd.DataFrame(df['Family_Name'].value_counts()).reset_index()

family_size.columns = ["Family_Name", "Family_Size"]

df = df.merge(family_size, how="left", left_on='Family_Name', right_on='Family_Name')
df = df.fillna(0)
from sklearn import preprocessing

def create_dummies( df, colname ):

    col_dummies = pd.get_dummies(df[colname], prefix=colname)

    col_dummies.drop(col_dummies.columns[0], axis=1, inplace=True)

    df = pd.concat([df, col_dummies], axis=1)

    df.drop( colname, axis = 1, inplace = True )

    return df
dummizar = ['Pclass','Sex','Parch','pronoun','cabin_prefx', 'Family_Name', 'prefix_ticket']



df_antigo = df



for i in dummizar:

    df = create_dummies(df, i)
df.head()
df_antigo_test = df_antigo[df_antigo["set"] == "test"]

df_antigo = df_antigo[df_antigo["set"] == "train"]



df_test = df[df["set"] == "test"]

df = df[df["set"] == "train"]
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint8']

df_features = df.select_dtypes(include=numerics)

display(df_features.head())



numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint8']

df_features_test = df_test.select_dtypes(include=numerics)

display(df_features_test.head())
plt.figure(figsize=(15,8))

sns.countplot(df_antigo["Family_Size"], hue=df_antigo["Survived"], orient="h")
plt.figure(figsize=(20,10))

sns.countplot(y=df_antigo["pronoun"], hue=df_antigo["Survived"], orient="h")
plt.figure(figsize=(15,10))

sns.countplot(y=df_antigo["prefix_ticket"], hue=df_antigo["Survived"], orient="h")
plt.clf()

fig, axes = plt.subplots(1, 2, figsize=(20,5))



df_antigo_male = df_antigo[df_antigo["Sex"] == "male"]

df_antigo_female = df_antigo[df_antigo["Sex"] == "female"]



sns.distplot( df_antigo_male[df_antigo_male["Survived"]==1]["Age"], hist=False, color="skyblue", label="Male Survived", ax=axes[0])

sns.distplot( df_antigo_male[df_antigo_male["Survived"]==0]["Age"], hist=False, color="red", label="Male Died", ax=axes[0])



sns.distplot( df_antigo_female[df_antigo_female["Survived"]==0]["Age"] , hist=False, color="red", label="Female Died", ax=axes[1])

sns.distplot( df_antigo_female[df_antigo_female["Survived"]==1]["Age"] , hist=False, color="skyblue", label="Female Survived", ax=axes[1])



plt.legend()

fig, axes = plt.subplots(1, 2, figsize=(18, 6))



boxen = sns.boxenplot(x='Survived', y='Age', hue='Sex', data=df_antigo, ax=axes[0]);

plt.setp(boxen.artists, alpha=.5, linewidth=2, edgecolor="k");



strip = sns.stripplot(x='Survived', y='Age', hue='Sex', data=df_antigo, ax=axes[1]);

plt.setp(strip.artists, alpha=.5, linewidth=2, edgecolor="k");



plt.xticks(rotation=45);
quantitative_features_list = ["Sex_male","pronoun_Mr","Pclass_3","Fare","pronoun_Mrs","pronoun_Miss","Age","SibSp","Parch_1","cabin_prefx_B","cabin_prefx_D","cabin_prefx_E","Pclass_2","Parch_2","pronoun_Master","cabin_prefx_C","prefix_ticket_A","prefix_ticket_FCC","prefix_ticket_PC"]

df_quantitative_values = df[quantitative_features_list]

df_quantitative_values.head()
corr = df_quantitative_values.corr().round(1)

plt.figure(figsize=(len(corr)/2, len(corr)/2))



sns.heatmap(corr, cmap='BrBG', vmax=1.0, vmin=-1.0, center=0, annot=True, square=True, cbar=False);
df_features.head()
X_train = df_features.iloc[:, 2:]

y_train = df_features.iloc[:, 1]
display(X_train.head())

display(y_train.head())
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=10, shuffle=True)

kf.get_n_splits(X_train)

print(kf)
x_train_list, y_train_list, x_test_list, y_test_list = list(), list(), list(), list()

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

accuracy_scores = []

precision_scores = []

recall_scores = []

f1_scores = []

feature_importances = []
x_train_list, y_train_list, x_test_list, y_test_list = list(), list(), list(), list()

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

accuracy_scores = []

precision_scores = []

recall_scores = []

f1_scores = []

feature_importances = []



from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()



for train_index,test_index in kf.split(X_train,y_train):

    

    X_train_kf, y_train_kf = X_train.iloc[train_index], y_train.iloc[train_index]

    X_test_kf, y_test_kf = X_train.iloc[test_index], y_train.iloc[test_index]



    res = classifier.fit(X_train_kf, y_train_kf)

    y_pred_kf = classifier.predict(X_test_kf)

    

    accuracy_scores.append(accuracy_score(y_test_kf, y_pred_kf))

    precision_scores.append(precision_score(y_test_kf, y_pred_kf))

    recall_scores.append(recall_score(y_test_kf, y_pred_kf))

    f1_scores.append(f1_score(y_test_kf, y_pred_kf))
fig, axes = plt.subplots(1, 1, figsize=(20,7))

axes.set_title('Logistic Regression')

axes.set(ylim=(0.5, 1))

df_results = pd.DataFrame(list(zip(accuracy_scores, precision_scores, recall_scores, f1_scores)), columns=["Accuracy", "Precision", "Recall", "F1-score"])

display(df_results.describe())

display(sns.lineplot(data=df_results, palette="husl", linewidth=3))
x_train_list, y_train_list, x_test_list, y_test_list = list(), list(), list(), list()

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

accuracy_scores = []

precision_scores = []

recall_scores = []

f1_scores = []

feature_importances = []



from sklearn.naive_bayes import BernoulliNB

nb = BernoulliNB()



for train_index,test_index in kf.split(X_train,y_train):

    X_train_kf, y_train_kf = X_train.iloc[train_index], y_train.iloc[train_index]

    X_test_kf, y_test_kf = X_train.iloc[test_index], y_train.iloc[test_index]

    

    res = nb.fit(X_train_kf, y_train_kf)

#    importances = list(nb.feature_importances_)

    y_pred_kf = nb.predict(X_test_kf)

    

#    feature_importances.append(importances)

    accuracy_scores.append(accuracy_score(y_test_kf, y_pred_kf))

    precision_scores.append(precision_score(y_test_kf, y_pred_kf))

    recall_scores.append(recall_score(y_test_kf, y_pred_kf))

    f1_scores.append(f1_score(y_test_kf, y_pred_kf))
fig, axes = plt.subplots(1, 1, figsize=(20,7))

axes.set_title('Naive Bayes')

axes.set(ylim=(0.5, 1))

df_results = pd.DataFrame(list(zip(accuracy_scores, precision_scores, recall_scores, f1_scores)), columns=["Accuracy", "Precision", "Recall", "F1-score"])

display(df_results.describe())

display(sns.lineplot(data=df_results, palette="husl", linewidth=3))
x_train_list, y_train_list, x_test_list, y_test_list = list(), list(), list(), list()

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

accuracy_scores = []

precision_scores = []

recall_scores = []

f1_scores = []

feature_importances = []



# Import the model we are using

from sklearn.ensemble import RandomForestClassifier

# Instantiate model with 1000 decision trees

rf = RandomForestClassifier(n_estimators = 2000, max_depth=5)



for train_index,test_index in kf.split(X_train,y_train):

    #print("TRAIN:", train_index, "TEST:", test_index)

    

    X_train_kf, y_train_kf = X_train.iloc[train_index], y_train.iloc[train_index]

    X_test_kf, y_test_kf = X_train.iloc[test_index], y_train.iloc[test_index]

    

    res = rf.fit(X_train_kf, y_train_kf)

    importances = list(rf.feature_importances_)

    y_pred_kf = rf.predict(X_test_kf)

    

    feature_importances.append(importances)

    accuracy_scores.append(accuracy_score(y_test_kf, y_pred_kf))

    precision_scores.append(precision_score(y_test_kf, y_pred_kf))

    recall_scores.append(recall_score(y_test_kf, y_pred_kf))

    f1_scores.append(f1_score(y_test_kf, y_pred_kf))
fig, axes = plt.subplots(1, 1, figsize=(20,7))

axes.set_title('Random Forest')

axes.set(ylim=(0.5, 1))

df_results = pd.DataFrame(list(zip(accuracy_scores, precision_scores, recall_scores, f1_scores)), columns=["Accuracy", "Precision", "Recall", "F1-score"])

display(df_results.describe())

display(sns.lineplot(data=df_results, palette="husl", linewidth=3))
fig, axes = plt.subplots(1, 1, figsize=(15,10))

feature_list = X_train.columns

df_features_importances = pd.DataFrame(list(zip(feature_list, importances)), columns=["feature", "importance"])

df_features_importances = df_features_importances.sort_values("importance", ascending=False)

plot_features = df_features_importances.head(20)

sns.barplot(y=plot_features["feature"], x=plot_features["importance"], palette="rocket")
relevant_features = df_features_importances.iloc[:20, 0].values
x_train_list, y_train_list, x_test_list, y_test_list = list(), list(), list(), list()

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

accuracy_scores = []

precision_scores = []

recall_scores = []

f1_scores = []

feature_importances = []



X_train_fi = X_train[relevant_features]



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 2000, max_depth=5)



for train_index,test_index in kf.split(X_train_fi,y_train):

    #print("TRAIN:", train_index, "TEST:", test_index)

    

    X_train_kf, y_train_kf = X_train_fi.iloc[train_index], y_train.iloc[train_index]

    X_test_kf, y_test_kf = X_train_fi.iloc[test_index], y_train.iloc[test_index]

    

    res = rf.fit(X_train_kf, y_train_kf)

    importances = list(rf.feature_importances_)

    y_pred_kf = rf.predict(X_test_kf)

    

    feature_importances.append(importances)

    accuracy_scores.append(accuracy_score(y_test_kf, y_pred_kf))

    precision_scores.append(precision_score(y_test_kf, y_pred_kf))

    recall_scores.append(recall_score(y_test_kf, y_pred_kf))

    f1_scores.append(f1_score(y_test_kf, y_pred_kf))
fig, axes = plt.subplots(1, 1, figsize=(20,7))

axes.set_title('Random Forest')

axes.set(ylim=(0.5, 1))

df_results = pd.DataFrame(list(zip(accuracy_scores, precision_scores, recall_scores, f1_scores)), columns=["Accuracy", "Precision", "Recall", "F1-score"])

display(df_results.describe())

display(sns.lineplot(data=df_results, palette="husl", linewidth=3))
fig, axes = plt.subplots(1, 1, figsize=(15,10))

feature_list = X_train.columns

df_features_importances = pd.DataFrame(list(zip(feature_list, importances)), columns=["feature", "importance"])

df_features_importances = df_features_importances.sort_values("importance", ascending=False)

sns.barplot(y=df_features_importances["feature"], x=df_features_importances["importance"], palette="rocket")
x_train_list, y_train_list, x_test_list, y_test_list = list(), list(), list(), list()

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

accuracy_scores = []

precision_scores = []

recall_scores = []

f1_scores = []

feature_importances = []



X_train_fi = X_train



from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=1000, max_depth=2)



for train_index,test_index in kf.split(X_train_fi,y_train):

    X_train_kf, y_train_kf = X_train_fi.iloc[train_index], y_train.iloc[train_index]

    X_test_kf, y_test_kf = X_train_fi.iloc[test_index], y_train.iloc[test_index]

    

    res = gb.fit(X_train_kf, y_train_kf)

    importances = list(gb.feature_importances_)

    y_pred_kf = gb.predict(X_test_kf)

    

    feature_importances.append(importances)

    accuracy_scores.append(accuracy_score(y_test_kf, y_pred_kf))

    precision_scores.append(precision_score(y_test_kf, y_pred_kf))

    recall_scores.append(recall_score(y_test_kf, y_pred_kf))

    f1_scores.append(f1_score(y_test_kf, y_pred_kf))
fig, axes = plt.subplots(1, 1, figsize=(20,7))

axes.set_title('Gradient Boosting')

axes.set(ylim=(0.5, 1))

df_results = pd.DataFrame(list(zip(accuracy_scores, precision_scores, recall_scores, f1_scores)), columns=["Accuracy", "Precision", "Recall", "F1-score"])

display(df_results.describe())

display(sns.lineplot(data=df_results, palette="husl", linewidth=3))
fig, axes = plt.subplots(1, 1, figsize=(15,10))

feature_list = X_train.columns



df_vamos_ver = pd.DataFrame(feature_importances, columns=feature_list)

df_features_importances = pd.DataFrame(df_vamos_ver.describe().iloc[1,:]).reset_index()

df_features_importances.columns=["feature", "importance"]



df_features_importances = df_features_importances.sort_values("importance", ascending=False)

plot_features = df_features_importances.head(20)

sns.barplot(y=plot_features["feature"], x=plot_features["importance"], palette="rocket")
relevant_features = df_features_importances.iloc[:10, 0].values

print(relevant_features)
x_train_list, y_train_list, x_test_list, y_test_list = list(), list(), list(), list()

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

accuracy_scores = []

precision_scores = []

recall_scores = []

f1_scores = []

feature_importances = []



X_train_fi = X_train[relevant_features]



from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=1000, max_depth=2, random_state=50)



for train_index,test_index in kf.split(X_train_fi,y_train):

    X_train_kf, y_train_kf = X_train_fi.iloc[train_index], y_train.iloc[train_index]

    X_test_kf, y_test_kf = X_train_fi.iloc[test_index], y_train.iloc[test_index]

    

    res = gb.fit(X_train_kf, y_train_kf)

    importances = list(gb.feature_importances_)

    y_pred_kf = gb.predict(X_test_kf)

    

    feature_importances.append(importances)

    accuracy_scores.append(accuracy_score(y_test_kf, y_pred_kf))

    precision_scores.append(precision_score(y_test_kf, y_pred_kf))

    recall_scores.append(recall_score(y_test_kf, y_pred_kf))

    f1_scores.append(f1_score(y_test_kf, y_pred_kf))
fig, axes = plt.subplots(1, 1, figsize=(20,7))

axes.set_title('Gradient Boosting Filtered 10 best features')

axes.set(ylim=(0.5, 1))

df_results = pd.DataFrame(list(zip(accuracy_scores, precision_scores, recall_scores, f1_scores)), columns=["Accuracy", "Precision", "Recall", "F1-score"])

display(df_results.describe())

display(sns.lineplot(data=df_results, palette="husl", linewidth=3))
fig, axes = plt.subplots(1, 1, figsize=(15,10))

feature_list = X_train_fi.columns

df_features_importances = pd.DataFrame(list(zip(feature_list, importances)), columns=["feature", "importance"])

df_features_importances = df_features_importances.sort_values("importance", ascending=False)

plot_features = df_features_importances.head(20)

sns.barplot(y=plot_features["feature"], x=plot_features["importance"], palette="rocket")
print(relevant_features)
X_train_fi = X_train[relevant_features]

X_train_fi.head()
classifier = gb

classifier.fit(X_train_fi, y_train)
X_valid = df_features_test[relevant_features]

Ids = df_features_test.iloc[:, 0].reset_index()
X_valid.head()
len(X_valid)
y_pred = classifier.predict(X_valid)

y_pred_df = pd.DataFrame(y_pred)



submit_base = pd.DataFrame(pd.concat([Ids["PassengerId"],y_pred_df], axis=1, ignore_index=True))

submit_base.columns = ["PassengerId","Survived"]

submit_base = submit_base.sort_values("PassengerId")

display(submit_base.head())
submit_base.to_csv("submission.csv", index=False)