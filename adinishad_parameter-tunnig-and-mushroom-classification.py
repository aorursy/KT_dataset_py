# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 23) 
import os
print(os.listdir("../input/mushroom-classification"))
data = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
data.head()
data.info()
# copy data
dataset = data.copy()

# rename all values
dataset = dataset.replace({'class' : {'p': 'poisonous', 'e': 'edible'}})
dataset = dataset.replace({'cap-shape' : {'b':'bell','c':'conical','x':'convex','f':'flat','k':'knobbed','s':'sunken'}})
dataset = dataset.replace({'cap-surface' : {'f':'fibrous','g':'grooves','y':'scaly','s':'smooth'}})
dataset = dataset.replace({'cap-color' : {'n':'brown','b':'buff','c':'cinnamon','g':'gray','r':'green',
                                                    'p':'pink','u':'purple','e':'red','w':'white','y':'yellow'}})
dataset = dataset.replace({'bruises' : {'t':'bruises','f':'no'}})
dataset = dataset.replace({'odor' : {'a':'almond','l':'anise','c':'creosote','y':'fishy','f':'foul','m':'musty','n':'none',
                                              'p':'pungent','s':'spicy'}})
dataset = dataset.replace({'gill-attachment' : {'a':'attached','d':'descending','f':'free','n':'notched'}})
dataset = dataset.replace({'gill-spacing' : {'c':'close','w':'crowded','d':'distant'}})
dataset = dataset.replace({'gill-size' : {'b':'broad','n':'narrow'}})
dataset = dataset.replace({'gill-color' : {'k':'black','n':'brown','b':'buff','h':'chocolate','g':'gray','r':'green',
                                                    'o':'orange','p':'pink','u':'purple','e':'red','w':'white','y':'yellow'}})
dataset = dataset.replace({'stalk-shape' : {'e':'enlarging','t':'tapering'}})
dataset = dataset.replace({'stalk-root' : {'b':'bulbous','c':'club','u':'cup','e':'equal','z':'rhizomorphs',
                                                     'r':'rooted','?':"missing"}})
dataset = dataset.replace({'stalk-surface-above-ring' : {'f':'fibrous','y':'scaly','k':'silky','s':'smooth'}})
dataset = dataset.replace({'stalk-surface-below-ring' : {'f':'fibrous','y':'scaly','k':'silky','s':'smooth'}})
dataset = dataset.replace({'stalk-color-above-ring' : {'n':'brown','b':'buff','c':'cinnamon','g':'gray','o':'orange',
                                                                'p':'pink','e':'red','w':'white','y':'yellow'}})
dataset = dataset.replace({'stalk-color-below-ring' : {'n':'brown','b':'buff','c':'cinnamon','g':'gray','o':'orange',
                                                                'p':'pink','e':'red','w':'white','y':'yellow'}})
dataset = dataset.replace({'veil-type' : {'p':'partial','u':'universal'}})
dataset = dataset.replace({'veil-color' : {'n':'brown','o':'orange','w':'white','y':'yellow'}})
dataset = dataset.replace({'ring-number' : {'n':'none','o':'one','t':'two'}})
dataset = dataset.replace({'ring-type' : {'c':'cobwebby','e':'evanescent','f':'flaring','l':'large','n':'none',
                                                   'p':'pendant','s':'sheathing','z':'zone'}})
dataset = dataset.replace({'spore-print-color' : {'k':'black','n':'brown','b':'buff','h':'chocolate','r':'green','o':'orange',
                                                           'u':'purple','w':'white','y':'yellow'}})

dataset = dataset.replace({'population' : {'a':'abundant','c':'clustered','n':'numerous','s':'scattered','v':'several',
                                                     'y':'solitary'}})
dataset = dataset.replace({'habitat' : {'g':'grasses','l':'leaves','m':'meadows','p':'paths','u':'urban','w':'waste','d':'woods'}})
dataset.head()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
sns.heatmap(dataset.isnull())
print(dataset.isnull().sum())
dataset["class"].value_counts()
a = dataset["class"].value_counts()
colors = ["#43b581", "#f04747"]
fig = px.pie(a, values="class", title='Percentage of poisonous and edible', names=a.index, hole=0.3)
fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(colors=colors, line=dict(color='#ffffff', width=2)), textfont_size=20)
fig.show()
plt.subplots(figsize=(10,6))
sns.set_style("whitegrid")
g = sns.countplot(dataset["cap-shape"], hue=dataset["class"], palette="Set2")

plt.legend(loc="upper right")
plt.title("cap shape VS class", fontweight="bold")
for p in g.patches:
    g.annotate('{:2}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()+1), va='bottom', ha="center", fontweight='bold')
# Poisonous
poi = dataset[dataset["class"]=="poisonous"][["cap-color"]]

a = poi["cap-color"].value_counts()
colors = ['brown', 'red', 'gray', 'yellow', 'white', 'buff', 'pink', 'cinnamon']
fig = px.pie(a, values="cap-color", title='Poisonous mushroom according to color', names=a.index, hole=0.3)
fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(colors=colors, line=dict(color='#313131', width=2)), textfont_size=20)
fig.show()

# Edible
edi = dataset[dataset["class"]=="edible"][["cap-color"]]

b = edi["cap-color"].value_counts()
colors = ['brown', 'gray', 'white', 'red', 'yellow', 'pink', 'buff', 'cinnamon', 'purple', 'green']
fig = px.pie(b, values="cap-color", title='Edible mushroom according to color', names=b.index, hole=0.3)
fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(colors=colors, line=dict(color='#313131', width=2)), textfont_size=20)
fig.show()
plt.subplots(figsize=(10,6))
sns.set_style("whitegrid")
g = sns.countplot(dataset["gill-attachment"], hue=dataset["class"], palette="Set2")

plt.legend(loc="upper right")
plt.title("gill attachment VS class", fontweight="bold")
for p in g.patches:
    g.annotate('{:2}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()+1), va='bottom', ha="center", fontweight='bold')
# Poisonous mushroom habitat
poi = dataset[dataset["class"]=="poisonous"][["habitat"]]

a = poi["habitat"].value_counts()
fig = px.pie(a, values="habitat", title='Poisonous mushroom according to habitat', names=a.index, hole=0.3)
fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#313131', width=2)), textfont_size=20)
fig.show()

# Edible mushroom habitat
edi = dataset[dataset["class"]=="edible"][["habitat"]]

b = edi["habitat"].value_counts()
fig = px.pie(b, values="habitat", title='Edible mushroom according to habitat', names=b.index, hole=0.3)
fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#313131', width=2)), textfont_size=20)
fig.show()
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 6))
dataset["ring-number"].value_counts().plot(kind="bar", ax=ax1)
g = sns.countplot(dataset["ring-number"], hue=dataset["class"], ax=ax2, palette="Set2")
plt.suptitle("Ring number VS class", fontweight="bold")
for p in g.patches:
    g.annotate('{:2}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()+1), va='bottom', ha="center")
plt.legend(loc="upper right")
from sklearn.preprocessing import LabelEncoder
lencoder=LabelEncoder()
for col in data.columns:
    dataset[col] = lencoder.fit_transform(dataset[col])
dataset.drop("veil-type", axis=1, inplace=True)

corre = dataset.corr()
plt.subplots(figsize=(16, 8))
sns.heatmap(corre, annot=True)
corre["class"].sort_values(ascending=False)
from sklearn.model_selection import train_test_split
X = dataset.drop(["class"], axis=1)
y = dataset["class"]

seed = 42
test = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
model_params = {
    "svm": {
        "model": SVC(gamma="auto"),
        "params":{
            "C": [1, 10, 20],
            "kernel": ["rbf", "linear"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [1, 5, 10]
        }
    },
    "Logistic Regression":{
        "model": LogisticRegression(solver="liblinear", multi_class="auto"),
        "params": {
            "C": [1, 10, 20]
        }
    },
    "ExtraTree": {
        "model": ExtraTreesClassifier(),
        "params": {
            "n_estimators" : [1, 5, 10]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "criterion": ["gini", "entropy"]
        }
    }
}
scores = []
for model_name, param in model_params.items():
    clf = GridSearchCV(param["model"], param["params"], cv=10, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        "model": model_name,
        "best_score": clf.best_score_,
        "best_params": clf.best_params_,
    })
df = pd.DataFrame(scores, columns=["model", "best_score", "best_params"])
df
model = SVC(C=1, kernel="rbf")

model.fit(X_train, y_train)

predict = model.predict(X_test)

predict
print(classification_report(predict, y_test))
print(confusion_matrix(predict, y_test))
