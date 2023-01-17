# Import the necessary libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression

from sklearn import svm



from sklearn.neighbors import KNeighborsClassifier





from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, KFold, train_test_split 

from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve 

from sklearn.metrics import confusion_matrix, classification_report
# plots' parameters with seaborn

sns.set_style("whitegrid")

sns.set_context("notebook", font_scale = 1, rc = {"lines.linewidth": 2, 'font_family': [u'times']})
# Load our dataset

df = pd.read_csv("../input/game-of-thrones/character-predictions.csv")
df.head()
# Check for NANs values

nans = df.isna().sum()

nans
# Mean age

print(df["age"].mean())
# Check which characters have a negative age and it's value.

print(df["name"][df["age"] < 0])

print(df['age'][df['age'] < 0])
#According to https://gameofthrones.fandom.com/wiki/Doreah Doreah is actually around 25

#and 

#accoding to https://gameofthrones.fandom.com/wiki/Rhaego Rhaego was never even born so;

# Replace negative ages

df.loc[1684, "age"] = 25.0

df.loc[1868, "age"] = 0.0
# Mean is correct now

print(df["age"].mean())
# Fill the nans we can

df["age"].fillna(df["age"].mean(), inplace=True)

df.fillna("", inplace=True)
nans = df.isna().sum()

nans
sns.violinplot("isPopular", "isNoble", hue="isAlive", data=df ,split=True).set_title('Noble and Popular vs Mortality')

sns.violinplot("isPopular", "isMarried", hue="isAlive", data=df ,split=True).set_title('Married and Popular vs Mortality')

sns.violinplot("isPopular", "book1", hue="isAlive", data=df ,split=True).set_title('Book_1 and Popular vs Mortality')
# Get all of the culture values in our dataset

set(df['culture'])
# Lots of different names for one culture so lets group them up

cult = {

    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],

    'Ghiscari': ['ghiscari', 'ghiscaricari',  'ghis'],

    'Asshai': ["asshai'i", 'asshai'],

    'Lysene': ['lysene', 'lyseni'],

    'Andal': ['andal', 'andals'],

    'Braavosi': ['braavosi', 'braavos'],

    'Dornish': ['dornishmen', 'dorne', 'dornish'],

    'Myrish': ['myr', 'myrish', 'myrmen'],

    'Westermen': ['westermen', 'westerman', 'westerlands'],

    'Westerosi': ['westeros', 'westerosi'],

    'Stormlander': ['stormlands', 'stormlander'],

    'Norvoshi': ['norvos', 'norvoshi'],

    'Northmen': ['the north', 'northmen'],

    'Free Folk': ['wildling', 'first men', 'free folk'],

    'Qartheen': ['qartheen', 'qarth'],

    'Reach': ['the reach', 'reach', 'reachmen'],

    'Ironborn': ['ironborn', 'ironmen'],

    'Mereen': ['meereen', 'meereenese'],

    'RiverLands': ['riverlands', 'rivermen'],

    'Vale': ['vale', 'valemen', 'vale mountain clans']

}



def get_cult(value):

    value = value.lower()

    v = [k for (k, v) in cult.items() if value in v]

    return v[0] if len(v) > 0 else value.title()

df.loc[:, "culture"] = [get_cult(x) for x in df["culture"]]

#how does culter affect survival

df.loc[:, "culture"] = [get_cult(x) for x in df.culture.fillna("")]

data = df.groupby(["culture", "isAlive"]).count()["S.No"].unstack().copy(deep = True)

data.loc[:, "total"]= data.sum(axis = 1)

p = data[data.index != ""].sort_values("total")[[0, 1]].plot.barh(stacked = True, rot = 0, figsize = (14, 12),)

_ = p.set(xlabel = "No. of Characters", ylabel = "Culture"), p.legend(["Dead", "Alive"], loc = "lower right")
#saving a copy of the dataset just in case

df2 = df.copy(deep=True)
df
#droping columns that are not useful

drop = ["S.No", "plod", "title", "dateOfBirth", "DateoFdeath", "mother", "father", "heir", "house", 

        "spouse", "book2", "book3", "book4", "book5", "isAliveMother", "isAliveFather","isAliveHeir",

        "isAliveSpouse", "popularity", "name"]

df = df.drop(drop, axis=1)
#turning categorical variables into one-hot encoded variables

df = pd.get_dummies(df)
#creating response and explanatory variables 

y=df.isAlive #response

X=df.drop('isAlive', axis=1)
#making logit

lr = LogisticRegression()
lr.fit(X,y)
lr.predict(X)
#model's score is pretty good

lr.score(X,y)
#finding arya's index number

df2[df2["name"]=="Arya Stark"].index
X.iloc[1466]
#finding survival rate which is 0.557

lr.predict_proba([X.iloc[1466]])
lr.predict([X.iloc[1466]])
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif