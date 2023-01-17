import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
pk = pd.read_csv("../input/pokemon.csv")
pk.head()
pk.columns
pk = pk[pk["capture_rate"] != "30 (Meteorite)255 (Core)"]

pk.capture_rate = pk.capture_rate.astype(np.int64)
pk["height_m"] = pk["height_m"].fillna(pk["height_m"].median())

pk["percentage_male"] = pk["percentage_male"].fillna(-1)

pk["weight_kg"] = pk["weight_kg"].fillna(pk["weight_kg"].median())

pk.describe()
pk["type1"].value_counts()
capture_type = pk[["capture_rate", "type1"]]

capture_type = capture_type.groupby("type1").median().sort_values("capture_rate", ascending=False)

capture_type.plot(kind="bar", title="Capture rate for each (Primary) type")

plt.show()
best_against = pk.groupby("type1").median()

best_against = best_against[['against_bug', 'against_dark', 'against_dragon',

       'against_electric', 'against_fairy', 'against_fight', 'against_fire',

       'against_flying', 'against_ghost', 'against_grass', 'against_ground',

       'against_ice', 'against_normal', 'against_poison', 'against_psychic',

       'against_rock', 'against_steel', 'against_water']].transpose()



best_against.style.highlight_max()
best_against.sum().sort_values(ascending=False)
legendary_type = pk[["type1", "is_legendary"]]

legendaries = legendary_type[legendary_type["is_legendary"] == 1]
chance_of_legend = legendaries.groupby("type1").count() / legendary_type.groupby("type1").count()

chance_of_legend = chance_of_legend.sort_values("is_legendary", ascending=False)

chance_of_legend.plot(kind="bar", title="Proportion of legendaries Pokemon of each (Primary) type")

plt.show()
pokemons = pk.groupby("type1").median()

pokemons_height_weight = pokemons[["height_m", "weight_kg"]]



plt.rcParams['figure.figsize'] = [15, 8]



ax = pokemons.plot.scatter(x='weight_kg', y='height_m')

for i, txt in enumerate(pokemons.index):

    ax.annotate(txt, (pokemons.weight_kg.iat[i],pokemons.height_m.iat[i]))

plt.show()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn import preprocessing



def remove_unnecessary_features(df):

    del df["abilities"]

    del df["name"]

    del df["japanese_name"]

    del df["classfication"]

    del df["type2"]

    

def label_encode(df, column):

    label_encoder = preprocessing.LabelEncoder()

    label_encoder.fit(df[column])

    df[column] = label_encoder.transform(df[column])

    return label_encoder



X = pk.copy()

X.drop([25, 149, 150]) # Let's remove some Pokemons to try and predict them later

remove_unnecessary_features(X)

type_encoder = label_encode(X, "type1")



y = X.copy()["is_legendary"]

del X["is_legendary"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
classifier = RandomForestClassifier(n_estimators=5)

classifier.fit(X_train, y_train)
import seaborn as sns



cm = sns.light_palette("green", as_cmap=True)



feat_importance = pd.DataFrame([dict(zip(X_train.columns, classifier.feature_importances_))]).transpose().sort_values(0, ascending=False)

feat_importance.style.background_gradient(cmap=cm)
y_hat = classifier.predict(X_test)
from sklearn.metrics import f1_score

f1_score(y_test, y_hat)
X_predict = pk.loc[[149, 150]]

remove_unnecessary_features(X_predict)

type_encoder = label_encode(X_predict, "type1")

del X_predict["is_legendary"]
classifier.predict(X_predict)
X_predict = pk.loc[[25]]

remove_unnecessary_features(X_predict)

type_encoder = label_encode(X_predict, "type1")

del X_predict["is_legendary"]

classifier.predict(X_predict)