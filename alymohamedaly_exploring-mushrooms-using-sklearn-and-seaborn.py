import numpy as np

import pandas as pd
df = pd.read_csv("../input/mushrooms.csv")

print(df.shape)

for col in df.columns:

    df.rename(columns={col:col.capitalize()}, inplace=True)   #I just like it capitalized

df.describe()
def SporeNamer(x):

    if x == 'k':

        return 'Black'

    if x == 'n':

        return 'Brown'

    if x == 'b':

        return 'Buff'

    if x == 'h':

        return 'Chocolate'

    if x == 'r':

        return 'Green'

    if x == 'o':

        return 'Orange'

    if x == 'u':

        return 'Purple'

    if x == 'w':

        return 'White'

    return 'Yellow'

def OdorNamer(x):

    if x == 'a':

        return 'Almond'

    if x == 'l':

        return 'Anise'

    if x == 'c':

        return 'Creosote'

    if x == 'y':

        return 'Fishy'

    if x == 'f':

        return 'Foul'

    if x == 'm':

        return 'Musty'

    if x == 'n':

        return 'None'

    if x == 'p':

        return 'Pungent'

    return 'Spicy'

def GillNamer(x):

    if x == 'k':

        return 'Black'

    if x == 'n':

        return 'Brown'

    if x == 'b':

        return 'Buff'

    if x == 'h':

        return 'Chocolate'

    if x == 'r':

        return 'Green'

    if x == 'o':

        return 'Orange'

    if x == 'p':

        return 'Pink'

    if x == 'e':

        return 'Red'

    if x == 'u':

        return 'Purple'

    if x == 'w':

        return 'White'

    return 'Yellow'

def PopNamer(x):

    if x == 'a':

        return 'Abundant'

    if x == 'c':

        return 'Clustered'

    if x == 'n':

        return 'Numerous'

    if x == 's':

        return 'Scattered'

    if x == 'v':

        return 'Several'

    return 'Solitary'
df['Class'] = df['Class'].apply(lambda x: 'Edible' if x == 'e' else 'Poisonous')

df['Bruises'] = (df['Bruises'] == 't')



df['Odor'] = df['Odor'].apply(OdorNamer)

df['Spore-print-color'] = df['Spore-print-color'].apply(SporeNamer)

df['Gill-color'] = df['Population'].apply(GillNamer)

df['Population'] = df['Population'].apply(PopNamer)

df.rename(columns={'Class':'Edible'}, inplace=True)

df.drop('Veil-type',axis=1,  inplace=True)

df.head()
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

dfEncoded = df.apply(lambda col: LE.fit_transform(col))

dfEncoded.head()
dfEncoded.describe()
# Classification

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier



# Modelling Helpers :

from sklearn.model_selection import train_test_split
RFC = RandomForestClassifier(n_estimators=666, random_state=82)

KNN = KNeighborsClassifier(n_neighbors = 1)

BAG = BaggingClassifier(random_state = 222, n_estimators=92)

GradBost = GradientBoostingClassifier(random_state = 15)

ADA = AdaBoostClassifier(random_state = 37)

DT = DecisionTreeClassifier(random_state=12)
x = dfEncoded.copy()

y = x.pop('Edible')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 37)
RFC.fit(x_train,y_train)

RFC_pred = RFC.predict(x_test)

print("accuracy: {} %".format((RFC.score(x_test,y_test)*100)))

for Counter, i in enumerate(RFC.feature_importances_):

    if i > 0.10:

        print("{} makes up {} % of the decision making process".format(x.columns[Counter], ("%.2f" % (i*100))))
DT.fit(x_train,y_train)

DT_pred = DT.predict(x_test)

print("accuracy: "+ str(DT.score(x_test,y_test)*100) + "%")

for Counter, i in enumerate(DT.feature_importances_):

    if i > 0.10:

        print("{} makes up {} % of the decision making process".format(x.columns[Counter], ("%.2f" % (i*100))))
ADA.fit(x_train,y_train)

ADA_pred = ADA.predict(x_test)

print("accuracy: "+ str(ADA.score(x_test,y_test)*100) + "%")

for Counter, i in enumerate(ADA.feature_importances_):

    if i > 0.10:

        print("{} makes up {} % of the decision making process".format(x.columns[Counter], ("%.2f" % (i*100))))
GradBost.fit(x_train,y_train)

GradBost_pred = GradBost.predict(x_test)

print("accuracy: "+ str(("%.2f" %(GradBost.score(x_test,y_test)*100))) + "%")

for Counter, i in enumerate(GradBost.feature_importances_):

    if i > 0.10:

        print("{} makes up {} % of the decision making process".format(x.columns[Counter], ("%.2f" % (i*100))))
BAG.fit(x_train,y_train)

BAG_pred = BAG.predict(x_test)

print("accuracy: "+ str(BAG.score(x_test,y_test)*100) + "%")
KNN.fit(x_train,y_train)

KNN_pred = KNN.predict(x_test)

print("accuracy: "+ str(KNN.score(x_test,y_test)*100) + "%")
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



fig, ax = plt.subplots(figsize=(18,6))

g = sns.countplot(df["Spore-print-color"], ax=ax, data = df[["Spore-print-color", 'Edible']],

                  hue='Edible', palette='hls')
fig, ax = plt.subplots(figsize=(18,6))

g = sns.countplot(df["Odor"], ax=ax, data = df[["Odor", 'Edible']], hue='Edible', palette='hls')
fig, ax = plt.subplots(figsize=(18,6))

g = sns.countplot(df["Gill-color"], ax=ax, data = df[["Gill-color", 'Edible']], hue='Edible', palette='hls')
fig, ax = plt.subplots(figsize=(18,6))

g = sns.countplot(df["Population"], ax=ax, data = df[["Population", 'Edible']], hue='Edible', palette='hls')