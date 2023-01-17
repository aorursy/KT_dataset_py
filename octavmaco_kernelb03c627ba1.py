# Exemplu de analiza exploratorie a datelor in Jupyter Notebook

# Pasul 1 - Importul pachetelor și al bibliotecilor necesare analizei



import pandas as pd

import numpy as np



from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.tree import export_graphviz



from matplotlib import pyplot as plt

import seaborn as sns



import graphviz

import imageio

import pydotplus

import io

import scipy

import scipy.misc



%matplotlib inline
# importul datelor din fisierul .csv

data = pd.read_csv('../input/Date_Spotify.csv')

type(data)
data.head()
data.describe()
data.info()
train, test = train_test_split(data, test_size = 0.15)

print("Dimensiunea setului de antrenare: {}; Dimensiunea setului de test: {}".format(len(train), len(test)))
# modificarea culorilor cu ajutorul bibliotecii seaborn

red_blue = ["#19B5FE", "#EF4836"]

palette = sns.color_palette(red_blue)

sns.set_palette(palette)

sns.set_style("white")
# separarea datelor coloanei tempo in functie de preferinte utilizatorului, respectiv coloana target

poz_tempo = data[data['target'] == 1 ]['tempo']

neg_tempo = data[data['target'] == 0 ]['tempo']

poz_dance = data[data['target'] == 1 ]['danceability']

neg_dance = data[data['target'] == 0 ]['danceability']

poz_duration = data[data['target'] == 1 ]['duration_ms']

neg_duration = data[data['target'] == 0 ]['duration_ms']

poz_loudness = data[data['target'] == 1 ]['loudness']

neg_loudness = data[data['target'] == 0 ]['loudness']

poz_valence = data[data['target'] == 1 ]['valence']

neg_valence = data[data['target'] == 0 ]['valence']

poz_energy = data[data['target'] == 1 ]['energy']

neg_energy = data[data['target'] == 0 ]['energy']

poz_acousticness = data[data['target'] == 1 ]['acousticness']

neg_acousticness = data[data['target'] == 0 ]['acousticness']

poz_key = data[data['target'] == 1 ]['key']

neg_key = data[data['target'] == 0 ]['key']

poz_instrumentalness = data[data['target'] == 1 ]['instrumentalness']

neg_instrumentalness = data[data['target'] == 0 ]['instrumentalness']
# reprezentarea grafica a celor doua serii de mai sus prin histograme

fig = plt.figure(figsize=(12, 8))

plt.title("Preferintele utilizatorului on functie de tempoul melodiei")

poz_tempo.hist(alpha = 0.7, bins = 30, label='pozitiv')

neg_tempo.hist(alpha = 0.7, bins = 30, label='negativ')

plt.legend(loc = "upper right")
fig2 = plt.figure(figsize=(15, 15))



#tempo

ax3 = fig2.add_subplot(331)

ax3.set_xlabel('Tempo')

ax3.set_ylabel('Frecventa')

ax3.set_title('Distributia preferintelor privind tempoul melodiilor')

poz_tempo.hist(alpha=0.5, bins=30)

ax4 = fig2.add_subplot(331)

neg_tempo.hist(alpha=0.5, bins=30)



#dansabilitate

ax5 = fig2.add_subplot(332)

ax5.set_xlabel('Dansabilitate')

ax5.set_ylabel('Frecventa')

ax5.set_title('Distributia preferintelor privind dansabilitatea melodiilor')

poz_dance.hist(alpha=0.5, bins=30)

ax6 = fig2.add_subplot(332)

neg_dance.hist(alpha=0.5, bins=30)



#durata

ax7 = fig2.add_subplot(333)

ax7.set_xlabel('Durată')

ax7.set_ylabel('Frecvență')

ax7.set_title('Distributia preferintelor privind durata melodiilor')

poz_duration.hist(alpha=0.5, bins=30)

ax8 = fig2.add_subplot(333)

neg_duration.hist(alpha=0.5, bins=30)



#intensitate

ax9 = fig2.add_subplot(334)

ax9.set_xlabel('Intensitate')

ax9.set_ylabel('Frecvență')

ax9.set_title('Distributia preferintelor privind intensitatea melodiilor')

poz_loudness.hist(alpha=0.5, bins=30)

ax10 = fig2.add_subplot(334)

neg_loudness.hist(alpha=0.5, bins=30)



#valență

ax11 = fig2.add_subplot(335)

ax11.set_xlabel('Valență')

ax11.set_ylabel('Frecvență')

ax11.set_title('Distributia preferintelor privind valența melodiilor')

poz_valence.hist(alpha=0.5, bins=30)

ax12 = fig2.add_subplot(335)

neg_valence.hist(alpha=0.5, bins=30)



#energie

ax13 = fig2.add_subplot(336)

ax13.set_xlabel('Energie')

ax13.set_ylabel('Frecvență')

ax13.set_title('Distributia preferintelor privind energia melodiilor')

poz_energy.hist(alpha=0.5, bins=30)

ax14 = fig2.add_subplot(336)

neg_energy.hist(alpha=0.5, bins=30)



#acustică

ax15 = fig2.add_subplot(337)

ax15.set_xlabel('Acustică')

ax15.set_ylabel('Frecvență')

ax15.set_title('Distributia preferintelor privind acustica melodiilor')

poz_acousticness.hist(alpha=0.5, bins=30)

ax16 = fig2.add_subplot(337)

neg_acousticness.hist(alpha=0.5, bins=30)



#cheie

ax17 = fig2.add_subplot(338)

ax17.set_xlabel('Cheie')

ax17.set_ylabel('Frecvență')

ax17.set_title('Distributia preferintelor privind cheia melodiilor')

poz_key.hist(alpha=0.5, bins=30)

ax18 = fig2.add_subplot(338)

neg_key.hist(alpha=0.5, bins=30)



#instrumentație

ax19 = fig2.add_subplot(339)

ax19.set_xlabel('Cheie')

ax19.set_ylabel('Instrumentație')

ax19.set_title('Distributia preferintelor privind instrumentația melodiilor')

poz_instrumentalness.hist(alpha=0.5, bins=30)

ax20 = fig2.add_subplot(339)

neg_instrumentalness.hist(alpha=0.5, bins=30)
c = DecisionTreeClassifier(min_samples_split=100)
caracteristici = ["danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness", "key", "speechiness", "duration_ms"]
X_antrenare = train[caracteristici]

Y_antrenare = train["target"]



X_test = train[caracteristici]

Y_test = train["target"]
dt = c.fit(X_antrenare, Y_antrenare)
def arata_arbore(tree, caracteristici, cale):

    f = io.StringIO()

    export_graphviz(tree, out_file = f, feature_names = caracteristici)

    pydotplus.graph_from_dot_data(f.getvalue()).write_png(cale)

    img = imageio.imread(cale)

    #img = misc.imread(cale)

    plt.rcParams["figure.figsize"] = (20, 20)

    plt.imshow(img)

    # arata_arbore(dt, caracteristici, 'arbore_decizie_01.png')
arata_arbore(dt, caracteristici, 'arbore_decizie_01.png')
y_pred = c.predict(X_test)
pd.options.display.max_seq_items = 2000
print(y_pred)
from sklearn.metrics import accuracy_score

score = accuracy_score(Y_test, y_pred) * 100
print("Acuratețea utilizând arborii decizionali: ", round(score, 1), "%")