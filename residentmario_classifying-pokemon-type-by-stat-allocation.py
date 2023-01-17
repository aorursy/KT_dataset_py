import pandas as pd

from pandas.tools.plotting import parallel_coordinates

%matplotlib inline

import matplotlib.pyplot as plt
pokemon = pd.read_csv('../input/Pokemon.csv')
plt.figure(figsize=(12, 5))

parallel_coordinates(pokemon[

        ['Type 1', 'HP', 'Speed', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def']

    ], 'Type 1')
plt.figure(figsize=(12, 5))

parallel_coordinates(pokemon[pokemon['Type 1'].isin(['Steel', 'Fairy'])][

        ['Type 1', 'HP', 'Speed', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def']

    ], 'Type 1',  colormap=lambda t: 'DarkOrchid' if t == 0 else 'DarkGray')
plt.figure(figsize=(12, 5))

parallel_coordinates(pokemon[pokemon['Type 1'].isin(['Psychic', 'Fighting'])][

        ['Type 1', 'HP', 'Speed', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def']

    ], 'Type 1',  colormap=lambda t: 'PaleVioletRed' if t == 0 else 'FireBrick')
plt.figure(figsize=(12, 5))

parallel_coordinates(pokemon[pokemon['Type 1'].isin(['Ghost', 'Psychic'])][

        ['Type 1', 'HP', 'Speed', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def']

    ], 'Type 1', colormap=lambda t: 'PaleVioletRed' if t == 0 else 'Indigo')
plt.figure(figsize=(8, 4))

parallel_coordinates(pokemon.groupby('Type 1').mean().reset_index()[

        ['Type 1', 'HP', 'Speed', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def']

    ], 'Type 1')

ax = plt.gca()

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.3, 1))
def predict_primary_type(hp, speed, attack, defense, sp_atk, sp_def, bound=(25, 25)):

    ret = pokemon

    for v in zip(['HP', 'Speed', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def'], [hp, speed, attack, defense, sp_atk, sp_def]):

        v_low = v[1] - bound[0]

        v_high = v[1] + bound[1]

        ret = ret[(v_low < ret[v[0]]) & (ret[v[0]] < v_high)]

    ret = ret.groupby('Type 1').count()['#']

    return ret / ret.sum()
predict_primary_type(70, 70, 70, 70, 70, 70)
steel_avg = pokemon.groupby('Type 1').mean().ix['Steel'][['HP', 'Speed', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def']].values
predict_primary_type(*steel_avg)
for _type in labels:

    type_avg = pokemon.groupby('Type 1').mean().ix[_type][['HP', 'Speed', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def']].values

    print(_type.ljust(10), predict_primary_type(*type_avg)[_type])
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

steel_or_fairy = pokemon[pokemon['Type 1'].isin(['Steel', 'Fairy'])][features]

type_actual = pokemon[pokemon['Type 1'].isin(['Steel', 'Fairy'])]['Type 1']
df_norm = steel_or_fairy.copy()

df_norm[features] = StandardScaler().fit(steel_or_fairy[features]).transform(steel_or_fairy[features])



steel_or_fairy_pca = PCA()

pca_outcomes = steel_or_fairy_pca.fit_transform(df_norm[features])
steel_or_fairy_pca.explained_variance_ 
first_two_principal_components = pca_outcomes[:,[0,1]]



plt.title("Fairy or Steel Type, Actual")

plt.scatter(first_two_principal_components[:,0], first_two_principal_components[:,1],

            c=['DarkOrchid' if t else 'DarkGray' for t in (type_actual == 'Fairy')])
from sklearn import svm



clf = svm.SVC(kernel='linear', C=1.0).fit(first_two_principal_components, type_actual)



f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,4))

ax1.set_title('Fairy or Steel, Actual')

ax2.set_title('Fairy or Steel, Predicted')

ax1.scatter(first_two_principal_components[:,0], first_two_principal_components[:,1],

            c=['DarkOrchid' if t else 'DarkGray' for t in (type_actual == 'Fairy')])

ax2.scatter(first_two_principal_components[:,0], first_two_principal_components[:,1],

            c=['DarkOrchid' if t else 'DarkGray' for t in (clf.predict(first_two_principal_components) == 'Fairy')])
import matplotlib.patches as mpatches



def pairwise_classify(type_1, type_2):

    features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

    df = pokemon[pokemon['Type 1'].isin([type_1, type_2])][features]

    df_norm = df.copy()

    df_norm[features] = StandardScaler().fit(df[features]).transform(df[features])



    pca_outcomes = PCA().fit_transform(df_norm[features])

    principal_components = pca_outcomes[:,[0,1]]

    

    y = pokemon[pokemon['Type 1'].isin([type_1, type_2])]['Type 1'].values



    clf = svm.SVC(kernel='linear', C=1.0).fit(principal_components, y)

    

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,4))

    ax1.set_title('{0} or {1}, Actual'.format(type_1, type_2))

    ax2.set_title('{0} or {1}, Predicted'.format(type_1, type_2))

    

    ax1.scatter(principal_components[:,0], principal_components[:,1],

            c=['#b4464b' if t else '#4682b4' for t in (y == type_1)], lw = 0, s=40)

    ax2.scatter(principal_components[:,0], principal_components[:,1],

            c=['#b4464b' if t else '#4682b4' for t in (clf.predict(principal_components) == type_1)], lw = 0, s=40)

    

    red_patch = mpatches.Patch(color='#b4464b', label=type_1)

    blue_patch = mpatches.Patch(color='#4682b4', label=type_2)

    ax1.legend(handles=[red_patch, blue_patch])

    ax2.legend(handles=[red_patch, blue_patch])

    

    plt.show()
pairwise_classify('Steel', 'Fairy')
pairwise_classify('Dragon', 'Ice')
pairwise_classify('Fighting', 'Psychic')
pairwise_classify('Bug', 'Ground')
from sklearn import metrics

import numpy as np



def distinguishability(type_1, type_2):

    features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

    df = pokemon[pokemon['Type 1'].isin([type_1, type_2])][features]

    df_norm = df.copy()

    df_norm[features] = StandardScaler().fit(df[features]).transform(df[features])



    pca_outcomes = PCA().fit_transform(df_norm[features])

    principal_components = pca_outcomes[:,[0,1]]

    

    y = pokemon[pokemon['Type 1'].isin([type_1, type_2])]['Type 1'].values



    clf = svm.SVC(kernel='linear', C=1.0).fit(principal_components, y)

        

    conf = metrics.confusion_matrix(y, clf.predict(principal_components))

    

    # If we simply classify all records as one type or the other our classifier has failed to be useful.

    # In that case return np.nan.

    if all(conf[:,0] == [0,0]) or all(conf[:,1] == [0,0]):

        return np.nan

    else:

        return (conf[0][0] + conf[1][1]) / conf.sum()
distinguishability('Dragon', 'Ice')
distinguishability('Bug', 'Ice')
types = np.unique(pokemon['Type 1'])



values = []

for type_1 in types:

    values.append([])

    for type_2 in types:

        if type_1 != type_2:

            values[-1].append(distinguishability(type_1, type_2))

        else:

            values[-1].append(np.nan)
type_distinguishabilities = pd.DataFrame(index=types, columns=types, data=values)

classification_gain = type_distinguishabilities - 0.5
classification_gain
classification_gain.fillna(0).mean()
pairwise_classify('Dragon', 'Normal')
pairwise_classify('Dragon', 'Steel')
pairwise_classify('Steel', 'Flying')
pairwise_classify('Ground', 'Electric')