# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt



df = pd.read_csv(r'/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')

df['count'] = 1  #Assit col

df = df[df['value_eur'] > 1]  #Exclude player with 0 value

df.head(5)

def top_n_charts(field, n):

    df_club = df.groupby([field]).mean()

    df_club = df_club.sort_values('overall', ascending = False).reset_index()



    ls = df.groupby([field]).sum()

    ls = ls[ls['count'] > 10].index



    df_club = df_club[df_club[field].isin(ls)]



    f, ax = plt.subplots(figsize = (20,5))

    sns.barplot(x = field, y = 'overall', data = df_club.iloc[:n])

    ax.set(ylim = (60,85))



top_n_charts('club', 10)

top_n_charts('nationality', 10)
df['best_pos'] = df['player_positions'].str.split(',').str[0]

df_player = df[df['best_pos']!= 'GK'].copy()



dct_pos = {

    'ST': 'Fwd_Centre',

    'CF': 'Fwd_Centre',

    'LW': 'Fwd_Winger',

    'RW': 'Fwd_Winger',

    'CM': 'Mid_Centre',

    'CAM': 'Mid_Centre',

    'CDM': 'Mid_Centre',

    'LM': 'Mid_Side',

    'RM': 'Mid_Side',

    'CB': 'Back_Centre',

    'LB': 'Back_Side',

    'RB': 'Back_Side',

    'LWB': 'Back_Winger',

    'RWB': 'Back_Winger',

}



df_player['best_pos'] = df_player['best_pos'].map(dct_pos)



s = ['Back_Centre',

 'Back_Side',

 'Back_Winger',

 'Mid_Centre',

 'Mid_Side',

 'Fwd_Centre',

 'Fwd_Winger']
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix



from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.tree import export_graphviz



from sklearn.neural_network import MLPClassifier



def display_confusion_matrix(y, y_hat, label, axs):

    res = confusion_matrix(y, y_hat, labels = label)

    row_sums = res.astype(np.float).sum(axis=1)

    res = res/row_sums[:, np.newaxis]

    res = pd.DataFrame(res, columns = label, index = label)

    sns.heatmap(res, cmap="Blues", annot=True, ax=axs)



#pd.DataFrame(y_test).groupby('best_pos')['best_pos'].count()



cols = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']

for col in cols:

    df_player.loc[:, 'n_'+col] = df_player[col] / df_player['overall']



req_col_basic = ['n_'+col for col in cols]



X = df_player[req_col_basic]

y = df_player['best_pos']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



fig, ax = plt.subplots(ncols=3, figsize=(25,5))



#KNN Classifier

classifier = KNeighborsClassifier(n_neighbors=10)

classifier.fit(X_train, y_train)



display_confusion_matrix(y_test, classifier.predict(X_test), s, ax[0])



#Decision Tree Classifier

model = tree.DecisionTreeClassifier(max_depth=5)

model.fit(X_train, y_train)



display_confusion_matrix(y_test, model.predict(X_test), s, ax[1])



#Neural Net Classifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,

                    hidden_layer_sizes=(7,7,7), 

                    activation='relu',random_state=1)

clf.fit(X_train, y_train)



display_confusion_matrix(y_test, clf.predict(X_test), s, ax[2])



ax[0].set_title('KNN-Classifier')

ax[1].set_title('Decision Tree Classifier')

ax[2].set_title('Neural Net Classifier')

cols = ['attacking_crossing','attacking_finishing','attacking_heading_accuracy',

        'attacking_short_passing','attacking_volleys','skill_dribbling','skill_curve',

        'skill_fk_accuracy','skill_long_passing','skill_ball_control','movement_acceleration',

        'movement_sprint_speed','movement_agility','movement_reactions','movement_balance',

        'power_shot_power','power_jumping','power_stamina','power_strength','power_long_shots',

        'mentality_aggression','mentality_interceptions','mentality_positioning','mentality_vision',

        'mentality_penalties','mentality_composure','defending_marking','defending_standing_tackle',

        'defending_sliding_tackle']



for col in cols:

    df_player['n_'+col] = df_player[col] / df_player['overall']



req_col_detail = ['n_'+col for col in cols]



X = df_player[req_col_detail]

y = df_player['best_pos']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



fig, ax = plt.subplots(ncols=3, figsize=(25,5))



#KNN Classifier

classifier = KNeighborsClassifier(n_neighbors=10)

classifier.fit(X_train, y_train)



display_confusion_matrix(y_test, classifier.predict(X_test), s, ax[0])



#Decision Tree Classifier

model = tree.DecisionTreeClassifier(max_depth=5)

model.fit(X_train, y_train)



display_confusion_matrix(y_test, model.predict(X_test), s, ax[1])



#Neural Net Classifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,

                    hidden_layer_sizes=(7,7,7), 

                    activation='relu',random_state=1)

clf.fit(X_train, y_train)



display_confusion_matrix(y_test, clf.predict(X_test), s, ax[2])



ax[0].set_title('KNN-Classifier')

ax[1].set_title('Decision Tree Classifier')

ax[2].set_title('Neural Net Classifier')

df_analysis = df_player[req_col_basic + ['best_pos']]

tmp = df_analysis.groupby('best_pos').mean()

sns.heatmap(tmp, cmap="Blues", annot=True)