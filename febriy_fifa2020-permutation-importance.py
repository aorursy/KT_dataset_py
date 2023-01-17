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
fifa_20 = pd.read_csv('../input//fifa-20-complete-player-dataset/players_20.csv')
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

fifa_20.columns.tolist()
fifa_20['nation_position'].unique()
national_team_players = fifa_20[fifa_20['nation_position'].notna()][["short_name","age","nationality","overall","value_eur"]]
national_team_stats = national_team_players.groupby('nationality').mean().sort_values(by='overall',ascending=True)

national_team_stats
plt.figure(figsize=(20,10))

national_team_stats['overall'].plot()

plt.xticks(np.arange(len(national_team_stats.index)), national_team_stats.index, rotation=90)

plt.show()
plt.figure(figsize=(20,10))

national_team_stats.sort_values(by='value_eur')['value_eur'].plot()

plt.xticks(np.arange(len(national_team_stats.index)), national_team_stats.index, rotation=90)

plt.show()
plt.figure(figsize=(20,10))

national_team_stats.sort_values(by='age')['age'].plot()

plt.xticks(np.arange(len(national_team_stats.index)), national_team_stats.index, rotation=90)

plt.show()
ax = national_team_players["age"].plot.hist(bins=20, alpha=0.5, figsize=(20,10))
young_national = fifa_20[fifa_20['nation_position'].notna()].query("age<=21")[["short_name","age","nationality","overall","potential","value_eur"]]
#who are the top young players?

young_national.sort_values(by='overall',ascending=True,inplace=True)

young_national.index = young_national.short_name



plt.figure(figsize=(20,10))

young_national['overall'].plot()

plt.xticks(np.arange(len(young_national.index)), young_national.index, rotation=90)

plt.show()
young_team_stats = young_national.groupby('nationality').mean().sort_values(by='overall',ascending=True)

#young_team_stats
plt.figure(figsize=(20,10))

young_team_stats['overall'].plot()

plt.xticks(np.arange(len(young_team_stats.index)), young_team_stats.index, rotation=90)

plt.show()
chosen_cols = ["age","height_cm","weight_kg","value_eur","preferred_foot","work_rate","body_type",

              "team_position","pace","shooting","passing","dribbling","defending",

               'attacking_crossing',

                 'attacking_finishing',

                 'attacking_heading_accuracy',

                 'attacking_short_passing',

                 'attacking_volleys',

                 'skill_dribbling',

                 'skill_curve',

                 'skill_fk_accuracy',

                 'skill_long_passing',

                 'skill_ball_control',

                 'movement_acceleration',

                 'movement_sprint_speed',

                 'movement_agility',

                 'movement_reactions',

                 'movement_balance',

                 'power_shot_power',

                 'power_jumping',

                 'power_stamina',

                 'power_strength',

                 'power_long_shots',

                 'mentality_aggression',

                 'mentality_interceptions',

                 'mentality_positioning',

                 'mentality_vision',

                 'mentality_penalties',

                 'mentality_composure',

                 'defending_marking',

                 'defending_standing_tackle',

                 'defending_sliding_tackle',]
fifa_20_chosen=fifa_20[chosen_cols]

fifa_20_chosen=fifa_20_chosen[fifa_20_chosen["team_position"]!="GK"]

fifa_20_chosen.head()
fifa_20_chosen["body_type"].unique()
fifa_20_chosen['body_type'] = fifa_20_chosen['body_type'].map({"Messi": 'Lean',

                                                               "C. Ronaldo":"Normal",

                                                               "Neymar":"Normal",

                                                               "PLAYER_BODY_TYPE_25":"Normal",

                                                               "Shaqiri":"Stocky",

                                                               "Akinfenwa":"Stocky"})
# new data frame with split value columns 

new = fifa_20_chosen["work_rate"].str.split("/", n = 1, expand = True) 

# making separate first name column from new data frame 

fifa_20_chosen["max_work_rate"]= new[0] 

  

# making separate last name column from new data frame 

fifa_20_chosen["min_work_rate"]= new[1] 

  

# Dropping old Name columns 

fifa_20_chosen.drop(columns =["work_rate"], inplace = True) 
print(__doc__)

import matplotlib.pyplot as plt

import numpy as np



from sklearn.ensemble import RandomForestRegressor

from sklearn.impute import SimpleImputer

from sklearn.inspection import permutation_importance

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
fifa_20_chosen.head()
X=fifa_20_chosen.drop(columns=['value_eur'])

y=fifa_20_chosen[['value_eur']]
rng = np.random.RandomState(seed=42)

X['random_cat'] = rng.randint(3, size=X.shape[0])

X['random_num'] = rng.randn(X.shape[0])



categorical_columns = ['preferred_foot', 'body_type', 'team_position','max_work_rate','min_work_rate', 'random_cat']

numerical_columns = ['age', 'height_cm', 'weight_kg','pace',

                 'shooting', 'passing', 'dribbling', 'defending', 

                 'attacking_crossing',

                 'attacking_finishing',

                 'attacking_heading_accuracy',

                 'attacking_short_passing',

                 'attacking_volleys',

                 'skill_dribbling',

                 'skill_curve',

                 'skill_fk_accuracy',

                 'skill_long_passing',

                 'skill_ball_control',

                 'movement_acceleration',

                 'movement_sprint_speed',

                 'movement_agility',

                 'movement_reactions',

                 'movement_balance',

                 'power_shot_power',

                 'power_jumping',

                 'power_stamina',

                 'power_strength',

                 'power_long_shots',

                 'mentality_aggression',

                 'mentality_interceptions',

                 'mentality_positioning',

                 'mentality_vision',

                 'mentality_penalties',

                 'mentality_composure',

                 'defending_marking',

                 'defending_standing_tackle',

                 'defending_sliding_tackle','random_num']



X = X[categorical_columns + numerical_columns]



X_train, X_test, y_train, y_test = train_test_split(

    X, y, random_state=42)



categorical_pipe = Pipeline([

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])

numerical_pipe = Pipeline([

    ('imputer', SimpleImputer(strategy='mean'))

])



preprocessing = ColumnTransformer(

    [('cat', categorical_pipe, categorical_columns),

     ('num', numerical_pipe, numerical_columns)])



rf = Pipeline([

    ('preprocess', preprocessing),

    ('classifier', RandomForestRegressor(random_state=42))

])

rf.fit(X_train, y_train)
print("RF train accuracy: %0.3f" % rf.score(X_train, y_train))

print("RF test accuracy: %0.3f" % rf.score(X_test, y_test))
ohe = (rf.named_steps['preprocess']

         .named_transformers_['cat']

         .named_steps['onehot'])

feature_names = ohe.get_feature_names(input_features=categorical_columns)

feature_names = np.r_[feature_names, numerical_columns]



tree_feature_importances = (

    rf.named_steps['classifier'].feature_importances_)

sorted_idx = tree_feature_importances.argsort()



y_ticks = np.arange(0, len(feature_names))

fig, ax = plt.subplots(figsize=(15,15))

ax.barh(y_ticks, tree_feature_importances[sorted_idx])

ax.set_yticklabels(feature_names[sorted_idx])

ax.set_yticks(y_ticks)

ax.set_title("Random Forest Feature Importances (MDI)")

fig.tight_layout()

plt.show()
result = permutation_importance(rf, X_test, y_test, n_repeats=10,

                                random_state=42, n_jobs=2)

sorted_idx = result.importances_mean.argsort()



fig, ax = plt.subplots(figsize=(15,15))

ax.boxplot(result.importances[sorted_idx].T,

           vert=False, labels=X_test.columns[sorted_idx])

ax.set_title("Permutation Importances (test set)")

fig.tight_layout()

plt.show()
result = permutation_importance(rf, X_train, y_train, n_repeats=10,

                                random_state=42, n_jobs=2)

sorted_idx = result.importances_mean.argsort()



fig, ax = plt.subplots(figsize=(15,15))

ax.boxplot(result.importances[sorted_idx].T,

           vert=False, labels=X_train.columns[sorted_idx])

ax.set_title("Permutation Importances (train set)")

fig.tight_layout()

plt.show()
fifa_20_corr=fifa_20[["wage_eur",'age', 'height_cm', 'weight_kg','pace',

                 'shooting', 'passing', 'dribbling', 'defending', 

                 'attacking_crossing',

                 'attacking_finishing',

                 'attacking_heading_accuracy',

                 'attacking_short_passing',

                 'attacking_volleys',

                 'skill_dribbling',

                 'skill_curve',

                 'skill_fk_accuracy',

                 'skill_long_passing',

                 'skill_ball_control',

                 'movement_acceleration',

                 'movement_sprint_speed',

                 'movement_agility',

                 'movement_reactions',

                 'movement_balance',

                 'power_shot_power',

                 'power_jumping',

                 'power_stamina',

                 'power_strength',

                 'power_long_shots',

                 'mentality_aggression',

                 'mentality_interceptions',

                 'mentality_positioning',

                 'mentality_vision',

                 'mentality_penalties',

                 'mentality_composure',

                 'defending_marking',

                 'defending_standing_tackle',

                 'defending_sliding_tackle']]
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



sns.set(style="white")



# Compute the correlation matrix

corr = fifa_20_corr.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 12))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center= 0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
from __future__ import print_function

import time

from sklearn.manifold import TSNE

X=fifa_20_corr.dropna()

time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne_results = tsne.fit_transform(X)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
X['tsne-2d-one'] = tsne_results[:,0]

X['tsne-2d-two'] = tsne_results[:,1]
X.head()
plt.figure(figsize=(16,10))

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    palette=sns.color_palette("hls", 10),

    data=X,

    legend="full",

    alpha=0.3

)