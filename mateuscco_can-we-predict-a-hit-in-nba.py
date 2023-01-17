# Libs to deal with tabular data

import numpy as np

import pandas as pd

pd.options.mode.chained_assignment = None



# Plotting packages

import seaborn as sns

sns.axes_style("darkgrid")

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn')



# Machine Learning

from sklearn import preprocessing

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, KFold

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import roc_auc_score

from sklearn.inspection import permutation_importance



# To display stuff in notebook

from IPython.display import display, Markdown



# Misc tqdm.notebook.tqdm

from tqdm.notebook import tqdm

import time
shots = pd.read_csv('../input/nba-shot-logs/shot_logs.csv')
shots.sample(5)
shots.shape
shots.columns = shots.columns.str.lower()

shots.dtypes
shots.isnull().sum()
with pd.option_context('display.max_columns', None):

    display(shots[shots['game_id'] == shots['game_id'].sample().iloc[0]].head(10))
shots['touch_time'].value_counts()
(shots['touch_time'] < 0).sum()
with pd.option_context('display.max_columns', None):

    display(shots.loc[shots['touch_time'] < 0, :].head())
# Convert game clock to seconds

shots['game_clock'] = shots['game_clock'].apply(

    lambda x: 60*int(x.split(':')[0]) + int(x.split(':')[1])

)



# Replacing abnormal values with NaNs

shots.loc[shots['touch_time'] < 0, 'touch_time'] = np.nan



# Converting type of shot (2 or 3 points) to categorical

shots['pts_type'] = (shots['pts_type'] == 3) * 1



# Converting location

shots['location'] = (shots['location'] == 'H') * 1



# Renaming columns

shots = shots.rename(columns = {

    'fgm':'hit',

    'pts_type':'3pts_shot',

    'location':'home_match'

})



# Dropping informative columns (not useful to modelling) as well as 

# future variables which won't be available at predicting time

shots = shots.drop(columns = [

    'game_id',

    'matchup',

    'w',

    'final_margin',

    'closest_defender_player_id',

    'player_id',

    'shot_result',

    'closest_defender',

    'player_name',

    'pts'

])
shots.head(5)
shots.shape
def plot_distribution(col, bins=10):

    shots[col].plot.hist(bins=bins)

    plt.title(col, fontsize=16)

    plt.show()

    

def plot_relationship(x, y):

    sns.boxplot(data = shots, y = y, x = x)

    plt.title('{} vs {}'.format(x, y), fontsize=16)

    plt.show()

    

def show_frequency(x, y):

    joint = pd.crosstab(shots[x], shots[y], margins = True)

    joint = joint / joint.loc['All', 'All']

    display(joint)

    

def plot_scatter(x, y):

    sns.scatterplot(data = shots_scaled, x = x, y = y)

    plt.title('{} vs {}'.format(x, y), fontsize=16)

    plt.show()
ax = shots['hit'].replace({

    0:'Miss',

    1:'Hit'

}).value_counts().plot.bar(rot=0)

ax.set_title('hit', fontsize=16)

plt.show()
ax = shots['home_match'].replace({

    0:'Away',

    1:'Home'

}).value_counts().plot.bar(rot=0)

ax.set_title('home_match', fontsize=16)

plt.show()



ax = shots['period'].value_counts().plot.bar(rot=0)

ax.set_title('period', fontsize=16)

plt.show()



plot_distribution('game_clock', bins=10)
ax = shots['3pts_shot'].replace({

    0:'2 points',

    1:'3 points'

}).value_counts().plot.bar(rot=0)

ax.set_title('3pts_shot', fontsize=16)

plt.show()
plot_distribution('shot_dist', bins=25)
plot_distribution('shot_clock', bins=20)
plot_distribution('close_def_dist', bins=40)
plot_distribution('shot_number', bins=10)

plot_distribution('dribbles', bins=30)

plot_distribution('touch_time', bins=20)
shots.describe()
show_frequency('3pts_shot', 'hit')
plot_relationship('hit', 'close_def_dist')
plot_relationship('hit', 'shot_dist')

plot_relationship('hit', 'touch_time')

plot_relationship('hit', 'dribbles')
plot_relationship('hit', 'shot_clock')

plot_relationship('3pts_shot', 'close_def_dist')
shots_scaled = shots.copy()



# Stardardization

shots_scaled[['shot_clock']] = preprocessing.StandardScaler().fit_transform(shots[['shot_clock']].values)



# Robust scaling

skewed_cols = ['shot_number', 'dribbles', 'touch_time', 'close_def_dist']

shots_scaled[skewed_cols] = preprocessing.RobustScaler().fit_transform(shots[skewed_cols].values)

    

# Min max transformation

min_max_cols = ['period', 'game_clock', 'shot_dist']

shots_scaled[min_max_cols] = preprocessing.MinMaxScaler().fit_transform(shots[min_max_cols].values)



# Filling NaNs with mean

shots_scaled['shot_clock'] = shots_scaled['shot_clock'].fillna(shots_scaled['shot_clock'].mean())

shots_scaled['touch_time'] = shots_scaled['touch_time'].fillna(shots_scaled['touch_time'].median())

shots['shot_clock'] = shots['shot_clock'].fillna(shots['shot_clock'].mean())

shots['touch_time'] = shots['touch_time'].fillna(shots['touch_time'].median())
x_corr = shots_scaled[['shot_number', 'shot_clock', 'dribbles', 'touch_time', 'shot_dist', 'close_def_dist', 'period', 'game_clock']].corr()

corr_mask = np.zeros_like(x_corr, dtype=np.bool)

corr_mask[np.tril_indices_from(corr_mask, k=0)] = True



sns.heatmap(x_corr, mask = corr_mask, annot=True)

plt.show()
plot_scatter('dribbles', 'touch_time')
plot_scatter('shot_dist', 'close_def_dist')
plot_scatter('period', 'shot_number')
# Fitting PCA and showing PVE

pca = PCA(random_state=42).fit(shots_scaled.drop('hit', axis=1).values)

pve = pca.explained_variance_ratio_
plt.plot(range(1, len(pve) + 1), pve.cumsum())

plt.title('Cumulative sum of explained variance', fontsize=16)

plt.xlabel('Components')

plt.ylabel('Percentage (%)')

plt.show()
pca_components = pd.DataFrame(

    pca.components_[:4,:].T,

    index = shots.columns[:-1]

)

sns.heatmap(pca_components, annot=True)
n_clusters = range(2, 41)

kms, inertias = [], []

for n in tqdm(n_clusters):

    ts = time.time()

    km = KMeans(n_clusters=n, random_state=42)

    km.fit(shots_scaled.drop('hit', axis=1).values)

    kms.append(km)

    inertias.append(km.inertia_)
# Checking the within-cluster variation.

plt.plot(range(2,41), inertias)

plt.title('Within-cluster variance', fontsize=16)

plt.xlabel('Number of clusters')

plt.ylabel('Sum of squared distances')

# plt.show()
# Splitting original dataset

X = shots.drop('hit', axis = 1).values

y = shots['hit'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)



# Splitting scaled and transformed by PCA dataset

X_scaled = pca.transform(shots_scaled.drop('hit', axis = 1).values)[:,:4]

x_train_scaled, x_test_scaled, _, _ = train_test_split(X_scaled, y, test_size = 0.2, random_state=42)
def decision_tree_cv(x, y, folds=5):

    cv = KFold(folds, random_state=42, shuffle=True)

    depths = list(range(1, 101))

    scores = np.zeros((len(depths), folds, 2, 2)) #depth, fold, split, metric

    

    for id_split, array_idxs in tqdm(enumerate(cv.split(x))):

        train_index, val_index = array_idxs[0], array_idxs[1]

        x_train, x_val = x[train_index], x[val_index]

        y_train, y_val = y[train_index], y[val_index]

        

        for depth in depths:

            clf = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(x_train, y_train)

            scores[depth - 1, id_split, 0, 0] = clf.score(x_train, y_train)

            scores[depth - 1, id_split, 1, 0] = clf.score(x_val, y_val)

            scores[depth - 1, id_split, 0, 1] = roc_auc_score(y_train, clf.predict_proba(x_train)[:,1])

            scores[depth - 1, id_split, 1, 1] = roc_auc_score(y_val, clf.predict_proba(x_val)[:,1])

            

    return scores



def report_cv(scores):

    sns.lineplot(data = pd.DataFrame(scores.mean(1)[:,:, 1], index = list(range(1, 101)), columns = ['train', 'test']))

    plt.show()

    

    val_scores = scores.mean(1)[:, 1]

    

    print('Best model')

    print('***********')

    print('Mean validation accuracy: ', scores.mean(1)[:, 1, 0].max())

    print('Mean validation AUC: ', scores.mean(1)[:, 1, 1].max())

    print('Depth of the best model: ', scores.mean(1)[:, 1, 1].argmax() + 1)
scores = decision_tree_cv(x_train, y_train)
report_cv(scores)
scores_pca = decision_tree_cv(x_train_scaled, y_train)
report_cv(scores_pca)
depth_best_model = scores.mean(1)[:, 1, 1].argmax() + 1

clf = DecisionTreeClassifier(max_depth=depth_best_model, random_state=42).fit(x_train, y_train)
importances = pd.Series(clf.feature_importances_, index=shots.columns[:-1]).sort_values(ascending=False)



sns.barplot(x = importances.values, y = importances.index, orient='h', palette='Reds_r')

plt.title('Gini importance of features', fontsize=16)

plt.show()
perm_importances = permutation_importance(

    clf,

    x_train,

    y_train,

    scoring = 'roc_auc',

    n_repeats = 10,

    n_jobs = -1

)

df_perm_importances = pd.DataFrame(perm_importances.importances, index=shots.columns[:-1]).T

df_perm_importances = df_perm_importances.melt(

    value_vars = df_perm_importances.columns,

    var_name = 'feature',

    value_name = 'importance'

)



plt.figure(figsize=(10,5))

sns.boxplot(data = df_perm_importances, x = 'importance', y = 'feature')

plt.title('Permutation importance using AUC (train set)', fontsize=16)

plt.show()
plt.figure(figsize=(14,10))

plot_tree(

    clf,

    feature_names = shots.columns[:-1], 

    class_names = ['Miss', 'Hit'],

    filled = True,

    impurity = False,

    max_depth = 2

)

plt.show()
plot_tree(

    clf,

    feature_names = shots.columns[:-1], 

    class_names = ['Miss', 'Hit'],

    filled = True,

    impurity = False

)

plt.show()