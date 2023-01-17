%matplotlib inline 

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GMM
def encode_categorical(df, col):
    dummies = pd.get_dummies(df[col], prefix=col)
    df.drop(col, axis=1, inplace=True)
    df = df.join(dummies)
    return df

def binning(df, col, bins):
    df[col] = pd.cut(df[col], bins)
    return df

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=2, 
                        train_sizes=np.linspace(.1, 1.0, 5), scoring='log_loss'):
    plt.figure()
    plt.title(title)
    
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel('Training size')
    plt.ylabel('Score')
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, 
                                                            train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
        
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    
    print("Last cross-val score: {}".format(test_scores.flatten()[-1]))
    
    plt.grid()
    plt.show()
    
def init_learning_data(df):
    unknown_mask = df['shot_made_flag'].isnull()

    features = df[~unknown_mask]
    labels = df.loc[~unknown_mask, 'shot_made_flag']

    X = features.drop('shot_made_flag', axis=1).copy()
    Y = labels
    
    return X,Y
raw_df = pd.read_csv('../input/data.csv')
X, Y = init_learning_data(raw_df)

seed = 7
num_folds=5
scoring='log_loss'
num_instances=len(X)

kfold = KFold(n=num_instances, n_folds=num_folds, random_state=seed)

rf = RandomForestClassifier(n_estimators=50)
gbc = GradientBoostingClassifier()
lr = LogisticRegression()
# Calculate a vector of previous shots (probably could be done more efficient ...)
def get_prev_made_shot(row, df):
    game_id = row['game_id']
    game_event_id = row['game_event_id']
    
    game_df = df.loc[df['game_id'] == game_id].sort_values('game_event_id')
    prev_events_df = game_df.loc[game_df['game_event_id'] < game_event_id]
    
    # Was previous shot in game made?
    last = prev_events_df.tail(1)
    last_shot = last['shot_made_flag']
    
    # if game starts, or previous shot was unknown
    if (last_shot.empty | pd.isnull(last_shot).any()):
        return 0.5
    
    return int(last_shot.values[0])

prev_shot_made_s = X.apply(lambda x: get_prev_made_shot(x, X.join(Y)), axis=1)
prev_shot_made_s.name = "prev_shot_made"
# Calculate a vector of shot location cluster based on Gaussian Mixture Model
def get_shot_location_clusters(df, num_clusters):
    gmm = GMM(
        n_components=num_clusters,
        covariance_type='full',
        params='wmc',
        init_params='wmc',
        random_state=1,
        n_init=3,
        verbose=0)

    gmm.fit(df.loc[:, ['loc_x', 'loc_y']])
    
    return gmm.predict(df.loc[:, ['loc_x', 'loc_y']]) 

location_clusters_s = get_shot_location_clusters(X, 13)
X, Y = init_learning_data(raw_df)
features = ['loc_y', 'loc_x']

plot_learning_curve(gbc, 'GBC', X.loc[:,features], Y, cv=kfold)
X, Y = init_learning_data(raw_df)
X = (X
     .assign(seconds_from_period_end = lambda x: 60 * x['minutes_remaining'] + x['seconds_remaining'])
     .assign(last_5_sec_in_period = lambda x: x['seconds_from_period_end'] < 5))

features = ['loc_x', 'loc_y', 'last_5_sec_in_period']
    
plot_learning_curve(gbc, 'GBC', X.loc[:,features], Y, cv=kfold)
X, Y = init_learning_data(raw_df)

features = ['loc_x', 'loc_y', 'shot_distance']

plot_learning_curve(gbc, 'GBC', X.loc[:,features], Y, cv=kfold)
X, Y = init_learning_data(raw_df)

cst_dummies = pd.get_dummies(X['combined_shot_type'], prefix='combined_shot_type', prefix_sep=":")
X = X.join(cst_dummies)

features = np.concatenate([['loc_x', 'loc_y'], cst_dummies.columns])

plot_learning_curve(gbc, 'GBC', X.loc[:,features], Y, cv=kfold)
X, Y = init_learning_data(raw_df)

X['shot_distance']= np.sqrt((X['loc_x']/10)**2 + (X['loc_y']/10)**2)

at_dummies = pd.get_dummies(X['action_type'], prefix='action_type', prefix_sep=":")
X = X.join(at_dummies)

features = np.concatenate([['shot_distance'] ,at_dummies.columns])
plot_learning_curve(gbc, 'GBC', X.loc[:,features], Y, cv=kfold)
X, Y = init_learning_data(raw_df)

X = X.join(prev_shot_made_s)

features = np.concatenate([['loc_x', 'loc_y', 'prev_shot_made']])
plot_learning_curve(gbc, 'GBC', X.loc[:,features], Y, cv=kfold)
X, Y = init_learning_data(raw_df)

X = X.join(prev_shot_made_s)
X = X.join(at_dummies)

features = np.concatenate([['prev_shot_made', 'shot_distance'], at_dummies.columns])
plot_learning_curve(gbc, 'GBC', X.loc[:,features], Y, cv=kfold)
X, Y = init_learning_data(raw_df)

lc_dummies = pd.get_dummies(location_clusters_s, prefix='location_cluster', prefix_sep=":")
lc_dummies = lc_dummies.set_index(X.index)

X = X.join(lc_dummies)
X = X.join(at_dummies)
X = X.join(cst_dummies)

features = np.concatenate([lc_dummies.columns, at_dummies.columns])
plot_learning_curve(gbc, 'GBC', X.loc[:,features], Y, cv=kfold)
X, Y = init_learning_data(raw_df)

loc_x_20_bins = binning(X, 'loc_x', 20)
loc_y_20_bins = binning(X, 'loc_y', 20)

loc_x_20_dummies = pd.get_dummies(X['loc_x'], prefix='loc_x')
loc_y_20_dummies = pd.get_dummies(X['loc_y'], prefix='loc_y')

X = X.join(loc_x_20_dummies)
X = X.join(loc_y_20_dummies)

features = np.concatenate([loc_x_20_dummies.columns, loc_y_20_dummies.columns])
plot_learning_curve(gbc, 'GBC', X.loc[:,features], Y, cv=kfold)