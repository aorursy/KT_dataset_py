%matplotlib inline 



import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.decomposition import PCA, KernelPCA

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.metrics import make_scorer

from sklearn.grid_search import GridSearchCV

from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, chi2

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, AdaBoostClassifier



sns.set_style('whitegrid')

pd.set_option('display.max_columns', None) # display all columns
data = pd.read_csv('../input/data.csv')



data.set_index('shot_id', inplace=True)

data["action_type"] = data["action_type"].astype('object')

data["combined_shot_type"] = data["combined_shot_type"].astype('category')

data["game_event_id"] = data["game_event_id"].astype('category')

data["game_id"] = data["game_id"].astype('category')

data["period"] = data["period"].astype('object')

data["playoffs"] = data["playoffs"].astype('category')

data["season"] = data["season"].astype('category')

data["shot_made_flag"] = data["shot_made_flag"].astype('category')

data["shot_type"] = data["shot_type"].astype('category')

data["team_id"] = data["team_id"].astype('category')
data.head(2)
data.dtypes
data.shape
data.describe(include=['number'])
data.describe(include=['object', 'category'])
ax = plt.axes()

sns.countplot(x='shot_made_flag', data=data, ax=ax);

ax.set_title('Target class distribution')

plt.show()
f, axarr = plt.subplots(4, 2, figsize=(15, 15))



sns.boxplot(x='lat', y='shot_made_flag', data=data, showmeans=True, ax=axarr[0,0])

sns.boxplot(x='lon', y='shot_made_flag', data=data, showmeans=True, ax=axarr[0, 1])

sns.boxplot(x='loc_y', y='shot_made_flag', data=data, showmeans=True, ax=axarr[1, 0])

sns.boxplot(x='loc_x', y='shot_made_flag', data=data, showmeans=True, ax=axarr[1, 1])

sns.boxplot(x='minutes_remaining', y='shot_made_flag', showmeans=True, data=data, ax=axarr[2, 0])

sns.boxplot(x='seconds_remaining', y='shot_made_flag', showmeans=True, data=data, ax=axarr[2, 1])

sns.boxplot(x='shot_distance', y='shot_made_flag', data=data, showmeans=True, ax=axarr[3, 0])



axarr[0, 0].set_title('Latitude')

axarr[0, 1].set_title('Longitude')

axarr[1, 0].set_title('Loc y')

axarr[1, 1].set_title('Loc x')

axarr[2, 0].set_title('Minutes remaining')

axarr[2, 1].set_title('Seconds remaining')

axarr[3, 0].set_title('Shot distance')



plt.tight_layout()

plt.show()
sns.pairplot(data, vars=['loc_x', 'loc_y', 'lat', 'lon', 'shot_distance'], hue='shot_made_flag', size=3)

plt.show()
f, axarr = plt.subplots(8, figsize=(15, 25))



sns.countplot(x="combined_shot_type", hue="shot_made_flag", data=data, ax=axarr[0])

sns.countplot(x="season", hue="shot_made_flag", data=data, ax=axarr[1])

sns.countplot(x="period", hue="shot_made_flag", data=data, ax=axarr[2])

sns.countplot(x="playoffs", hue="shot_made_flag", data=data, ax=axarr[3])

sns.countplot(x="shot_type", hue="shot_made_flag", data=data, ax=axarr[4])

sns.countplot(x="shot_zone_area", hue="shot_made_flag", data=data, ax=axarr[5])

sns.countplot(x="shot_zone_basic", hue="shot_made_flag", data=data, ax=axarr[6])

sns.countplot(x="shot_zone_range", hue="shot_made_flag", data=data, ax=axarr[7])



axarr[0].set_title('Combined shot type')

axarr[1].set_title('Season')

axarr[2].set_title('Period')

axarr[3].set_title('Playoffs')

axarr[4].set_title('Shot Type')

axarr[5].set_title('Shot Zone Area')

axarr[6].set_title('Shot Zone Basic')

axarr[7].set_title('Shot Zone Range')



plt.tight_layout()

plt.show()
unknown_mask = data['shot_made_flag'].isnull()
data_cl = data.copy() # create a copy of data frame

target = data_cl['shot_made_flag'].copy()



# Remove some columns

data_cl.drop('team_id', axis=1, inplace=True) # Always one number

data_cl.drop('lat', axis=1, inplace=True) # Correlated with loc_x

data_cl.drop('lon', axis=1, inplace=True) # Correlated with loc_y

data_cl.drop('game_id', axis=1, inplace=True) # Independent

data_cl.drop('game_event_id', axis=1, inplace=True) # Independent

data_cl.drop('team_name', axis=1, inplace=True) # Always LA Lakers

data_cl.drop('shot_made_flag', axis=1, inplace=True)
def detect_outliers(series, whis=1.5):

    q75, q25 = np.percentile(series, [75 ,25])

    iqr = q75 - q25

    return ~((series - series.median()).abs() <= (whis * iqr))



## For now - do not remove anything
# Remaining time

data_cl['seconds_from_period_end'] = 60 * data_cl['minutes_remaining'] + data_cl['seconds_remaining']

data_cl['last_5_sec_in_period'] = data_cl['seconds_from_period_end'] < 5



data_cl.drop('minutes_remaining', axis=1, inplace=True)

data_cl.drop('seconds_remaining', axis=1, inplace=True)

data_cl.drop('seconds_from_period_end', axis=1, inplace=True)



## Matchup - (away/home)

data_cl['home_play'] = data_cl['matchup'].str.contains('vs').astype('int')

data_cl.drop('matchup', axis=1, inplace=True)



# Game date

data_cl['game_date'] = pd.to_datetime(data_cl['game_date'])

data_cl['game_year'] = data_cl['game_date'].dt.year

data_cl['game_month'] = data_cl['game_date'].dt.month

data_cl.drop('game_date', axis=1, inplace=True)



# Loc_x, and loc_y binning

data_cl['loc_x'] = pd.cut(data_cl['loc_x'], 25)

data_cl['loc_y'] = pd.cut(data_cl['loc_y'], 25)



# Replace 20 least common action types with value 'Other'

rare_action_types = data_cl['action_type'].value_counts().sort_values().index.values[:20]

data_cl.loc[data_cl['action_type'].isin(rare_action_types), 'action_type'] = 'Other'
categorial_cols = [

    'action_type', 'combined_shot_type', 'period', 'season', 'shot_type',

    'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'game_year',

    'game_month', 'opponent', 'loc_x', 'loc_y']



for cc in categorial_cols:

    dummies = pd.get_dummies(data_cl[cc])

    dummies = dummies.add_prefix("{}#".format(cc))

    data_cl.drop(cc, axis=1, inplace=True)

    data_cl = data_cl.join(dummies)
# TODO
# Separate dataset for validation

data_submit = data_cl[unknown_mask]



# Separate dataset for training

X = data_cl[~unknown_mask]

Y = target[~unknown_mask]
threshold = 0.90

vt = VarianceThreshold().fit(X)



# Find feature names

feat_var_threshold = data_cl.columns[vt.variances_ > threshold * (1-threshold)]

feat_var_threshold
model = RandomForestClassifier()

model.fit(X, Y)



feature_imp = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["importance"])

feat_imp_20 = feature_imp.sort_values("importance", ascending=False).head(20).index

feat_imp_20
X_minmax = MinMaxScaler(feature_range=(0,1)).fit_transform(X)

X_scored = SelectKBest(score_func=chi2, k='all').fit(X_minmax, Y)

feature_scoring = pd.DataFrame({

        'feature': X.columns,

        'score': X_scored.scores_

    })



feat_scored_20 = feature_scoring.sort_values('score', ascending=False).head(20)['feature'].values

feat_scored_20
rfe = RFE(LogisticRegression(), 20)

rfe.fit(X, Y)



feature_rfe_scoring = pd.DataFrame({

        'feature': X.columns,

        'score': rfe.ranking_

    })



feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values

feat_rfe_20
features = np.hstack([

        feat_var_threshold, 

        feat_imp_20,

        feat_scored_20,

        feat_rfe_20

    ])



features = np.unique(features)

print('Final features set:\n')

for f in features:

    print("\t-{}".format(f))
data_cl = data_cl.ix[:, features]

data_submit = data_submit.ix[:, features]

X = X.ix[:, features]



print('Clean dataset shape: {}'.format(data_cl.shape))

print('Subbmitable dataset shape: {}'.format(data_submit.shape))

print('Train features shape: {}'.format(X.shape))

print('Target label shape: {}'. format(Y.shape))
components = 8

pca = PCA(n_components=components).fit(X)
pca_variance_explained_df = pd.DataFrame({

    "component": np.arange(1, components+1),

    "variance_explained": pca.explained_variance_ratio_            

    })



ax = sns.barplot(x='component', y='variance_explained', data=pca_variance_explained_df)

ax.set_title("PCA - Variance explained")

plt.show()
X_pca = pd.DataFrame(pca.transform(X)[:,:2])

X_pca['target'] = Y.values

X_pca.columns = ["x", "y", "target"]



sns.lmplot('x','y', 

           data=X_pca, 

           hue="target", 

           fit_reg=False, 

           markers=["o", "x"], 

           palette="Set1", 

           size=7,

           scatter_kws={"alpha": .2}

          )

plt.show()
seed = 7

processors=1

num_folds=3

num_instances=len(X)

scoring='log_loss'



kfold = KFold(n=num_instances, n_folds=num_folds, random_state=seed)
# Prepare some basic models

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('K-NN', KNeighborsClassifier(n_neighbors=5)))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

#models.append(('SVC', SVC(probability=True)))



# Evaluate each model in turn

results = []

names = []



for name, model in models:

    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)

    results.append(cv_results)

    names.append(name)

    print("{0}: ({1:.3f}) +/- ({2:.3f})".format(name, cv_results.mean(), cv_results.std()))
cart = DecisionTreeClassifier()

num_trees = 100



model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)



results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)

print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))
num_trees = 100

num_features = 10



model = RandomForestClassifier(n_estimators=num_trees, max_features=num_features)



results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)

print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))
num_trees = 100

num_features = 10



model = ExtraTreesClassifier(n_estimators=num_trees, max_features=num_features)



results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)

print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))
model = AdaBoostClassifier(n_estimators=100, random_state=seed)



results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)

print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))
model = GradientBoostingClassifier(n_estimators=100, random_state=seed)



results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)

print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))
lr_grid = GridSearchCV(

    estimator = LogisticRegression(random_state=seed),

    param_grid = {

        'penalty': ['l1', 'l2'],

        'C': [0.001, 0.01, 1, 10, 100, 1000]

    }, 

    cv = kfold, 

    scoring = scoring, 

    n_jobs = processors)



lr_grid.fit(X, Y)



print(lr_grid.best_score_)

print(lr_grid.best_params_)
lda_grid = GridSearchCV(

    estimator = LinearDiscriminantAnalysis(),

    param_grid = {

        'solver': ['lsqr'],

        'shrinkage': [0, 0.25, 0.5, 0.75, 1],

        'n_components': [None, 2, 5, 10]

    }, 

    cv = kfold, 

    scoring = scoring, 

    n_jobs = processors)



lda_grid.fit(X, Y)



print(lda_grid.best_score_)

print(lda_grid.best_params_)
knn_grid = GridSearchCV(

    estimator = Pipeline([

        ('min_max_scaler', MinMaxScaler()),

        ('knn', KNeighborsClassifier())

    ]),

    param_grid = {

        'knn__n_neighbors': [25],

        'knn__algorithm': ['ball_tree'],

        'knn__leaf_size': [2, 3, 4],

        'knn__p': [1]

    }, 

    cv = kfold, 

    scoring = scoring, 

    n_jobs = processors)



knn_grid.fit(X, Y)



print(knn_grid.best_score_)

print(knn_grid.best_params_)
rf_grid = GridSearchCV(

    estimator = RandomForestClassifier(warm_start=True, random_state=seed),

    param_grid = {

        'n_estimators': [100, 200],

        'criterion': ['gini', 'entropy'],

        'max_features': [18, 20],

        'max_depth': [8, 10],

        'bootstrap': [True]

    }, 

    cv = kfold, 

    scoring = scoring, 

    n_jobs = processors)



rf_grid.fit(X, Y)



print(rf_grid.best_score_)

print(rf_grid.best_params_)
ada_grid = GridSearchCV(

    estimator = AdaBoostClassifier(random_state=seed),

    param_grid = {

        'algorithm': ['SAMME', 'SAMME.R'],

        'n_estimators': [10, 25, 50],

        'learning_rate': [1e-3, 1e-2, 1e-1]

    }, 

    cv = kfold, 

    scoring = scoring, 

    n_jobs = processors)



ada_grid.fit(X, Y)



print(ada_grid.best_score_)

print(ada_grid.best_params_)
gbm_grid = GridSearchCV(

    estimator = GradientBoostingClassifier(warm_start=True, random_state=seed),

    param_grid = {

        'n_estimators': [100, 200],

        'max_depth': [2, 3, 4],

        'max_features': [10, 15, 20],

        'learning_rate': [1e-1, 1]

    }, 

    cv = kfold, 

    scoring = scoring, 

    n_jobs = processors)



gbm_grid.fit(X, Y)



print(gbm_grid.best_score_)

print(gbm_grid.best_params_)
# Create sub models

estimators = []



estimators.append(('lr', LogisticRegression(penalty='l2', C=1)))

estimators.append(('gbm', GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, max_features=15, warm_start=True, random_state=seed)))

estimators.append(('rf', RandomForestClassifier(bootstrap=True, max_depth=8, n_estimators=200, max_features=20, criterion='entropy', random_state=seed)))

estimators.append(('ada', AdaBoostClassifier(algorithm='SAMME.R', learning_rate=1e-2, n_estimators=10, random_state=seed)))



# create the ensemble model

ensemble = VotingClassifier(estimators, voting='soft', weights=[2,3,3,1])



results = cross_val_score(ensemble, X, Y, cv=kfold, scoring=scoring,n_jobs=processors)

print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))
model = ensemble



model.fit(X, Y)

preds = model.predict_proba(data_submit)



submission = pd.DataFrame()

submission["shot_id"] = data_submit.index

submission["shot_made_flag"]= preds[:,0]



submission.to_csv("sub.csv",index=False)