# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('..//input/itea-goal-prediction//goal_train.csv', encoding='utf-8')

test = pd.read_csv('..//input/itea-goal-prediction//goal_test.csv', encoding='utf-8')
train.info()
test.info()
len(train.columns)
train.head()
train.sample(10)
y = train['is_goal']

y
train = train.drop(columns='is_goal')

train.head()
train['part'] = 1

test['part'] = 2

df = train.append(test, ignore_index=True)

df.head()
df.sample(10)
df.info()
df.describe()
df.isna().any()[lambda x: x]
df['foot'] = df['foot'].fillna(df['foot'].mode()[0])
df['weight'] = df['weight'].fillna(df['weight'].mean())
df['passportArea'] = df['passportArea'].fillna(df['passportArea'].mode()[0])
df['firstName'] = df['firstName'].fillna(df['firstName'].mode()[0])
df['middleName'].isna().value_counts()
df = df.drop(columns='middleName')
df['lastName'] = df['lastName'].fillna(df['lastName'].mode()[0])
df['currentTeamId'] = df['currentTeamId'].fillna(df['currentTeamId'].mode()[0])
df['currentNationalTeamId'] = df['currentNationalTeamId'].fillna(df['currentNationalTeamId'].mode()[0])
df['birthDate'] = df['birthDate'].fillna(df['birthDate'].mode()[0])
df['shortName'] = df['shortName'].fillna(df['shortName'].mode()[0])
df['height'] = df['height'].fillna(df['height'].mean())
df['role'] = df['role'].fillna(df['role'].mode()[0])
df['birthArea'] = df['birthArea'].fillna(df['birthArea'].mode()[0])
df['eventSec'] = np.round(df['eventSec'], decimals=2)
df[['area', 'birthArea', 'league']]
df['playingInMotherland'] = df['area'] == df['birthArea']

df['playingInMotherland']
df['flang'] = np.where (((df['y_1'] > 50) & (df['foot'] == 'left')), 1, 

                           np.where((df['y_1'] <= 50) & (df['foot'] == 'right'), 1, 0))
df['legioner'] = np.where((df['league'] == "IT") & (df['passportArea'] == "Italy"), 0,

                            np.where((df['league'] == "SP") & (df['passportArea'] == "Spain"), 0,

                                    np.where((df['league'] == "GE") & (df['passportArea'] == "Germany"), 0,

                                            np.where((df['league'] == "FR") & (df['passportArea'] == "France"), 0,

                                                    np.where((df['league'] == "EN") & (df['passportArea'] == "England"), 0, 1)))))
from datetime import datetime
df['age'] = df['birthDate'].map( lambda date: np.round((datetime.strptime('2020-09-20', '%Y-%m-%d') - datetime.strptime(date, '%Y-%m-%d')).days / 365.25, decimals=2))
df.plot.scatter(x='x_1', y='y_1', grid=True, legend=True)
from sklearn.neighbors import DistanceMetric

dist = DistanceMetric.get_metric('euclidean')
df['distanceToGates'] = df.apply(lambda row: np.round(dist.pairwise([[row['x_1'], row['y_1']], [50, 50]])[0][1], decimals=2), axis=1)
import numpy as np



def angle_between(p1, p2):

    ang1 = np.arctan2(*p1[::-1])

    ang2 = np.arctan2(*p2[::-1])

    return np.rad2deg((ang2 - ang1) % (2 * np.pi))
import category_encoders as ce
df['angleToGates'] = df.apply(lambda row: np.round(angle_between((row['x_1'], row['y_1']), (50, 50)), decimals=2), axis=1)
df._get_numeric_data()
df.columns[df.isna().any()].tolist()
import matplotlib.pyplot as plt
def histograms_plot(dataframe, features, rows, cols, figsize=(20, 20)):

    fig = plt.figure(figsize=figsize)

    for i, feature in enumerate(features):

        ax = fig.add_subplot(rows,cols, i+1)

        dataframe[feature].hist(bins=20, ax=ax,facecolor='#56cfe1')

        ax.set_title(feature, color='#e76f51')



    fig.tight_layout()  

    plt.show()

num_columns = ['eventSec', 'y_1', 'x_1', 'weight', 'height', 'age', 'distanceToGates', 'angleToGates']

    

histograms_plot(df, num_columns, 4, 2)
for col_name in num_columns:

    print('Skew of :', col_name, ':', df[col_name].skew())

    print('Kurtosis of :', col_name, ':', df[col_name].kurtosis())

    print('-------------------')
fixed_x_1 = np.log(np.abs(df['x_1']))

print('Skew of  x_1 : ', fixed_x_1.skew())

print('Kurtosis of x_1 : ', fixed_x_1.kurtosis())
fixed_x_1 = np.square(df['x_1'])

print('Skew of  x_1 : ', fixed_x_1.skew())

print('Kurtosis of x_1 : ', fixed_x_1.kurtosis())

df['x_1'] = fixed_x_1
def sigmoid(x):

    return 1/(1 + np.exp(-x)) 
fixed_eventSec = np.sqrt(np.abs(df['eventSec']))

print('Skew of  eventSec : ', fixed_eventSec.skew())

print('Kurtosis of eventSec : ', fixed_eventSec.kurtosis())

df['eventSec_1'] = fixed_eventSec

histograms_plot(df, ['eventSec', 'eventSec_1'], 1, 2, figsize=(10, 5))

df['eventSec'] = df['eventSec_1']

df = df.drop(columns='eventSec_1')
fixed_weight= np.square(np.abs(df['weight']))

print('Skew of  weight : ', fixed_weight.skew())

print('Kurtosis of weight : ', fixed_weight.kurtosis())

df['weight_1'] = fixed_weight

histograms_plot(df, ['weight', 'weight_1'], 1, 2, figsize=(10, 5))

df['weight'] = df['weight_1']

df = df.drop(columns='weight_1')
fixed_height= np.square(np.abs(df['height']))

print('Skew of  height : ', fixed_height.skew())

print('Kurtosis of height : ', fixed_height.kurtosis())

df['height_1'] = fixed_height

histograms_plot(df, ['height', 'height_1'], 1, 2, figsize=(10, 5))

df['height'] = df['height_1']

df = df.drop(columns='height_1')
# fixed_age= np.log(np.abs(df['age']))

# print('Skew of  age : ', fixed_age.skew())

# print('Kurtosis of age : ', fixed_age.kurtosis())

# df['age_1'] = fixed_age

# histograms_plot(df, ['age', 'age_1'], 1, 2, figsize=(10, 5))

# df['age'] = df['age_1']

# df = df.drop(columns='age_1')

# Result

# Skew of  age :  -0.03771203652443697

# Kurtosis of age :  -0.47623913319654587

# Was

# Skew of : age : 0.2632797465060076

# Kurtosis of : age : -0.2891482093613438
# fixed_distanceToGates = np.square(np.abs(df['distanceToGates']))

# print('Skew of  distanceToGates : ', fixed_distanceToGates.skew())

# print('Kurtosis of distanceToGates : ', fixed_distanceToGates.kurtosis())

# df['distanceToGates_1'] = fixed_distanceToGates

# histograms_plot(df, ['distanceToGates', 'distanceToGates_1'], 1, 2, figsize=(10, 5))

# df['distanceToGates'] = df['distanceToGates_1']

# df = df.drop(columns='distanceToGates_1')

# Result

# Skew of  distanceToGates :  -0.2726414219391746

# Kurtosis of distanceToGates :  -0.56980014668585

# Was

# Skew of : distanceToGates : -0.7252659051840858

# Kurtosis of : distanceToGates : 0.11196854368607045
fixed_angleToGates= np.sqrt(np.sqrt(np.abs(df['angleToGates'])))

print('Skew of  angleToGates : ', fixed_angleToGates.skew())

print('Kurtosis of angleToGates : ', fixed_angleToGates.kurtosis())

df['angleToGates_1'] = fixed_angleToGates

histograms_plot(df, ['angleToGates', 'angleToGates_1'], 1, 2, figsize=(10, 5))

df['angleToGates'] = df['angleToGates_1']

df = df.drop(columns='angleToGates_1')
from scipy.stats import shapiro

from scipy.stats import normaltest

from scipy.stats import anderson
def normal_test(test, dataframe, features, p_threshold = 0.05):

    test_results = []

    for feature in features:

        test_res = test(dataframe[feature])

        print(feature, 'is distributted', 'normally' if test_res[1] > p_threshold else 'not normally')

        test_results.append(test_res)

    return test_results
normal_test(shapiro, df, num_columns)
normal_test(normaltest, df, num_columns)
def anderson_test(dataframe, features):

    for feature in features:

        test_res = anderson(dataframe[feature])

        stat = test_res.statistic

        critical_values = test_res.critical_values

        significance_level = test_res.significance_level

        for i in range(len(critical_values)):

            sl, cv = significance_level[i], critical_values[i]

            print(feature, 'is distributted', 'normally' if stat < cv else 'not normally', 'at the', sl, '% level')
anderson_test(df, num_columns)
histograms_plot(df, num_columns, 4, 2, figsize=(10, 10))
correlations = df.corr()

correlations
import seaborn as sns;
sns.heatmap(correlations)
correlations.unstack().sort_values(ascending=False).drop_duplicates()
df = df.drop(columns=['shot_id', 'matchId', 'teamId', 'playerId', 'birthDate'])

ce_bin = ce.BinaryEncoder(cols=['passportArea', 'currentTeamId', 'currentNationalTeamId', 'birthArea', 'city', 'officialName', 'shortName', 'name'])

df = ce_bin.fit_transform(df)

df = pd.get_dummies(df, columns=['matchPeriod', 'is_CA', 'body_part', 'foot', 'league', 'role', 'area', 'playingInMotherland', 'flang', 'legioner', 'firstName', 'lastName'])
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score
df_train = df[df['part'] == 1]

df_train = df_train.drop(columns='part')
df_test = df[df['part'] == 2]

df_test = df_test.drop(columns='part')
from sklearn.model_selection import KFold
def perform_cross_validation(model, X, y, threshold=0.5, n_splits=5):

    kf = KFold(shuffle=True, n_splits=n_splits, random_state=42)

    scores = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X.loc[train_index], X.loc[test_index]

        y_train, y_test = y.loc[train_index], y.loc[test_index]

        model.fit(X_train, y_train)

        scores.append(f1_score(y_test, (model.predict_proba(X_test)[:,1] > threshold).astype('int')))

    return scores
def find_best_f1_threshold(model, train_X, train_y, thresholds=[0.5]):

    overall_scores = []

    for threshold in thresholds:

        scores = perform_cross_validation(model, train_X, train_y, threshold)

        overall_scores.append(np.mean(scores))



    best_threshold_index = np.argmax(overall_scores)

    return thresholds[best_threshold_index]
find_threshold_model = LogisticRegression(random_state=42, max_iter=10000).fit(df_train, y)
best_threshold = find_best_f1_threshold(find_threshold_model, df_train, y, thresholds=[0.15, 0.32, 0.18, 0.23, 0.005, 0.42])
best_threshold
# best_threshold = 0.18
classifier = LogisticRegression(random_state=42, max_iter=10000).fit(df_train, y)
scores = perform_cross_validation(classifier, df_train, y, best_threshold)

np.mean(scores)
# 0.3596028179371231
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
standart_classifier = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=10000)).fit(df_train, y)
scores = perform_cross_validation(standart_classifier, df_train, y, best_threshold)

np.mean(scores)
# 0.3513648067597932
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest



# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1300, num = 100)]

# # Number of features to consider at every split

# max_features = ['auto', 'sqrt']

# # Maximum number of levels in tree

# max_depth = [1, 3, 5, 7, 10, 13, 15, 20]

# # Minimum number of samples required to split a node

# min_samples_split = [1, 2, 5, 10]

# # Minimum number of samples required at each leaf node

# min_samples_leaf = [1, 2, 4]

# # Method of selecting samples for training each tree

# bootstrap = [True, False]

# random_grid = {'n_estimators': n_estimators,

#                'max_features': max_features,

#                'max_depth': max_depth,

#                'min_samples_split': min_samples_split,

#                'min_samples_leaf': min_samples_leaf,

#                'bootstrap': bootstrap}



# random_forest_clf = RandomForestClassifier();



# rf_random = RandomizedSearchCV(estimator = random_forest_clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# rf_random.fit(df_train, y)
# rf_random.best_params_
random_forest_clf = RandomForestClassifier(random_state=42, n_estimators=1255, max_depth=15, min_samples_leaf=1, min_samples_split=2, bootstrap=True, max_features='sqrt').fit(df_train, y)
scores_random_forest = perform_cross_validation(random_forest_clf, df_train, y, best_threshold)

np.mean(scores_random_forest)
# 0.3063723046112142
pd.DataFrame({

    'variable': df_train.columns,

    'importance': random_forest_clf.feature_importances_

}).sort_values('importance', ascending=False)
from sklearn.ensemble import BaggingClassifier
bagging_logistic_classifier = BaggingClassifier(classifier, max_samples=0.85, max_features=0.85, random_state=42).fit(df_train, y)
scores = perform_cross_validation(bagging_logistic_classifier, df_train, y, best_threshold)

np.mean(scores)
# 0.3518289228036299
from sklearn.ensemble import VotingClassifier
voting_classifier = VotingClassifier(

    weights=[1.1, 0.1],

    estimators=[('lr', classifier), ('rf', random_forest_clf)],

    voting='soft'

).fit(df_train, y)
scores = perform_cross_validation(voting_classifier, df_train, y, best_threshold)

np.mean(scores)
# 0.36217970352147233
import xgboost as xgb
!nvidia-smi
# xg_class_grid = XGBClassifier(objective='binary:logistic', missing = None)

# param_grid = {

#         'learning_rate': [0, 0.01, 0.03, 0.1],

#         'min_split_loss': [0, 0.01, 0.2, 0.5, 1],

#         'max_depth': [3, 4, 14, 17],

#         'min_child_weight': [1, 3, 7, 8],

#         'subsample': [0.5, 0.7, 0.9, 1],

#         'n_estimators': [300, 500, 600, 700],

#         'seed' : [42],

#         'tree_method': ['gpu_hist'],

#         'early_stopping_rounds': [15, 22, 30, 70, 100]

#         }



# xg_grid = RandomizedSearchCV(estimator=xg_class_grid, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# xg_grid.fit(df_train, y)
# xg_grid.best_params_
xg_classifier = xgb.XGBClassifier(

    colsample_bytree=0.7,

    gamma=3,

    learning_rate=0.02,

    max_delta_step=0,

    max_depth=2,

    min_child_weight=8,

    n_estimators=600,

    reg_alpha=0.04,

    reg_lambda=0.04,

    scale_pos_weight=1,

    seed=0,

    subsample=0.8,

    missing=None, 

    nthread=4,

    objective='binary:logistic', 

    silent=False

).fit(df_train, y)
scores = perform_cross_validation(xg_classifier, df_train, y, best_threshold)

np.mean(scores)
# 0.4059622546375321
voting_classifier_with_xgb = VotingClassifier(

    weights=[1.3, 0.9, 0.05],

    estimators=[('xgboost', xg_classifier), ('lr', classifier), ('rf', random_forest_clf)],

    voting='soft'

).fit(df_train, y)
scores = perform_cross_validation(voting_classifier_with_xgb, df_train, y, n_splits=2, threshold=best_threshold)

np.mean(scores)
# defining various steps required for the genetic algorithm

# import random

# from sklearn.metrics import accuracy_score

# def initilization_of_population(size,n_feat):

#     population = []

#     for i in range(size):

#         chromosome = np.ones(n_feat,dtype=np.bool)

#         chromosome[:int(0.3*n_feat)]=False

#         np.random.shuffle(chromosome)

#         population.append(chromosome)

#     return population



# def fitness_score(population, model, X_train, X_test, y_train, y_test):

#     scores = []

#     for chromosome in population:

#         model.fit(X_train.iloc[:,chromosome],y_train)

#         predictions = model.predict(X_test.iloc[:,chromosome])

#         scores.append(accuracy_score(y_test,predictions))

#     scores, population = np.array(scores), np.array(population) 

#     inds = np.argsort(scores)

#     return list(scores[inds][::-1]), list(population[inds,:][::-1])



# def selection(pop_after_fit,n_parents):

#     population_nextgen = []

#     for i in range(n_parents):

#         population_nextgen.append(pop_after_fit[i])

#     return population_nextgen



# def crossover(pop_after_sel):

#     population_nextgen=pop_after_sel

#     for i in range(len(pop_after_sel)):

#         child=pop_after_sel[i]

#         child[3:7]=pop_after_sel[(i+1)%len(pop_after_sel)][3:7]

#         population_nextgen.append(child)

#     return population_nextgen



# def mutation(pop_after_cross,mutation_rate):

#     population_nextgen = []

#     for i in range(0,len(pop_after_cross)):

#         chromosome = pop_after_cross[i]

#         for j in range(len(chromosome)):

#             if random.random() < mutation_rate:

#                 chromosome[j]= not chromosome[j]

#         population_nextgen.append(chromosome)

#     #print(population_nextgen)

#     return population_nextgen



# def generations(size, n_feat, n_parents, mutation_rate, n_gen, X_train, X_test, y_train, y_test, model):

#     best_chromo= []

#     best_score= []

#     population_nextgen=initilization_of_population(size,n_feat)

#     for i in range(n_gen):

#         scores, pop_after_fit = fitness_score(population_nextgen, model, X_train, X_test, y_train, y_test)

#         print(scores[:2])

#         pop_after_sel = selection(pop_after_fit,n_parents)

#         pop_after_cross = crossover(pop_after_sel)

#         population_nextgen = mutation(pop_after_cross,mutation_rate)

#         best_chromo.append(pop_after_fit[0])

#         best_score.append(scores[0])

#     return best_chromo,best_score



# predictions = logmodel.predict(X_test.iloc[:,chromo[-1]])
# len(df_train.columns)
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=0.20, random_state=40)
# chromo, score = generations(size=200, n_feat=len(df_train.columns), n_parents=100, mutation_rate=0.10, n_gen=10, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, model=classifier)

# classifier.fit(df_train.iloc[:,chromo[-1]], y)
predictions = voting_classifier_with_xgb.predict_proba(df_test)

predictions
predictions[:,1]
f1_predictions = (predictions[:,1] > best_threshold).astype('int')

f1_predictions
f1_predictions.sum()
submissions = pd.read_csv('..//input/itea-goal-prediction//goal_submission.csv')

submissions.head()
submissions['is_goal'] = f1_predictions

submissions.to_csv('submission21.csv', index = False)