# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

import warnings



warnings.filterwarnings('ignore')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def initial_df(file, columns, option):

    df = pd.read_csv(file, skiprows=[0])

    if option == 'drop':

        df = df.drop(columns=columns)

    elif option == 'extract':

        df = df[columns]

    

    return df
standard_df = initial_df("../input/soccer-reference/standard data/team data.csv", 

                         ['# Pl', 'MP', 'Starts', 'Min', 'Gls', 'Ast', 'PK', 'PKatt', 'CrdR', 'CrdY', 'xG', 'npxG', 'xA'],

                        'drop')

standard_df.head()
shooting_df = initial_df("../input/soccer-reference/shooting/team.csv", ['Squad', 'SoT%', 'SoT/90', 'SoT', 'G/SoT', 'npxG/Sh'], 'extract')

shooting_df.head()
possession_df = initial_df("../input/soccer-reference/possession/team.csv", ['Squad', 'Succ%', 'Rec%'], 'extract')

possession_df.head()
playingTime_df = initial_df("../input/soccer-reference/playing time/team.csv", ['Squad', 'PPM', '+/-90', 'xG+/-90'], 'extract')

playingTime_df.head()
passing_df = initial_df("../input/soccer-reference/passing/team.csv", 

                        ['Squad', 'Cmp%', 'Cmp%.1', 'Cmp%.2', 'Cmp%.3'],

                       'extract')

passing_df.head()
goalkeeping_df = initial_df("../input/soccer-reference/goalkeeping/team.csv", 

                            ['Squad', 'GA90', 'Save%'],

                           'extract')

goalkeeping_df.head()
goalShotCreation_df = initial_df("../input/soccer-reference/goal and shot creation/team.csv", ['Squad', 'SCA90', 'GCA90'], 'extract')

goalShotCreation_df.head()
defence_df = initial_df("../input/soccer-reference/defensive actions/team.csv", ['Squad', 'Tkl%', '%'], 'extract')

defence_df.head()
adv_goalkeeping_df = initial_df("../input/soccer-reference/advanced goalkeeping/team.csv", 

                                ['Squad', '/90', 'Cmp%', 'Launch%', 'AvgLen', 'Launch%.1', 'AvgLen.1', 'Stp%', '#OPA/90', 'AvgDist'],

                               'extract')

adv_goalkeeping_df.head()
miscellaneous_df = initial_df('../input/soccer-reference/Miscellaneous.csv', ['Squad', 'Won%'], 'extract')

miscellaneous_df.head()
ranking_df = pd.read_csv('../input/soccer-reference/soccer-spi/spi_global_rankings.csv')[['name', 'off', 'def', 'spi']]

ranking_df.head()
# combine all involved features

aggregate = standard_df.merge(shooting_df, left_on='Squad', right_on='Squad', suffixes=('_std', '_sht'))

aggregate = aggregate.merge(possession_df, left_on='Squad', right_on='Squad', suffixes=('', '_psn'))

aggregate = aggregate.merge(playingTime_df, left_on='Squad', right_on='Squad', suffixes=('', '_pTime'))

aggregate = aggregate.merge(passing_df, left_on='Squad', right_on='Squad', suffixes=('', '_psg'))

aggregate = aggregate.merge(goalkeeping_df, left_on='Squad', right_on='Squad', suffixes=('', '_gkp'))

aggregate = aggregate.merge(goalShotCreation_df, left_on='Squad', right_on='Squad', suffixes=('', '_gsc'))

aggregate = aggregate.merge(defence_df, left_on='Squad', right_on='Squad', suffixes=('', '_dfc'))

aggregate = aggregate.merge(adv_goalkeeping_df, left_on='Squad', right_on='Squad', suffixes=('', '_agk'))
for i in range(len(aggregate)):

    x = str(aggregate['Squad'].iloc[i]).split(' ')

    if len(x) == 2:

        aggregate['Squad'].iloc[i] = x[1]

    elif len(x) == 3:

        aggregate['Squad'].iloc[i] = x[1] + ' ' + x[2]
def name_changer(df, col):

    df[col].loc[df[col]== 'Atlético Madrid'] = 'Atletico Madrid'

    df[col].loc[df[col]== 'Dortmund'] = 'Borussia Dortmund'

    df[col].loc[df[col]== 'Inter'] = 'Internazionale'

    df[col].loc[df[col]== 'Leverkusen'] = 'Bayer Leverkusen'

    df[col].loc[df[col]== 'Paris S-G'] = 'Paris Saint-Germain'

    df[col].loc[df[col]== 'RB Salzburg'] = 'FC Salzburg'

    df[col].loc[df[col]== 'Red Star'] = 'Red Star Belgrade'

    df[col].loc[df[col]== 'Shakhtar'] = 'Shakhtar Donetsk'

    df[col].loc[df[col]== 'Tottenham'] = 'Tottenham Hotspur'

    df[col].loc[df[col]== 'Zenit'] = 'Zenit St Petersburg'

    df[col].loc[df[col]== 'Loko Moscow'] = 'Lokomotiv Moscow'
name_changer(aggregate, 'Squad')
# aggregate = aggregate.merge(ranking_df, left_on='Squad', right_on='name')

# aggregate = aggregate.drop(columns=['name'])
aggregate.columns
# preview

import seaborn as sns



mask = np.tril(aggregate.corr())

sns.heatmap(aggregate.corr(), mask=mask)
fixtures = pd.read_csv('../input/soccer-reference/fixtures.csv')[:-6]

fixtures = fixtures[['Home', 'Score', 'Away']].dropna()

fixtures['Home_Score'] = 0

fixtures['Away_Score'] = 0

for i in range(len(fixtures)):

    x = str(fixtures['Score'].iloc[i]).split('–')

    fixtures['Home_Score'].iloc[i] = x[0]

    fixtures['Away_Score'].iloc[i] = x[1]
fixtures['Score'].value_counts().plot.barh()
Score = fixtures['Score']
for i in range(len(fixtures)):

    x = str(fixtures['Home'].iloc[i]).split(' ')

    y = str(fixtures['Away'].iloc[i]).split(' ')

    if len(x) == 2:

        fixtures['Home'].iloc[i] = x[0]

    elif len(x) == 3:

        fixtures['Home'].iloc[i] = x[0] + " " + x[1]

        

    if len(y) == 2:

        fixtures['Away'].iloc[i] = y[1]

    elif len(y) == 3:

        fixtures['Away'].iloc[i] = y[1] + " " + y[2]
name_changer(fixtures, 'Home')

name_changer(fixtures, 'Away')
fixtures = fixtures.merge(aggregate, left_on='Home', right_on='Squad')

fixtures = fixtures.merge(aggregate, left_on='Away', right_on='Squad', suffixes=('_Home', '_Away'))
fixtures = fixtures.drop(columns=['Squad_Home', 'Squad_Away', 'Score'])

fixtures.head()
fixtures[['Home_Score', 'Away_Score']].apply(pd.Series.value_counts).sum(axis=1)
future_games = pd.read_csv('../input/soccer-reference/fixtures.csv')[-6:][['Home', 'Away']].dropna()



for i in range(len(future_games)):

    x = str(future_games['Home'].iloc[i]).split(' ')

    y = str(future_games['Away'].iloc[i]).split(' ')

    if len(x) == 2:

        future_games['Home'].iloc[i] = x[0]

    elif len(x) == 3:

        future_games['Home'].iloc[i] = x[0] + " " + x[1]

        

    if len(y) == 2:

        future_games['Away'].iloc[i] = y[1]

    elif len(y) == 3:

        future_games['Away'].iloc[i] = y[1] + " " + y[2]

        

name_changer(future_games, 'Home')

name_changer(future_games, 'Away')



future_games = future_games.merge(aggregate, left_on='Home', right_on='Squad')

future_games = future_games.merge(aggregate, left_on='Away', right_on='Squad', suffixes=('_Home', '_Away'))

future_games = future_games.drop(columns=['Squad_Home', 'Squad_Away'])
future_games
from sklearn.decomposition import PCA



# squads = aggregate['Squad']

X = fixtures.drop(columns=['Home', 'Away', 'Home_Score', 'Away_Score'])

for k in range(2, 20, 2):

    pca = PCA(n_components=k, svd_solver='randomized')

    pca.fit(X)

    print("the cumulative variance of {} PCs is {}".format(k, pca.explained_variance_ratio_.sum()))
from sklearn.linear_model import LogisticRegression

from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error



X = fixtures.drop(columns=['Home', 'Away', 'Home_Score', 'Away_Score'])

y = fixtures[['Home_Score', 'Away_Score']].astype(int)

pca = PCA(n_components=14, svd_solver='randomized')

X = pca.fit_transform(X)

clf = MultiOutputRegressor(LogisticRegression(random_state=0)).fit(X, y)



mean_squared_error(clf.predict(X), y)
import xgboost as xgb



xgb_model = MultiOutputRegressor(xgb.XGBClassifier(random_state=42))

xgb_model.fit(X, y)

y_pred = xgb_model.predict(X)

mse=mean_squared_error(y, y_pred)

mse
from sklearn.tree import DecisionTreeClassifier



clf = MultiOutputRegressor(DecisionTreeClassifier(random_state=0))

clf.fit(X, y)

mean_squared_error(y, clf.predict(X))
from sklearn.mixture import GaussianMixture



gmm = MultiOutputRegressor(GaussianMixture(n_components=2))

gmm.fit(X, y)

mean_squared_error(y, gmm.predict(X))
from sklearn.model_selection import KFold



def CVKFold(k, X, y, model):

    np.random.seed(1)

    #reproducibility

    

    highest_accuracy = float('inf')



    kf = KFold(n_splits = k,shuffle =True)

    #CV loop

    

    test_accuracies = []

    

    for train_index,test_index in kf.split(X):#generation of the sets

    #generate the sets    

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        multi_model = MultiOutputRegressor(model)

        # model fitting

        multi_model.fit(X_train,y_train)

        y_test_pred = multi_model.predict(X_test)

    

        test_accuracy = mean_squared_error(y_test_pred, y_test)

        test_accuracies.append(test_accuracy)

        

    print(multi_model.get_params()['estimator'])

    print("The average accuracy is " + str(sum(test_accuracies) / len(test_accuracies)))

    print()
models = [LogisticRegression(random_state=0), xgb.XGBClassifier(random_state=42), DecisionTreeClassifier(random_state=0), GaussianMixture(n_components=2)]



for model in models:

    CVKFold(5, X, y, model)
from bayes_opt import BayesianOptimization

from sklearn.model_selection import cross_val_score



def xgb_cv(eta, min_child_weight, max_depth, gamma, subsample, colsample_bytree, alpha, n_estimators, learning_rate):

    val = np.mean(cross_val_score(MultiOutputRegressor(xgb.XGBClassifier(eta=float(eta), 

                                                   min_child_weight=float(min_child_weight),

                                                   max_depth=int(max_depth),

                                                   gamma=float(gamma),

                                                   subsample=float(subsample),

                                                   colsample_bytree=float(colsample_bytree),

                                                   alpha=float(alpha),

                                                   n_estimators=int(n_estimators),

                                                   learning_rate=learning_rate,

                                                   seed=42)),

                         X, y, cv=5))

    

    return val
bo = BayesianOptimization(

             xgb_cv,

             {'eta': (0.01, 0.3),

             'min_child_weight': (1, 25),

             'max_depth': (3, 10),

             'gamma': (0.0, 1.0),

             'subsample': (0.5, 1),

             'colsample_bytree': (0.5, 1),

             'alpha': (0.0, 2.0),

             'n_estimators': (10, 100),

             'learning_rate': (0.0001, 0.1)})
bo.maximize()
bo.max
xgb_optimized = xgb.XGBClassifier(alpha=bo.max['params']['alpha'], colsample_bytree=bo.max['params']['colsample_bytree'], eta=bo.max['params']['eta'], 

                                  gamma=bo.max['params']['gamma'], max_depth=int(bo.max['params']['max_depth']), 

                                  min_child_weight=bo.max['params']['min_child_weight'], subsample=bo.max['params']['subsample'], 

                                  n_estimators=int(bo.max['params']['n_estimators']),

                                  learning_rate=bo.max['params']['learning_rate'], random_state=42)

CVKFold(5, X, y, xgb_optimized)
future_games_pca = pca.transform(future_games.drop(columns=['Home', 'Away']))
xgb_optimized = MultiOutputRegressor(xgb_optimized)

xgb_optimized.fit(X, y)

predicted_results = xgb_optimized.predict(future_games_pca)
predicted_results[:4]
def winner_judgment(fixtures, home, away, legs=False, home_prev=None, away_prev=None):

    '''

    input:

    home: the scores of home team at the current fixture

    away: the scores of away team at the current fixture

    legs: flag if it is a two-leg competition

    home_prev: the scores of home team in the previous battle if there exists one

    away_prev: the scores of away team in the previous battle if there exists one

    '''

    if legs is not True:

        if home > away:

            return fixtures['Home']

        elif home < away:

            return fixtures['Away']

        else:

            return fixtures['Away'] if random.random() > 0.5 else fixtures['Home']

    else:

        if home + home_prev > away + away_prev:

            return fixtures['Home']

        elif home + home_prev < away + away_prev:

            return fixtures['Away']

        elif home_prev > away:

            return fixtures['Home']

        else:

            return fixtures['Away'] if random.random() > 0.5 else fixtures['Home']
leg_1 = None

for i in range(4):

    leg_1 = pd.concat([leg_1, fixtures[(fixtures['Home']==future_games['Away'].iloc[i]) & (fixtures['Away']==future_games['Home'].iloc[i])]], ignore_index=True)
leg_1 = leg_1[['Home', 'Away', 'Home_Score', 'Away_Score']]
winners = []

for i in range(4):

    winners.append(winner_judgment(future_games[['Home', 'Away']].iloc[i], predicted_results[i][0], predicted_results[i][1], True, 

                                   int(leg_1['Away_Score'].iloc[i]), int(leg_1['Home_Score'].iloc[i])))
winners
# quarter finals



quarterFinals = pd.DataFrame({'Home': [winners[1], 'Atletico Madrid', winners[3], 'Paris Saint-Germain'],

                             'Away': [winners[0], 'RB Leipzig', winners[2], 'Atalanta']})

quarterFinals = quarterFinals.merge(aggregate, left_on='Home', right_on='Squad')

quarterFinals = quarterFinals.merge(aggregate, left_on='Away', right_on='Squad', suffixes=('_Home', '_Away'))

quarterFinals = quarterFinals.drop(columns=['Squad_Home', 'Squad_Away'])

quarterFinals
quarterFinals_pca = pca.transform(quarterFinals.drop(columns=['Home', 'Away']))

quarterFinalPrediction = xgb_optimized.predict(quarterFinals_pca)
quarterFinalPrediction
semi_qualifiers = []

for i in range(4):

    semi_qualifiers.append(winner_judgment(quarterFinals[['Home', 'Away']].iloc[i], quarterFinalPrediction[i][0], quarterFinalPrediction[i][1]))
semi_qualifiers
# semi-finals



semiFinals = pd.DataFrame({'Home': [semi_qualifiers[2], semi_qualifiers[3]],

                             'Away': [semi_qualifiers[0], semi_qualifiers[1]]})

semiFinals = semiFinals.merge(aggregate, left_on='Home', right_on='Squad')

semiFinals = semiFinals.merge(aggregate, left_on='Away', right_on='Squad', suffixes=('_Home', '_Away'))

semiFinals = semiFinals.drop(columns=['Squad_Home', 'Squad_Away'])

semiFinals
semiFinals_pca = pca.transform(semiFinals.drop(columns=['Home', 'Away']))

semiFinalsPrediction = xgb_optimized.predict(semiFinals_pca)

semiFinalsPrediction
finalists = []

for i in range(2):

    finalists.append(winner_judgment(semiFinals[['Home', 'Away']].iloc[i], semiFinalsPrediction[i][0], semiFinalsPrediction[i][1]))
finalists
# Final



Finals = pd.DataFrame({'Home': [finalists[1]],

                             'Away': [finalists[0]]})

Finals = Finals.merge(aggregate, left_on='Home', right_on='Squad')

Finals = Finals.merge(aggregate, left_on='Away', right_on='Squad', suffixes=('_Home', '_Away'))

Finals = Finals.drop(columns=['Squad_Home', 'Squad_Away'])

Finals
Finals_pca = pca.transform(Finals.drop(columns=['Home', 'Away']))

FinalPrediction = xgb_optimized.predict(Finals_pca)
FinalPrediction
print('19/20 UCL:')

print(winner_judgment(Finals[['Home', 'Away']].iloc[0], FinalPrediction[0][0], FinalPrediction[0][1]))
fixtures['Score'] = Score.reset_index()['Score']

# fixtures = fixtures.dropna(subset=['Score'])
fixture_info = fixtures[['Home', 'Away', 'Home_Score', 'Away_Score']]

fixtures = fixtures.drop(columns=['Home', 'Away', 'Home_Score', 'Away_Score'])
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

fixtures['Score'] = le.fit_transform(fixtures['Score'].astype(str))
from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler(random_state=42)

fixtures, Score_ = ros.fit_resample(fixtures.drop(columns=['Score']), fixtures['Score'])
fixtures['Score'] = le.inverse_transform(Score_)
fixtures['Score'].value_counts().plot.barh()
fixtures['Home_Score'] = 0

fixtures['Away_Score'] = 0

for i in range(len(fixtures)):

    x = str(fixtures['Score'].iloc[i]).split('–')

    fixtures['Home_Score'].iloc[i] = x[0]

    fixtures['Away_Score'].iloc[i] = x[1]
X = fixtures.drop(columns=['Score', 'Home_Score', 'Away_Score'])

y = fixtures[['Home_Score', 'Away_Score']].astype(int)



X = pca.fit_transform(X) 
for model in models:

    CVKFold(5, X, y, model)
def xgb_cv(eta, min_child_weight, max_depth, gamma, subsample, colsample_bytree, alpha, n_estimators, learning_rate):

    val = np.mean(cross_val_score(MultiOutputRegressor(xgb.XGBClassifier(eta=float(eta), 

                                                   min_child_weight=float(min_child_weight),

                                                   max_depth=int(max_depth),

                                                   gamma=float(gamma),

                                                   subsample=float(subsample),

                                                   colsample_bytree=float(colsample_bytree),

                                                   alpha=float(alpha),

                                                   n_estimators=int(n_estimators),

                                                   learning_rate=learning_rate,

                                                   seed=42)),

                         X, y, cv=5))

    

    return val
overSampled_bo = BayesianOptimization(

             xgb_cv,

             {'eta': (0.01, 0.3),

             'min_child_weight': (1, 25),

             'max_depth': (3, 5),

             'gamma': (0.0, 1.0),

             'subsample': (0.5, 1),

             'colsample_bytree': (0.5, 1),

             'alpha': (0.0, 2.0),

             'n_estimators': (90, 100),

             'learning_rate': (0.0001, 0.1)})
overSampled_bo.maximize()
overSampled_bo.max
overSampled_xgb_optimized = xgb.XGBClassifier(alpha=overSampled_bo.max['params']['alpha'], colsample_bytree=overSampled_bo.max['params']['colsample_bytree'], 

                                              eta=overSampled_bo.max['params']['eta'], gamma=overSampled_bo.max['params']['gamma'], 

                                              max_depth=int(overSampled_bo.max['params']['max_depth']), min_child_weight=overSampled_bo.max['params']['min_child_weight'], 

                                              subsample=overSampled_bo.max['params']['subsample'], n_estimators=int(overSampled_bo.max['params']['n_estimators']),

                                              learning_rate=overSampled_bo.max['params']['learning_rate'], random_state=42)

CVKFold(5, X, y, overSampled_xgb_optimized)
overSampled_xgb_optimized = MultiOutputRegressor(overSampled_xgb_optimized)

overSampled_xgb_optimized.fit(X, y)

predicted_results = overSampled_xgb_optimized.predict(future_games_pca)

predicted_results[:4]
winners = []

for i in range(4):

    winners.append(winner_judgment(future_games[['Home', 'Away']].iloc[i], predicted_results[i][0], predicted_results[i][1], True, 

                                   int(leg_1['Away_Score'].iloc[i]), int(leg_1['Home_Score'].iloc[i])))
winners
quarterFinals = pd.DataFrame({'Home': [winners[1], 'Atletico Madrid', winners[3], 'Paris Saint-Germain'],

                             'Away': [winners[0], 'RB Leipzig', winners[2], 'Atalanta']})

quarterFinals = quarterFinals.merge(aggregate, left_on='Home', right_on='Squad')

quarterFinals = quarterFinals.merge(aggregate, left_on='Away', right_on='Squad', suffixes=('_Home', '_Away'))

quarterFinals = quarterFinals.drop(columns=['Squad_Home', 'Squad_Away'])

quarterFinals
quarterFinals_pca = pca.transform(quarterFinals.drop(columns=['Home', 'Away']))

quarterFinalPrediction = overSampled_xgb_optimized.predict(quarterFinals_pca)
quarterFinalPrediction
semi_qualifiers = []

for i in range(4):

    semi_qualifiers.append(winner_judgment(quarterFinals[['Home', 'Away']].iloc[i], quarterFinalPrediction[i][0], quarterFinalPrediction[i][1]))
semi_qualifiers
semiFinals = pd.DataFrame({'Home': [semi_qualifiers[2], semi_qualifiers[3]],

                             'Away': [semi_qualifiers[0], semi_qualifiers[1]]})

semiFinals = semiFinals.merge(aggregate, left_on='Home', right_on='Squad')

semiFinals = semiFinals.merge(aggregate, left_on='Away', right_on='Squad', suffixes=('_Home', '_Away'))

semiFinals = semiFinals.drop(columns=['Squad_Home', 'Squad_Away'])

semiFinals
semiFinals_pca = pca.transform(semiFinals.drop(columns=['Home', 'Away']))

semiFinalsPrediction = overSampled_xgb_optimized.predict(semiFinals_pca)

semiFinalsPrediction
finalists = []

for i in range(2):

    finalists.append(winner_judgment(semiFinals[['Home', 'Away']].iloc[i], semiFinalsPrediction[i][0], semiFinalsPrediction[i][1]))
finalists
Finals = pd.DataFrame({'Home': [finalists[1]],

                             'Away': [finalists[0]]})

Finals = Finals.merge(aggregate, left_on='Home', right_on='Squad')

Finals = Finals.merge(aggregate, left_on='Away', right_on='Squad', suffixes=('_Home', '_Away'))

Finals = Finals.drop(columns=['Squad_Home', 'Squad_Away'])

Finals
Finals_pca = pca.transform(Finals.drop(columns=['Home', 'Away']))

FinalPrediction = overSampled_xgb_optimized.predict(Finals_pca)
FinalPrediction
print('19/20 UCL:')

print(winner_judgment(Finals[['Home', 'Away']].iloc[0], FinalPrediction[0][0], FinalPrediction[0][1]))
proba = {'Manchester City': {'qtr': 0, 'semi': 0, 'final': 0, 'champion': 0},

        'Bayern Munich': {'qtr': 0, 'semi': 0, 'final': 0, 'champion': 0},

        'Paris Saint-Germain': {'qtr': 500, 'semi': 0, 'final': 0, 'champion': 0},

        'Real Madrid': {'qtr': 0, 'semi': 0, 'final': 0, 'champion': 0},

        'Juventus': {'qtr': 0, 'semi': 0, 'final': 0, 'champion': 0},

        'Lyon': {'qtr': 0, 'semi': 0, 'final': 0, 'champion': 0},

        'Barcelona': {'qtr': 0, 'semi': 0, 'final': 0, 'champion': 0},

        'Napoli': {'qtr': 0, 'semi': 0, 'final': 0, 'champion': 0},

        'Chelsea': {'qtr': 0, 'semi': 0, 'final': 0, 'champion': 0},

        'Atalanta': {'qtr': 500, 'semi': 0, 'final': 0, 'champion': 0},

        'RB Leipzig': {'qtr': 500, 'semi': 0, 'final': 0, 'champion': 0},

        'Atletico Madrid': {'qtr': 500, 'semi': 0, 'final': 0, 'champion': 0}}
def simulate():

    overSampled_bo = BayesianOptimization(

             xgb_cv,

             {'eta': (0.01, 0.3),

             'min_child_weight': (1, 25),

             'max_depth': (3, 5),

             'gamma': (0.0, 1.0),

             'subsample': (0.5, 1),

             'colsample_bytree': (0.5, 1),

             'alpha': (0.0, 2.0),

             'n_estimators': (90, 100),

             'learning_rate': (0.0001, 0.1)})

    overSampled_bo.maximize()

    overSampled_xgb_optimized = xgb.XGBClassifier(alpha=overSampled_bo.max['params']['alpha'], colsample_bytree=overSampled_bo.max['params']['colsample_bytree'], 

                                                  eta=overSampled_bo.max['params']['eta'], gamma=overSampled_bo.max['params']['gamma'], 

                                                  max_depth=int(overSampled_bo.max['params']['max_depth']), min_child_weight=overSampled_bo.max['params']['min_child_weight'], 

                                                  subsample=overSampled_bo.max['params']['subsample'], n_estimators=int(overSampled_bo.max['params']['n_estimators']),

                                                  learning_rate=overSampled_bo.max['params']['learning_rate'], random_state=42)

    overSampled_xgb_optimized = MultiOutputRegressor(overSampled_xgb_optimized)

    overSampled_xgb_optimized.fit(X, y)

    # round of 16

    predicted_results = overSampled_xgb_optimized.predict(future_games_pca)

    winners = []

    for i in range(4):

        winner = winner_judgment(future_games[['Home', 'Away']].iloc[i], predicted_results[i][0], predicted_results[i][1], True, 

                                       int(leg_1['Away_Score'].iloc[i]), int(leg_1['Home_Score'].iloc[i]))

        winners.append(winner)

        proba[winner]['qtr'] += 1



    # quarter finals

    quarterFinals = pd.DataFrame({'Home': [winners[1], 'Atletico Madrid', winners[3], 'Paris Saint-Germain'],

                                 'Away': [winners[0], 'RB Leipzig', winners[2], 'Atalanta']})

    quarterFinals = quarterFinals.merge(aggregate, left_on='Home', right_on='Squad')

    quarterFinals = quarterFinals.merge(aggregate, left_on='Away', right_on='Squad', suffixes=('_Home', '_Away'))

    quarterFinals = quarterFinals.drop(columns=['Squad_Home', 'Squad_Away'])

    quarterFinals_pca = pca.transform(quarterFinals.drop(columns=['Home', 'Away']))

    quarterFinalPrediction = overSampled_xgb_optimized.predict(quarterFinals_pca)

    semi_qualifiers = []

    for i in range(4):

        semi_quafier = winner_judgment(quarterFinals[['Home', 'Away']].iloc[i], quarterFinalPrediction[i][0], quarterFinalPrediction[i][1])

        semi_qualifiers.append(semi_quafier)

        proba[semi_quafier]['semi'] += 1



    # semi finals

    semiFinals = pd.DataFrame({'Home': [semi_qualifiers[2], semi_qualifiers[3]],

                                 'Away': [semi_qualifiers[0], semi_qualifiers[1]]})

    semiFinals = semiFinals.merge(aggregate, left_on='Home', right_on='Squad')

    semiFinals = semiFinals.merge(aggregate, left_on='Away', right_on='Squad', suffixes=('_Home', '_Away'))

    semiFinals = semiFinals.drop(columns=['Squad_Home', 'Squad_Away'])

    semiFinals_pca = pca.transform(semiFinals.drop(columns=['Home', 'Away']))

    semiFinalsPrediction = overSampled_xgb_optimized.predict(semiFinals_pca)

    finalists = []

    for i in range(2):

        finalist = winner_judgment(semiFinals[['Home', 'Away']].iloc[i], semiFinalsPrediction[i][0], semiFinalsPrediction[i][1])

        finalists.append(finalist)

        proba[finalist]['final'] += 1



    # final

    Finals = pd.DataFrame({'Home': [finalists[1]],

                                 'Away': [finalists[0]]})

    Finals = Finals.merge(aggregate, left_on='Home', right_on='Squad')

    Finals = Finals.merge(aggregate, left_on='Away', right_on='Squad', suffixes=('_Home', '_Away'))

    Finals = Finals.drop(columns=['Squad_Home', 'Squad_Away'])

    Finals_pca = pca.transform(Finals.drop(columns=['Home', 'Away']))

    FinalPrediction = overSampled_xgb_optimized.predict(Finals_pca)

    champion = winner_judgment(Finals[['Home', 'Away']].iloc[0], FinalPrediction[0][0], FinalPrediction[0][1])

    proba[champion]['champion'] += 1
for i in range(500):

    simulate()
teams = [_ for _ in proba] * 4

stages =  np.reshape([[_] * 12 for _ in proba['Lyon']], (1, 48)).tolist()[0]
for team in proba.items():

  for stage in team[1].items():

    proba[team[0]][stage[0]] = proba[team[0]][stage[0]] / 500

    

possibilities = pd.DataFrame({'Team': teams,

                              'Stage': stages,

                              'Possibility': [proba[teams[i]][stages[i]] for i in range(48)]})

possibilities = possibilities.pivot("Team", "Stage", "Possibility")
possibilities = possibilities.rename(columns={'champion': 'Champion', 'final': 'Final', 'qtr': 'Quarter Final', 'semi': 'Semi Final'})

possibilities = possibilities.sort_values(by=['Champion', 'Final', 'Quarter Final', 'Semi Final'], ascending=False)

possibilities = possibilities[['Quarter Final', 'Semi Final', 'Final', 'Champion']]
import seaborn as sns



ax = sns.heatmap(possibilities, annot=True, cmap="YlGnBu")