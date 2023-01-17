import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

sns.set_style('darkgrid')

import plotly.express as px

from IPython.display import Image
# Importing and Exploring the data

fifa_df=pd.read_csv('../input/fifa-20-complete-player-dataset/players_20.csv')

fifa_df.head(3).T
fifa_df.describe().T
fifa_df.columns
fifa_df.drop(['sofifa_id', 'player_url', 'long_name', 'dob', 'nationality', 

              'joined', 'contract_valid_until', 'loaned_from', 'nation_jersey_number',

              'nation_position', 'player_tags', 'player_traits', 'team_jersey_number',

              'player_positions', 'release_clause_eur', 'real_face', 'body_type', 'ls', 'st', 'rs',

              'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm',

              'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb',

              'rcb', 'rb'], axis=1,inplace=True)
metrics = ['team_position', 'short_name', 'age', 'height_cm', 'weight_kg', 'club', 'overall',

       'potential', 'value_eur', 'wage_eur', 'preferred_foot',

       'international_reputation', 'weak_foot', 'skill_moves', 'work_rate']

fifa_df[metrics].isnull().sum()
# Eliminating players with unknown positions

fifa_df.drop(fifa_df[fifa_df['team_position'].isnull()].index, inplace=True)
#Renaming some features

fifa_df.rename(columns={'short_name': 'name', 'height_cm': 'height', 'weight_kg': 'weight',

       'value_eur': 'value', 'wage_eur': 'wage', 'team_position': 'position', 'attacking_crossing': 'crossing',

       'attacking_finishing': 'finishing', 'attacking_heading_accuracy': 'heading_accuracy',

       'skill_fk_accuracy': 'fk_accuracy', 'skill_long_passing': 'long_passing',

       'skill_ball_control': 'ball_control', 'movement_acceleration': 'acceleration', 'movement_sprint_speed': 'sprint_speed',

       'mentality_penalties': 'penalties_accuracy', 'power_shot_power': 'shot_power', 'power_jumping': 'jumping', 

       'power_stamina': 'stamina', 'power_strength': 'strength','power_long_shots': 'long_shots', 'defending_marking': 'marking'}, inplace=True)

fifa_df.columns
metrics = ['overall', 'potential', 'value', 'wage', 'finishing', 'dribbling', 'mentality_vision',

           'fk_accuracy', 'shot_power', 'penalties_accuracy', 'pace', 'passing', 'defending', 'marking']



for i in metrics:

    best_Overall = fifa_df.loc[fifa_df[i] == fifa_df[i].max(), ['name', i]].values.tolist()

    print('Top', i, ': ', best_Overall[0][0], '-->', best_Overall[0][1])
sns.lmplot(x = 'pace', y = 'movement_balance', data = fifa_df, col = 'preferred_foot',scatter_kws = {'alpha':0.3,'color':'orange'},

           line_kws={'color':'red'})
sns.lmplot(x = 'penalties_accuracy', y = 'fk_accuracy', data = fifa_df, col = 'preferred_foot',scatter_kws = {'alpha':0.3,'color':'blue'},

           line_kws={'color':'red'})
sns.jointplot(x="finishing", y="shooting", data=fifa_df, kind="hex", color="#4CB391")

sns.jointplot(x="dribbling", y="movement_balance", data=fifa_df, kind="kde", space=0, color="blue")

sns.jointplot(x="dribbling", y="pace", data=fifa_df, kind="kde", space=0, color="red")
sns.jointplot(x="long_shots", y="fk_accuracy", data=fifa_df, kind="kde", space=0, color="g")
fig = plt.gcf()

fig.set_size_inches(12, 9)



sns.boxenplot(fifa_df['overall'], fifa_df['age'], palette = 'rocket')
fig = plt.gcf()

fig.set_size_inches(12, 9)



sns.lineplot(x='age', y='stamina', data=fifa_df, legend='brief', label='stamina')

sns.lineplot(x='age', y='dribbling', data=fifa_df, legend='brief', label='dribbling')

sns.lineplot(x='age', y='pace', data=fifa_df, legend='brief', label='pace')

sns.lineplot(x='age', y='passing', data=fifa_df, legend='brief', label='passing')
fig = px.bar(fifa_df, x='age', y='value',

             hover_data=['age', 'value', 'overall'], color='value',

             labels={'pop':'Players Market Value Based On Their Age'}, height=600, title='Players Value Based On Their Age')

fig.show()
encoding = {"preferred_foot": {"Left": 1, "Right": 0},

      'position': {'RW': 'ST', 'LW': 'ST', 'CAM': 'AM', 'GK': 'GK', 'RCM': 'MF', 'LCB': 'DF', 'ST': 'ST', 'CDM': 'DM', 'LDM': 'DM', 'RM': 'MF',

                   'RCB': 'DF', 'LCM': 'MF', 'LM': 'MF', 'CF': 'ST', 'SUB': 'SUB', 'LB': 'DF', 'LS': 'ST', 'RB': 'DF', 'RDM': 'DM', 'RES': 'RES', 'RAM': 'AM',

                   'RS': 'ST', 'RF': 'ST', 'CM': 'MF', 'LF': 'ST', 'CB': 'DF', 'LAM': 'AM', 'RWB': 'DF', 'LWB': 'DF'}

     }

# Unify the Players postions as GK | DF | DM | MF | AM | SF

# Apply Ordinal Encoding on preferred_foot column

fifa_df.replace(encoding, inplace=True)
fifa_df.drop(['club', 'weight', 'height', 'wage', 'name', 'pace',

       'shooting', 'passing', 'dribbling', 'defending', 'physic', 'gk_diving',

       'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed',

       'gk_positioning'], axis=1, inplace=True)
df = fifa_df.dropna()

OHE = pd.get_dummies(fifa_df.position)

df = pd.concat([fifa_df,OHE], axis=1)

work_rate = pd.get_dummies(fifa_df.work_rate)

df = pd.concat([df,work_rate], axis=1)

df.head(10)
df.drop(['work_rate', 'position'], axis=1, inplace=True)
# Defining Label Field & Remove it From The Data

label = df['overall'].values

df.drop(['overall'], axis=1, inplace=True)
# Split the Data into 70% - 30%

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.3)
# Check Data Size

print(X_test.shape,X_train.shape)

print(y_test.shape,y_train.shape)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score





model_LR = LinearRegression() # Initialize Linear Model





# Using Grid Search for tuning Hyperparameters

parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

grid_LR = GridSearchCV(model_LR,parameters, cv=None)

grid_LR.fit(X_train, y_train)

predictions = grid_LR.predict(X_test)





# Check Best Parameter for the Model

print("Best Parameters for the Model: ", grid_LR.best_params_)





# Check Model Score

print("\nR2: ", grid_LR.best_score_)

print("Residual sum of squares: ",  np.mean((predictions - y_test) ** 2))

print('RMSE: '+str(np.sqrt(mean_squared_error(y_test, predictions))))
from eli5.sklearn import PermutationImportance

import eli5

perm = PermutationImportance(grid_LR.best_estimator_, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
# Plot The Fitted Model

plt.figure(figsize=(19,10))

sns.regplot(predictions,y_test, marker="+", line_kws={'color':'darkred','alpha':1.0})
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import make_scorer



model_GB = GradientBoostingRegressor() # Initialize GB Model



# Scoring Metrics that qill be used in fitting the model.

scoring_metrics = {'accuracy': make_scorer(accuracy_score),

           'precision': make_scorer(precision_score),'recall':make_scorer(recall_score)}



# Using Grid Search for tuning Hyperparameters

parameters = {

    

    "learning_rate": [0.01, 0.05, 0.1, 0.2],

    "max_depth":[3, 6, 10],

    "n_estimators":[100],

    'min_samples_leaf': [5,10],

    'min_samples_split': [5,10]

}



# Using GridSearch for tuning the HyperParameters

grid_GB = GridSearchCV(model_GB, parameters, cv=10, n_jobs=-1)

grid_GB.fit(X_train, y_train)

predictions = grid_GB.predict(X_test)





# Check Best Parameter for the Model

print("Best Parameters for the Model: ", grid_GB.best_params_)





# Check Model Score

print("\nR2: ", grid_GB.best_score_)

print("Residual sum of squares: ",  np.mean((predictions - y_test) ** 2))

print('RMSE: '+str(np.sqrt(mean_squared_error(y_test, predictions))))
perm = PermutationImportance(grid_GB.best_estimator_, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
plt.figure(figsize=(19,10))

sns.regplot(predictions,y_test, marker="+", line_kws={'color':'darkred','alpha':1.0})
parameter = pd.DataFrame({'Gradient Boosting':pd.Series(grid_GB.best_params_),'Linear Regression':pd.Series(grid_LR.best_params_)})

score = pd.DataFrame({'Gradient Boosting': grid_GB.best_score_, 'Learning Regression':grid_LR.best_score_}, index=[0])
score
parameter