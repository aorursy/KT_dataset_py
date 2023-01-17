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
train_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/test.csv', index_col=0)
train_df
test_df
train_df = train_df.drop(['lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm','rm','ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf','rw',
                          'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'player_traits', 'nation_position',
                          'joined', 'team_position', 'work_rate', 'player_positions'
                         ], axis=1)
train_df
print(train_df.isnull().sum())
train_df = train_df.drop(['nation_jersey_number', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning'
                         ], axis=1)
train_df = train_df.fillna(train_df.mode())
train_df = train_df.fillna(train_df.median())
train_df = pd.get_dummies(train_df, drop_first=True)
train_df
test_df = test_df.drop(['lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm','rm','ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf','rw',
                          'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'player_traits', 'nation_position',
                          'joined', 'team_position', 'work_rate', 'player_positions'
                         ], axis=1)
test_df
test_df = test_df.drop(['nation_jersey_number', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning'
                         ], axis=1)
test_df
test_df = test_df.fillna(train_df.mode())
test_df = test_df.fillna(train_df.median())
test_df = pd.get_dummies(test_df, drop_first=True)
test_df
print(train_df.isnull().sum())
import matplotlib.pyplot as plt
import seaborn as sns

corr = train_df[['age', 'height_cm', 'weight_kg', 'overall', 'potential',
                'international_reputation', 'weak_foot', 'skill_moves', 'value_eur']].corr()

plt.style.use('ggplot')
plt.figure(figsize=(15, 15)) 
sns.heatmap(corr, square=True, annot=True)
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

corr = train_df[['team_jersey_number', 'contract_valid_until', 'pace', 'shooting', 'passing', 'dribbling',
                 'defending', 'physic', 'value_eur']].corr()

plt.style.use('ggplot')
plt.figure(figsize=(15, 15)) 
sns.heatmap(corr, square=True, annot=True)
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

corr = train_df[['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
                 'attacking_short_passing','attacking_volleys', 'skill_dribbling','skill_curve',
                 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'value_eur']].corr()

plt.style.use('ggplot')
plt.figure(figsize=(15, 15)) 
sns.heatmap(corr, square=True, annot=True)
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

corr = train_df[['movement_acceleration', 'movement_sprint_speed', 'movement_agility', 
                 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping', 
                 'power_stamina', 'power_strength', 'power_long_shots', 'value_eur']].corr()

plt.style.use('ggplot')
plt.figure(figsize=(15, 15)) 
sns.heatmap(corr, square=True, annot=True)
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

corr = train_df[['mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 
                 'mentality_penalties', 'mentality_composure', 'defending_marking', 'defending_standing_tackle', 
                 'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling','goalkeeping_kicking', 
                 'goalkeeping_positioning', 'goalkeeping_reflexes', 'preferred_foot_Right', 'loaned_yes','value_eur']].corr()

plt.style.use('ggplot')
plt.figure(figsize=(15, 15)) 
sns.heatmap(corr, square=True, annot=True)
plt.show()
train_df = train_df[['overall', 'potential', 'international_reputation', 'skill_moves',
                     'shooting', 'passing', 'mentality_vision', 'mentality_penalties', 'skill_long_passing',
                     'value_eur']]

test_df = test_df[['overall', 'potential', 'international_reputation', 'skill_moves',
                   'shooting', 'passing', 'mentality_vision', 'mentality_penalties', 'skill_long_passing']]
import matplotlib.pyplot as plt
import seaborn as sns

corr = train_df.corr()

plt.style.use('ggplot')
plt.figure(figsize=(15, 15)) 
sns.heatmap(corr, square=True, annot=True)
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('ggplot')
plt.figure()
sns.pairplot(train_df)
plt.show()
train_y = train_df['value_eur'].to_numpy()
train_X = train_df.drop('value_eur', axis=1).to_numpy()
X_test = test_df.to_numpy()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
X_test = sc.transform(X_test)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=0.2)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingRegressor

estimators = [
        ('svr', SVR()),
        ('mlp', MLPRegressor()),
        ('rfr', RandomForestRegressor()),
        ('xgb', xgb.XGBRegressor(random_state=0))
        ]

reg = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
)
reg.fit(X_train, y_train)
import lightgbm as lgb
reg = lgb.LGBMRegressor(random_state=0)
reg.fit(X_train, y_train)
from sklearn.metrics import mean_squared_log_error

y_pred = np.abs(reg.predict(X_valid))  # 予測
np.sqrt(mean_squared_log_error(y_valid, y_pred))  # 評価
from sklearn.metrics import mean_squared_log_error

y_pred = np.abs(reg.predict(train_X))  # 予測
np.sqrt(mean_squared_log_error(train_y, y_pred))  # 評価
p_test = np.abs(reg.predict(X_test))
p_test
submit_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/sampleSubmission.csv', index_col=0)
submit_df['value_eur'] = p_test
submit_df
submit_df.to_csv('submission9.csv')

