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
df_train = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/train.csv', index_col = 0)

df_test = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/test.csv', index_col = 0)
pd.set_option('display.max_rows', 150)

pd.set_option('display.max_columns', 100)
df_train['loaned'] = df_train['loaned'].map({'yes':1, 'no':0})

df_test['loaned'] = df_test['loaned'].map({'yes':1, 'no':0})

df_train['preferred_foot'] = df_train['preferred_foot'].map({'Right':1, 'Left':0})

df_test['preferred_foot'] = df_test['preferred_foot'].map({'Right':1, 'Left':0})
train_len = len(df_train)

for le in range(train_len):

    if df_train['joined'][le] is not np.nan:

        df_train['joined'][le] = df_train['joined'][le][0:4]

    else:

        df_train['joined'][le] = 0
test_len = len(df_test)

for le in range(test_len):

    if df_test['joined'][le + train_len] is not np.nan:

        df_test['joined'][le + train_len] = df_test['joined'][le + train_len][0:4]

    else:

        df_test['joined'][le + train_len] = 0
df_train['joined'] = df_train['joined'].astype(int)

df_test['joined'] = df_test['joined'].astype(int)
dummy_train = pd.get_dummies(df_train['team_position'])

dummy_test = pd.get_dummies(df_test['team_position'])
#dummy_train
#dummy[261:263]
#dummy[32:34]
#print(df_train[df_train["team_position"].isin(['CAM'])])
#df_train[df_train[["team_position"]].isnull().sum(axis=1) >= 1]
#dummy.info()
for col in dummy_train.columns:

    df_train[col] = dummy_train[col]

    df_test[col] = dummy_test[col]
dummy_train = pd.get_dummies(df_train['work_rate'],drop_first=True)

dummy_test = pd.get_dummies(df_test['work_rate'],drop_first=True)
for col in dummy_train.columns:

    df_train[col] = dummy_train[col]

    df_test[col] = dummy_test[col]
df_train.isnull().sum()
df_train.info()
import matplotlib.pyplot as plt # 可視化

import seaborn as sns # pltをラッパーした可視化

%matplotlib inline

sns.set() # snsでpltの設定をラッパー
target_col="value_eur"

sns.countplot(df_train[target_col])

plt.figure()

plt.plot()
max(df_train[target_col])
min(df_train[target_col])
drop_list = [

    'player_traits',

    'nation_position',

    'team_position',

    'work_rate',

    'player_positions',



]
df_train = df_train.drop(columns=drop_list)

df_test = df_test.drop(columns=drop_list)
object_cols = []

for col, types in zip(df_test.columns, df_test.dtypes):

    if types != "float" and types != "int":

        try:

            df_train[col] = df_train[col].astype(float)

            df_test[col] = df_test[col].astype(float)

        except:

            object_cols.append(col)
object_cols
train_len = len(df_train)

for col in object_cols:

    for le in range(train_len):

        if df_train[col][le] is not np.nan:

            df_train[col][le] = int(df_train[col][le][0:2]) + int(df_train[col][le][3:4])

        else:

            df_train[col][le] = 0
test_len = len(df_test)

for col in object_cols:

    for le in range(test_len):

        if df_test[col][le + train_len] is not np.nan:

            df_test[col][le + train_len] = int(df_test[col][le + train_len][0:2]) + int(df_test[col][le + train_len][3:4])

        else:

            df_test[col][le + train_len] = 0
df_train.fillna(0,inplace=True)

df_test.fillna(0,inplace=True)
print(df_train.corr()[target_col].sort_values())
X=df_train.drop([target_col],axis=1).values

y=df_train[target_col].values

X_test=df_test.values
from sklearn.feature_selection import RFE

from sklearn.tree import DecisionTreeRegressor

est = DecisionTreeRegressor(random_state=0)

fs  = RFE(est, n_features_to_select=5)

fs.fit(X, y)

X_  = fs.transform(X)
#np.isnan(df_train['ls'][1])
#df_train['ls'].fillna(0,inplace=True)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_, y, test_size=0.2, random_state=0)

len(X_train),len(X_valid),len(y_train),len(y_valid)
from sklearn.metrics import mean_squared_log_error

from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(random_state=0)

rfr.fit(X_train, y_train)

#rfr.fit(X_res,y_res)

predict = rfr.predict(X_valid)

np.sqrt(mean_squared_log_error(y_valid,predict))
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state=0)

dtr.fit(X_train, y_train)

#dtr.fit(X_res,y_res)

predict = dtr.predict(X_valid)

np.sqrt(mean_squared_log_error(y_valid,predict))
from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(random_state=0)

lgbm.fit(X_train, y_train)

#lgbm.fit(X_res,y_res)

predict = lgbm.predict(X_valid)

for i in range(len(predict)):

    if predict[i] < 0:

        predict[i]=0

np.sqrt(mean_squared_log_error(y_valid,predict))
# feature_importanceを求める

feature_importances = rfr.feature_importances_

print(feature_importances)
import matplotlib.pyplot as plt



plt.figure(figsize=(100, 5))

plt.ylim([0, 0.6])

y_ = feature_importances

x = np.arange(len(y_))

plt.bar(x, y_, align="center")

plt.xticks(x, df_train)

plt.show()
import optuna

def objective(trial):

    max_depth = trial.suggest_int('max_depth', 1, 30)

    min_samples_leaf = trial.suggest_int('min_samples_leaf',1,10)

    min_samples_split = trial.suggest_int('min_samples_split',2,5)

    model = DecisionTreeRegressor(criterion='mse', max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=0)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    return np.sqrt(mean_squared_log_error(y_valid, y_pred))



study = optuna.create_study()

study.optimize(objective, n_trials=100)

study.best_params
max_depth = study.best_params['max_depth']

min_samples_split = study.best_params['min_samples_split']

min_samples_leaf = study.best_params['min_samples_leaf']

model = DecisionTreeRegressor(criterion='mse', max_depth=max_depth,  min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=0)

model.fit(X_train, y_train)

predict = model.predict(X_valid)

np.sqrt(mean_squared_log_error(y_valid,predict))
#model = DecisionTreeRegressor(random_state=0)
model.fit(X,y)
p_test = model.predict(X_test)
submit_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/sampleSubmission.csv',index_col=0)

submit_df[target_col] = p_test

submit_df
submit_df.to_csv('submission.csv')