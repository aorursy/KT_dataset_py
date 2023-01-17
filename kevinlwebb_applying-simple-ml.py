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
from sqlalchemy import create_engine
import pandas as pd
engine  = create_engine("sqlite:///../input/soccer/database.sqlite")
tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", engine)
tables
team_df = pd.read_sql_table('Team', engine)
team_df.head()
country_df = pd.read_sql_table('Country', engine)
country_df.head()
league_df = pd.read_sql_table('League', engine)
league_df.head()
match_df = pd.read_sql_table('Match', engine)
match_df["month"] = match_df["date"].str.split("-").str[1]
match_df["result"] = 0
match_df.loc[match_df.home_team_goal < match_df.away_team_goal,"result"] = 0
match_df.loc[match_df.home_team_goal == match_df.away_team_goal,"result"] = 1
match_df.loc[match_df.home_team_goal > match_df.away_team_goal,"result"] = 2
reduced_match_df = match_df[["country_id","league_id","month","home_team_api_id","away_team_api_id","home_team_goal","away_team_goal","result"]].copy()
reduced_team_df = team_df[["team_api_id","team_long_name"]].copy()
reduced_league_df = league_df[["country_id","name"]].copy()

total_df = pd.merge(reduced_match_df, reduced_team_df,left_on=['home_team_api_id'],right_on= ['team_api_id'])
total_df["home_team_name"] = total_df.team_long_name
total_df.drop(columns=['team_long_name', 'home_team_api_id',"team_api_id"], inplace=True)

total_df = pd.merge(total_df, reduced_team_df,left_on=['away_team_api_id'],right_on= ['team_api_id'])
total_df["away_team_name"] = total_df.team_long_name
total_df.drop(columns=['team_long_name', 'away_team_api_id',"team_api_id"], inplace=True)

total_df = pd.merge(total_df, country_df,left_on=['country_id'],right_on= ['id'])
total_df["country_name"] = total_df.name
total_df.drop(columns=['name', 'country_id',"id"], inplace=True)

total_df = pd.merge(total_df, reduced_league_df,left_on=['league_id'],right_on= ['country_id'])
total_df["league_name"] = total_df.name
total_df.drop(columns=['name', 'league_id',"country_id"], inplace=True)

categorical = ["month","result","home_team_name","away_team_name","country_name","league_name"]
total_df[categorical] = total_df[categorical].astype(str)
total_df.reset_index(drop=True, inplace=True)
total_df
total_df.dtypes
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
# Read the data
df = total_df.copy()

# Remove rows with missing target
df.dropna(axis=0, subset=['result'], inplace=True)

# Separate target from predictors
y = df.result         
X = df.drop(['result'], axis=1)

# Break off validation set from training data
X_train_full, X_test_full, y_train, y_test = train_test_split(X, y,
                                                                train_size=0.8,
                                                                test_size=0.2,
                                                                random_state=0)

integer_features = list(X.columns[X.dtypes == 'int64'])
#continuous_features = list(X.columns[X.dtypes == 'float64'])
categorical_features = list(X.columns[X.dtypes == 'object'])

# Keep selected columns only
my_cols = categorical_features + integer_features
X_train = X_train_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

integer_transformer = Pipeline(steps = [
   ('imputer', SimpleImputer(strategy = 'most_frequent')),
   ('scaler', StandardScaler())])

# continuous_transformer = Pipeline(steps = [
#    ('imputer', SimpleImputer(strategy = 'most_frequent')),
#    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
   transformers=[
       ('ints', integer_transformer, integer_features),
       #('cont', continuous_transformer, continuous_features),
       ('cat', categorical_transformer, categorical_features)])

base = Pipeline(steps=[('preprocessor', preprocessor),
                     ('classifier', RandomForestClassifier())])

# Preprocessing of training data, fit model 
base.fit(X_train, y_train)

base.predict(X_test)
integer_features
categorical_features
