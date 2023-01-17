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
import pandas as pd

import numpy as np



from requests import get

from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
games = {}

headers = ({'User-Agent':

            'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'

            }

           )



for i in range(1):

    url = 'https://www.pro-football-reference.com/years/2020/week_'+str(i+1)+'.htm'

    r=get(url, headers=headers)

    week_html = BeautifulSoup(r.text, 'html.parser')

    table = week_html.select_one('div.game_summaries')

    

    

    

    for game in table.find_all("div"):

        date = game.select_one('table.teams').select_one('tr.date').text

        away_team = game.select_one('table.teams').select('tr')[1].select_one('a').text

        away_score = game.select_one('table.teams').select('tr')[1].select_one('td.right').text

        home_team = game.select_one('table.teams').select('tr')[2].select_one('a').text

        home_score = game.select_one('table.teams').select('tr')[2].select_one('td.right').text

        

        games.setdefault('date',[]).append(date)

        games.setdefault('away_team',[]).append(away_team)

        games.setdefault('away_score',[]).append(away_score)

        games.setdefault('home_team',[]).append(home_team)

        games.setdefault('home_score',[]).append(home_score)

        

for i in range(16):

    url = 'https://www.pro-football-reference.com/years/2019/week_'+str(i+1)+'.htm'

    r=get(url, headers=headers)

    week_html = BeautifulSoup(r.text, 'html.parser')

    table = week_html.select_one('div.game_summaries')

    

    

    

    for game in table.find_all("div"):

        date = game.select_one('table.teams').select_one('tr.date').text

        away_team = game.select_one('table.teams').select('tr')[1].select_one('a').text

        away_score = game.select_one('table.teams').select('tr')[1].select_one('td.right').text

        home_team = game.select_one('table.teams').select('tr')[2].select_one('a').text

        home_score = game.select_one('table.teams').select('tr')[2].select_one('td.right').text

        

        games.setdefault('date',[]).append(date)

        games.setdefault('away_team',[]).append(away_team)

        games.setdefault('away_score',[]).append(away_score)

        games.setdefault('home_team',[]).append(home_team)

        games.setdefault('home_score',[]).append(home_score)
df = pd.DataFrame.from_dict(games)



df = df.astype({'home_score': 'int32', 'away_score':'int32'})



df["score_diff"] = df.home_score - df.away_score



df.drop(["date"], axis=1, inplace=True)
df.columns
df.head()
X = df.drop(['away_score',"home_score"], axis=1)

y = df[['away_score',"home_score","score_diff"]]



# Break off validation set from training data

#X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=1, test_size=0, random_state=0)



# Utilizing all data instead of typically

# splitting data between train and validate

X_train_full = X

y_train = y



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype == "object"]



# Select numeric columns

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

#X_valid = X_valid_full[my_cols].copy()
# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

model = RandomForestRegressor(n_estimators=100, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

rfr = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



# Preprocessing of training data, fit model 

rfr.fit(X_train, y_train)
# Preprocessing of validation data, get predictions

preds = rfr.predict(pd.DataFrame({

    "away_team":["Cleveland Browns"],

    "home_team":["Cincinnati Bengals"]

}))



pd.DataFrame({

    "away_score":[preds[0][0]],

    "home_score":[preds[0][1]],

    "score_diff":[preds[0][2]]

             })
def predict(week, model):

    '''

    Predicts the outcomes of all games based on given week

    '''

    

    games = {}

    url = 'https://www.pro-football-reference.com/years/2020/week_'+str(week)+'.htm'

    r=get(url, headers=headers)

    week_html = BeautifulSoup(r.text, 'html.parser')

    table = week_html.select_one('div.game_summaries')

    

    

    

    for game in table.find_all("div"):

        away_team = game.select_one('table.teams').select('tr')[1].select_one('a').text

        home_team = game.select_one('table.teams').select('tr')[2].select_one('a').text

        

        games.setdefault('away_team',[]).append(away_team)

        games.setdefault('home_team',[]).append(home_team)

        

        preds = model.predict(pd.DataFrame({

            "away_team":[away_team],

            "home_team":[home_team]

        }))

        print("Away: {} \nHome: {}".format(away_team, home_team))

        outcome = pd.DataFrame({

                    "away_score":[preds[0][0]],

                    "home_score":[preds[0][1]],

                    "score_diff":[preds[0][2]]

                             })

        print(outcome)

        print()

        print()
predict(3, rfr)