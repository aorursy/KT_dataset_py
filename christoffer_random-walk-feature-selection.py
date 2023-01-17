!pip install git+https://github.com/chrka/rwfs.git
import numpy as np

import pandas as pd



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline



np.random.seed(20191113)
def _rename_team(df, fro, to):

    df.loc[df['VisitorTeamAbbr'] == fro, 'VisitorTeamAbbr'] = to

    df.loc[df['HomeTeamAbbr'] == fro, 'HomeTeamAbbr'] = to



    

class TeamAbbrCleaner(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        df = X.copy()

        _rename_team(df, 'BAL', 'BLT')

        _rename_team(df, 'CLE', 'CLV')

        _rename_team(df, 'ARI', 'ARZ')

        _rename_team(df, 'HOU', 'HST')

        return df
class FeaturePreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        df = X.groupby('PlayId').first().reset_index().copy()

        

        # Rename unchanged variables

        df.rename(columns={

            'GameId': 'game_id',

            'PlayId': 'play_id',

            'Season': 'season',

            'Quarter': 'quarter',

            'Down': 'down',

            'Distance': 'distance',

            'Week': 'week',

            'OffenseFormation': 'offense_formation',

            'DefendersInTheBox': 'defenders_in_the_box'

        }, inplace=True)

        

        # Arrange features from the offense's direction

        team_on_offense = np.where(df['PossessionTeam'] == df['HomeTeamAbbr'],

                                  'home', 

                                   'away')

        df['offense_is_home'] = team_on_offense == 'home'



        df['offense_score'] = np.where(df['offense_is_home'],

                                      df['HomeScoreBeforePlay'],

                                      df['VisitorScoreBeforePlay'])



        df['defense_score'] = np.where(df['offense_is_home'],

                                       df['VisitorScoreBeforePlay'],

                                       df['HomeScoreBeforePlay'])

        

        # This works even at YardLine 50 when FieldPosition is NA

        df['line_of_scrimmage'] = np.where(df['FieldPosition'] == df['PossessionTeam'],

                                           df['YardLine'],

                                           100 - df['YardLine'])



        # Time between snap and handoff

        time_handoff = pd.to_datetime(df['TimeHandoff'])

        time_snap = pd.to_datetime(df['TimeSnap'])

        time_to_handoff = (time_handoff - time_snap).dt.total_seconds()

        time_to_handoff = np.round(time_to_handoff).astype(int)

        df['time_to_handoff'] = time_to_handoff

        

        # Convert game clock to seconds

        game_clock = df['GameClock'].str.extract(r'(?P<MM>\d\d):(?P<SS>\d\d):\d\d')

        df['game_clock'] = 60 * game_clock['MM'].astype(int) + game_clock['SS'].astype(int)

        

        return df[['game_id', 'play_id', 'season', 'quarter', 'down', 'distance', 'week',

                   'offense_formation', 'defenders_in_the_box', 

                   'offense_is_home', 'offense_score', 'defense_score', 

                   'line_of_scrimmage', 'time_to_handoff', 'game_clock']]
class TargetPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        return X.groupby('PlayId').first().reset_index()['Yards'].copy()
# ColumnTransformer does the same job, but loses column names.

# Adding them afterwords is quicker than this, but harder to maintain.

# A more convenient option is to use a name-preservering substitute or roll

# one's own.



def _hot_names(features, hot):

    names = []

    for feature, categories in zip(features, hot.categories_):

        names.extend([f"{feature}_{cat}" for cat in categories])

    return names



class PerPlayDatasetTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        # One-hot encode 'quarter' and 'down'

        self.quarter_down_one_hot = OneHotEncoder(handle_unknown='ignore', sparse=False)

        self.quarter_down_one_hot.fit(X[['quarter', 'down']])

        

        # Impute and one-hot encode 'offense_formation'

        self.encode_formation = Pipeline([

                ('imp', SimpleImputer(strategy='constant', fill_value='MISSING')),

                ('hot', OneHotEncoder(handle_unknown='ignore', sparse=False))

            ])

        self.encode_formation.fit(X[['offense_formation']])

        

        # One-hot encode 'time_to_handoff' for values of 1 and 2

        self.encode_tth = OneHotEncoder(categories=[[1, 2]], 

                                            handle_unknown='ignore',

                                            sparse=False)

        self.encode_tth.fit(X[['time_to_handoff']])



        # Min-max scale

        self.mm_scale = MinMaxScaler()

        self.mm_scale.fit(X[['distance', 'week', 'offense_score', 'defense_score', 

                             'line_of_scrimmage', 'game_clock']])



        self.scale_defenders_in_the_box = Pipeline([

            ('imp', SimpleImputer(strategy='most_frequent')),

            ('std', StandardScaler())

        ])

        self.scale_defenders_in_the_box.fit(X[['defenders_in_the_box']])

        

        return self

    

    def transform(self, X, y=None):

        # Passthrough values

        keep = X[['offense_is_home']]



        # One-hot encode 'quarter' and 'down'

        qd = self.quarter_down_one_hot.transform(X[['quarter', 'down']])

        qd = pd.DataFrame(qd, columns=_hot_names(['quarter', 'down'], self.quarter_down_one_hot))

        

        # Impute and one-hot encode 'offense_formation'

        fm = self.encode_formation.transform(X[['offense_formation']])

        fm = pd.DataFrame(fm, columns=_hot_names(['offense_formation'], 

                                                self.encode_formation.named_steps['hot']))

        

        # One-hot encode 'time_to_handoff' for values of 1 and 2

        tth = self.encode_tth.transform(X[['time_to_handoff']])

        tth = pd.DataFrame(tth, columns=_hot_names(['time_to_handoff'],

                                                  self.encode_tth))

        

        # Min-max scale

        mms = self.mm_scale.transform(X[['distance', 'week', 'offense_score', 'defense_score', 

                             'line_of_scrimmage', 'game_clock']])

        mms = pd.DataFrame(mms, columns=['distance', 'week', 'offense_score', 'defense_score', 

                           'line_of_scrimmage', 'game_clock'])

        

        ditb = self.scale_defenders_in_the_box.transform(X[['defenders_in_the_box']])

        ditb = pd.DataFrame(ditb, columns=['defenders_in_the_box'])

        

        return pd.concat([keep, qd, fm, tth, mms, ditb], axis=1)
dataset_pipeline = Pipeline([

    ('clean', TeamAbbrCleaner()),

    ('features', FeaturePreprocessor()),

    ('play', PerPlayDatasetTransformer())

])
raw_df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2020/train.csv", low_memory=False)
X_full = dataset_pipeline.fit_transform(raw_df)

feature_names = X_full.columns

X_full = X_full.values



y_full = TargetPreprocessor().fit_transform(raw_df)
from sklearn.linear_model import LinearRegression
def crps_exact(y_true, y_pred):

    """CRPS when y_true and y_pred are given as exact values."""

    return np.abs(y_true - y_pred) / 199
class LinearModel(BaseEstimator):

    def __init__(self, **lr_params):

        self.lr = None

        self.lr_params = lr_params

    

    def fit(self, X, y):

        self.lr = LinearRegression(**self.lr_params)

        self.lr.fit(X, y)

        return self

    

    def predict(self, X):

        return np.clip(np.round(self.lr.predict(X)), -99, 99)

    

    def score(self, X, y):

        y_pred = self.predict(X)

        # NB. Higher scores are better, but lower CRPS is better

        return -np.mean(crps_exact(y, y_pred))
from sklearn.model_selection import ShuffleSplit

from rwfs import RandomWalkFeatureSelection



cv_nfl = ShuffleSplit(n_splits=10, test_size=0.5, random_state=42)

model_nfl = LinearModel()

rwfs_nfl = RandomWalkFeatureSelection(model_nfl, cv_nfl, n_steps=1000, 

                                     initial_fraction=0.5,

                                     temperature=1e-4, cooldown_factor=0.99,

                                     agg=np.mean,

                                     cache_scores=False)
%%time

rwfs_nfl.fit(X_full, y_full, verbose=0)
import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'retina'

%matplotlib inline



import pandas as pd
diagnostics = pd.DataFrame(rwfs_nfl.diagnostics_)

plt.plot(-diagnostics['score'], 'g')

plt.plot(-diagnostics['score'] + diagnostics['se'], 'g--')

plt.plot(-diagnostics['score'] - diagnostics['se'], 'g--')

plt.plot(-diagnostics['mean_score'], 'r', alpha=0.5)

#plt.ylim(0.0, 1.0)

plt.title(f"Minimum CRPS found: {rwfs_nfl.best_score_:.5f}")

plt.show()
feature_names[list(rwfs_nfl.best_features_)]
importances = rwfs_nfl.feature_importances_

plt.bar(feature_names, importances)

plt.xticks(rotation='vertical')

plt.show()
feature_names[rwfs_nfl.feature_importances_ > 0.8]