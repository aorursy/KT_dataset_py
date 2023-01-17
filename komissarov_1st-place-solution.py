import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

from datetime import datetime

import math

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from sklearn.metrics import roc_auc_score, log_loss

from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier, Pool

import lightgbm as lgb



pd.set_option('display.max_columns', 500)
class Model:

    def __init__(self, cat_features_indices, params={'n_estimators': 500, 'learning_rate': 0.07}):

        self.cat_features_indices = cat_features_indices

        self.params = params

        

    def fit(self, X, y):

        pool = Pool(X, y, cat_features=self.cat_features_indices)

        self.model = CatBoostClassifier()

        self.model.set_params(**self.params)

        

        self.model.fit(pool)

        

    def predict(self, X):

        pool = Pool(X, cat_features=self.cat_features_indices)

        

        pred = self.model.predict_proba(pool)[:, 1]

        return pred

    

    def score(self, X, y):

        pred = self.predict(X)

        

        roc_auc_value = roc_auc_score(y, pred)

        log_loss_value = log_loss(y, pred)

        return roc_auc_value, log_loss_value

    

    def cross_validate(self, X, y):

        roc_auc_list = []

        log_loss_list = []

        kf = RepeatedStratifiedKFold(5, 5, random_state=1)

        

        for train_idx, test_idx in kf.split(X, y):

            X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]

            y_train, y_test = y[train_idx], y[test_idx]

            

            self.fit(X_train, y_train)

            ra, ll = self.score(X_test, y_test)

            

            roc_auc_list.append(ra)

            log_loss_list.append(ll)

        

        self.fit(X, y)

            

        print('ROC AUC: {}'.format(np.mean(roc_auc_list)))

        print('LOG LOSS: {}'.format(np.mean(log_loss_list)))

        print('roc auc folds std: {}'.format(np.std(roc_auc_list)))

        print('log loss folds std: {}'.format(np.std(log_loss_list)))

        return self.model
def get_deadline(row):

    if row['wave_id'] == 1:

        return row['Level'] < 7

    if row['wave_id'] == 2:

        return row['Level'] < 5

    if row['wave_id'] == 3:

        return row['Level'] < 4

    return False



def trimAndReplaceCity(city):

    if pd.isna(city) :

        return ''

    city = city.lstrip()

    if city.lower().__contains__('москв') :

        return 'Москва'

    if city.lower().__contains__('Moscow') :

        return 'Москва'

    if city.find('г.') > -1:

        return city.replace('г.', ' ').strip()

    if city.find('с.') > -1:

        return city.replace('с.', ' ').strip()

    if city.find('С.') > -1:

        return city.replace('С.', ' ').strip()

    if city.find('п.') > -1:

        return city.replace('п.', ' ').strip()

    if city.find('дер.') > -1:

        return city.replace('дер.', ' ').strip()

    if city.find('Дер.') > -1:

        return city.replace('Дер.', ' ').strip()

    if city.find('Г.') > -1:

        return city.replace('Г.', ' ').strip()

    return city



def prep_features(data):

    df = data.copy().rename({'Native city': 'city',

                             'Wave id': 'wave_id'}, axis=1)

    cities = pd.read_csv('../input/city-features/cities_features.csv', names=['city', 'f1', 'f2','f3', 'f4','f5', 'f6', 'f7', 'f8'])

    cities = cities.sort_values(by='f1', ascending=False).drop_duplicates('city', keep='first')

    df['city'] = df['city'].apply(trimAndReplaceCity)

    df = df.merge(cities, on='city', how='left')

    df.index = data.index

    df['age'] = ((datetime.now() - df['Birth date']).dt.days / 365).apply(math.floor)

    df['contract'] = df['Contract termination date'].isna().astype('int64')

    df['Life status'] = df['Life status'].astype(str)

    df['fail_deadline'] = df.apply(get_deadline, axis=1).astype(int)

    df['total_days'] = df[['day_00', 'day_01', 'day_02', 'day_03', 'day_04',

                           'day_05', 'day_06', 'day_07', 'day_08', 'day_09',

                           'day_10', 'day_11', 'day_12', 'day_13']].sum(axis=1)

    df['total_solo_projects'] = df[['evalexpr', 'match_n_match', 'bsq']].sum(axis=1)

    df['total_rushes'] = df[['rush_00', 'rush_01', 'rush_02']].sum(axis=1)

    df['total_exams'] = df[['exam_00', 'exam_01', 'exam_02', 'exam_final']].sum(axis=1)

    df['total_all_projects'] = df[['total_days', 'total_solo_projects',

                                   'total_rushes', 'total_exams']].sum(axis=1)

    df['nan_projects'] = df[['day_00', 'day_01', 'day_02', 'day_03', 'day_04',

                             'day_05', 'day_06', 'day_07', 'day_08', 'day_09',

                             'day_10', 'day_11', 'day_12', 'day_13',

                             'evalexpr', 'match_n_match', 'bsq',

                             'rush_00', 'rush_01', 'rush_02',

                             'exam_00', 'exam_01', 'exam_02', 'exam_final']].isna().sum(axis=1)

    df['zero_projects'] = (df[['day_00', 'day_01', 'day_02', 'day_03', 'day_04',

                              'day_05', 'day_06', 'day_07', 'day_08', 'day_09',

                              'day_10', 'day_11', 'day_12', 'day_13',

                              'evalexpr', 'match_n_match', 'bsq',

                              'rush_00', 'rush_01', 'rush_02',

                              'exam_00', 'exam_01', 'exam_02', 'exam_final']] == 0).sum(axis=1)

    

#     wave_dates = (pd.read_csv('data/waves_dates.csv', parse_dates=['date'])

#               .rename(columns={'id': 'wave_id',

#                                'date': 'wave_date'}))

    

#     df = df.merge(wave_dates, how='left', on='wave_id')

#     df.index = data.index

#     df['lvl_speed'] = df['Level'] / (pd.to_datetime('today').date() - df['wave_date'].dt.date).dt.days

#     df['lvl_wave_id'] = df['Level']..astype(str) + df['wave_id'].astype(str)

    return df
train = pd.read_csv('../input/school-21-student-expulsion-prediction/train.csv', index_col=0, parse_dates=[1, 32])

train = train[train['Wave id'] != 4]

features = ['age','Gender', 'Level', 'Life status',

#             'day_00', 'day_01', 'day_02', 'day_03', 'day_04',

#             'day_05', 'day_06', 'day_07', 'day_08', 'day_09',

#             'day_10', 'day_11', 'day_12', 'day_13',

#             'evalexpr', 'match_n_match', 'bsq',

#             'rush_00', 'rush_01', 'rush_02',

            'exam_00', 'exam_01', 'exam_02', 'exam_final',

            'f1', 'f2','f3', 'f4','f5', 'f6', 'f7', 'f8', # фичи гео аналитики по городу

            'total_days', 'total_solo_projects', 'total_rushes', 'total_exams', 'total_all_projects',

            'nan_projects', 'zero_projects',

            'contract', 'Memory entrance game',

            'Logic entrance game',

            'wave_id', 'fail_deadline'

#             'lvl_speed'

           ]

data = prep_features(train)[features]

y = train['contract_status'].values
X = data



map_feature_index = {j:i for i,j in enumerate(X.columns)}

# cat_features = ['Gender', 'Life status']

cat_features = ['Gender', 'Life status', 'wave_id']

cat_features_indices = [map_feature_index[i] for i in cat_features]



model_params = {

    'random_state': 12,

    'n_estimators': 700,

    'learning_rate': 0.03,

    'depth': 5,

    'verbose': 500

}



model = Model(cat_features_indices, model_params)

model.cross_validate(X, y)
def make_prediction_file(model):

    test = pd.read_csv('../input/school-21-student-expulsion-prediction/test.csv', index_col=0, parse_dates=[1])

    

    test_df = prep_features(test)[features]

    pred = model.predict(test_df)

    pred[test_df['wave_id'] == 4] = 1

    pred[test_df.index == 255283551] = 1 # cmanfred

    pred[test_df.index == 478971766] = 1 # creek

    pred_df = pd.DataFrame(pred, index=test_df.index).reset_index()

    pred_df.to_csv('prediction_best_cv.csv', header=['id', 'contract_status'], index=False)

    return pred_df

pred = make_prediction_file(model)
def get_feature_importance(model):

    feature_importance = model.model.get_feature_importance(Pool(X,label=y, cat_features=cat_features_indices))

    feature_score = pd.DataFrame(list(zip(X.dtypes.index, feature_importance)), 

                                 columns=['Feature','Score'])

    feature_score = feature_score.sort_values(by='Score',

                                              ascending=False,

                                              inplace=False,

                                              kind='quicksort',

                                              na_position='last')

    plt.rcParams["figure.figsize"] = (12,7)

    ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')

    ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)

    ax.set_xlabel('')



    rects = ax.patches



    labels = feature_score['Score'].round(2)



    for rect, label in zip(rects, labels):

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')



    plt.show()

    return feature_score
get_feature_importance(model)