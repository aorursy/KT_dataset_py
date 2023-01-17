import numpy as np

import pandas as pd

%matplotlib inline

import seaborn as sns

import gc



import warnings

warnings.filterwarnings('ignore')
base_path = '../input/ieee-minified-data/'
!ls ../input/
#train_data = transactions.merge(identity, on='TransactionID', how='left')

#test_data = transactions_test.merge(identity_test, on='TransactionID', how='left')
# data joined and minified using https://www.kaggle.com/kernels/svzip/20704078



train_data = pd.read_pickle(base_path + 'ieee_train_data.pkl')

test_data = pd.read_pickle(base_path + 'ieee_test_data.pkl')
def combine_string_cols(df, cols, inplace=False):

    def process_row(row):

        return '_'.join(map(lambda x: str(x).lower(), row.values))

    if not inplace:

        return df[cols].apply(process_row, axis=1)



def process_raw_data(data):

    data['transaction_dow'] = (np.floor((data['TransactionDT'] / (3600 * 24)) - 1) % 7).astype(int)

    data['transaction_hour'] = (np.floor((data['TransactionDT'] / 3600)) % 24).astype(int)

    data['transaction_dom'] = np.floor((data['TransactionDT'] / (3600 * 24) - 1) % 30).astype(int)

    ## card use count

    data.rename(columns={'addr2': 'billing_country', 'addr1': 'billing_region'}, inplace=True)

    country_counts = data['billing_country'].value_counts()

    def get_country_bin(country_code):

        if country_code == 87.0:

            return 'US'

        if country_code == 60.0:

            return 'Canada'

        if country_code == 96.0:

            return 'Mexico'

        if country_code == 'missing':

            return 'missing'

        if country_counts[int(country_code)] > 10:

            return 'other'

        else:

            return 'miniscule'



    data['billing_country'] = data['billing_country'].fillna('missing')

    data['billing_country_bin'] = data['billing_country'].apply(get_country_bin)

    data['billing_location'] = combine_string_cols(data, ['billing_country', 'billing_region'])

    

    #os_type

    data.rename(columns={'id_30': 'os_type'}, inplace=True)

    

    data.loc[data['os_type'].str.contains('Mac', na=False), 'os_type_bin'] = 'mac'

    data.loc[data['os_type'].str.contains('iOS', na=False), 'os_type_bin'] = 'iOS'

    data.loc[data['os_type'].str.contains('Android', na=False), 'os_type_bin'] = 'android'

    data.loc[data['os_type'].str.contains('Windows', na=False), 'os_type_bin'] = 'Windows'

    data.loc[data['os_type'].str.contains('Linux', na=False), 'os_type_bin'] = 'Linux'

    

    #device name

    data['device_name'] = data['DeviceInfo'].str.split('/', expand=True)[0]



    data.loc[data['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

    data.loc[data['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

    data.loc[data['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

    data.loc[data['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

    data.loc[data['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

    data.loc[data['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

    data.loc[data['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

    data.loc[data['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

    data.loc[data['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

    data.loc[data['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

    data.loc[data['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

    data.loc[data['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

    data.loc[data['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

    data.loc[data['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

    data.loc[data['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

    data.loc[data['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

    data.loc[data['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'



    data.loc[data.device_name.isin(data.device_name.value_counts()[data.device_name.value_counts() < 200].index), 'device_name'] = "Others"

    

    #browser

    data['browser'] = data['id_31'].str.replace('\d+', '')

        

    #https://www.kaggle.com/artgor/eda-and-models#Feature-engineering

    data['TransactionAmt_to_mean_card1'] = data['TransactionAmt'] / data.groupby(['card1'])['TransactionAmt'].transform('mean')

    data['TransactionAmt_to_mean_card4'] = data['TransactionAmt'] / data.groupby(['card4'])['TransactionAmt'].transform('mean')

    data['TransactionAmt_to_mean_card5'] = data['TransactionAmt'] / data.groupby(['card5'])['TransactionAmt'].transform('mean')

    data['TransactionAmt_to_mean_billing_country'] = data['TransactionAmt'] / data.groupby(['billing_country'])['TransactionAmt'].transform('mean')

    data['TransactionAmt_to_mean_id31'] = data['TransactionAmt'] / data.groupby(['id_31'])['TransactionAmt'].transform('mean')

    data['card1_card2'] = data['card1'].astype(str) + '_' + data['card2'].astype(str)

    data['billing_country_dist1'] = data['billing_country'].astype(str) + '_' + data['dist1'].astype(str)

    data['card1_billing_country'] = data['card1'].astype(str) + '_' + data['billing_country'].astype(str)

    data['card1_billing_region'] = data['card1'].astype(str) + '_' + data['billing_region'].astype(str)

    data['card2_billing_country'] = data['card2'].astype(str) + '_' + data['billing_country'].astype(str)

    data['card2_billing_region'] = data['card2'].astype(str) + '_' + data['billing_region'].astype(str)

    data['card4_billing_country'] = data['card4'].astype(str) + '_' + data['billing_country'].astype(str)

    data['card4_billing_region'] = data['card4'].astype(str) + '_' + data['billing_region'].astype(str)

    data['DeviceInfo_P_emaildomain'] = data['DeviceInfo'].astype(str) + '_' + data['P_emaildomain'].astype(str)

    data['P_emaildomain_billing_country'] = data['P_emaildomain'].astype(str) + '_' + data['billing_country'].astype(str)

    data['id01_billing_country'] = data['id_01'].astype(str) + '_' + data['billing_country'].astype(str)

    data['TransactionAmt_to_std_card1'] = data['TransactionAmt'] / data.groupby(['card1'])['TransactionAmt'].transform('std')

    data['TransactionAmt_to_std_card4'] = data['TransactionAmt'] / data.groupby(['card4'])['TransactionAmt'].transform('std')

    data['TransactionAmt_to_std_card5'] = data['TransactionAmt'] / data.groupby(['card5'])['TransactionAmt'].transform('std')

    data['TransactionAmt_to_std_billing_country'] = data['TransactionAmt'] / data.groupby(['billing_country'])['TransactionAmt'].transform('std')

    data['TransactionAmt_to_std_id31'] = data['TransactionAmt'] / data.groupby(['id_31'])['TransactionAmt'].transform('std')

    data['TransactionAmt_decimal'] = ((data['TransactionAmt'] - data['TransactionAmt'].astype(int)) * 1000).astype(int)

    

    data['id_02_to_mean_card1'] = data['id_02'] / data.groupby(['card1'])['id_02'].transform('mean')

    data['id_02_to_mean_card4'] = data['id_02'] / data.groupby(['card4'])['id_02'].transform('mean')

    data['id_02_to_std_card1'] = data['id_02'] / data.groupby(['card1'])['id_02'].transform('std')

    data['id_02_to_std_card4'] = data['id_02'] / data.groupby(['card4'])['id_02'].transform('std')



    data['D15_to_mean_card1'] = data['D15'] / data.groupby(['card1'])['D15'].transform('mean')

    data['D15_to_mean_card4'] = data['D15'] / data.groupby(['card4'])['D15'].transform('mean')

    data['D15_to_std_card1'] = data['D15'] / data.groupby(['card1'])['D15'].transform('std')

    data['D15_to_std_card4'] = data['D15'] / data.groupby(['card4'])['D15'].transform('std')



    data['D15_to_mean_addr1'] = data['D15'] / data.groupby(['billing_region'])['D15'].transform('mean')

    data['D15_to_mean_addr2'] = data['D15'] / data.groupby(['billing_country'])['D15'].transform('mean')

    data['D15_to_std_addr1'] = data['D15'] / data.groupby(['billing_region'])['D15'].transform('std')

    data['D15_to_std_addr2'] = data['D15'] / data.groupby(['billing_country'])['D15'].transform('std')

    

    data[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = data['P_emaildomain'].str.split('.', expand=True)

    data[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = data['R_emaildomain'].str.split('.', expand=True)

    

    return data
train = process_raw_data(train_data)

target = train_data['isFraud']

del train_data
train.drop('isFraud', axis=1, inplace=True)
test = process_raw_data(test_data)

del test_data
print(train.shape)

print(test.shape)

print(target.shape)
def get_null_columns(df, threshold=0.9):

    return df.columns[(df.isnull().sum() / df.shape[0]) > 0.9]



def get_big_top_value_columns(df, threshold=0.9):

    return [col for col in df.columns if 

                                df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
train_null_columns = get_null_columns(train)

test_null_columns = get_null_columns(test)

train_big_val_columns = get_big_top_value_columns(train)

test_big_val_columns = get_big_top_value_columns(test)
cols_to_drop = set(list(train_null_columns) + list(test_null_columns) + train_big_val_columns + test_big_val_columns)
print(f'number of dropped columns {len(cols_to_drop)}')
train.drop(cols_to_drop, axis=1, inplace=True)

test.drop(cols_to_drop, axis=1, inplace=True)
print(train.shape)

print(test.shape)
inferred_category_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',

'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',

'R_emaildomain', 'card1', 'card2', 'card3', 'card5', 'billing_region', 'billing_country', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9','P_emaildomain_1', 'P_emaildomain_2', 

'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3', 'os_type', 'billing_country_bin', 'billing_location',

'os_type_bin', 'device_name','browser','card1_card2','billing_country_dist1','card1_billing_country','card1_billing_region',

'card2_billing_country','card2_billing_region','card4_billing_country','card4_billing_region','DeviceInfo_P_emaildomain',

'P_emaildomain_billing_country','id01_billing_country']



def get_high_correlation_cols(df, corrThresh=0.9):

    numeric_cols = df._get_numeric_data().columns

    corr_matrix = df.loc[:, numeric_cols].corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    to_drop = [column for column in upper.columns if any(upper[column] > corrThresh)]

    return to_drop



def replace_na(data, numeric_replace=-1, categorical_replace='missing', cat_features=[]):

    numeric_cols = data._get_numeric_data().columns

    categorical_cols = list(set(list(set(data.columns) - set(numeric_cols)) + cat_features))

    categorical_cols = [col for col in categorical_cols if col in data.columns]

    if numeric_replace is not None:

        data[numeric_cols] = data[numeric_cols].fillna(numeric_replace)

    data[categorical_cols] = data[categorical_cols].fillna(categorical_replace)

    return data
train_corr_cols = get_high_correlation_cols(train.drop(['TransactionID', 'TransactionDT'], axis=1))

# test_corr_cols = get_high_correlation_cols(test.drop(['TransactionID', 'TransactionDT'], axis=1))
to_drop_high_corr = set(train_corr_cols)
train.drop(to_drop_high_corr, axis=1, inplace=True)

test.drop(to_drop_high_corr, axis=1, inplace=True)
print(train.shape)

print(test.shape)
train = replace_na(train, numeric_replace=None, cat_features=inferred_category_cols)

test = replace_na(test, numeric_replace=None, cat_features=inferred_category_cols)
from sklearn.preprocessing import LabelEncoder



class LabelEncoderExt(object):

    def __init__(self):

        self.label_encoder = LabelEncoder()



    def fit(self, data_list):

        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])

        self.classes_ = self.label_encoder.classes_

        return self



    def transform(self, data_list):

        unknown_index = ~(data_list.isin(self.classes_))

        data_list[unknown_index] = 'Unknown'

        return self.label_encoder.transform(data_list)



class DataFrameCategoryEncoder:

    def __init__(self):

        self.encoder_maps = {}

        

    def _get_cat_columns(self, df):

        numeric_columns = df._get_numeric_data().columns

        return [c for c in df.columns if c not in numeric_columns]

    

    def fit_transform(self, df, cat_columns=None):

        df = df.copy()

        if cat_columns is None:

            cat_columns = self._get_cat_columns(df)

            print(cat_columns)

        for col in cat_columns:

            _le = LabelEncoderExt().fit(df[col])

            df[col] = _le.transform(df[col])

            self.encoder_maps[col] = _le

        return df, cat_columns

            

    def transform(self, df, cat_columns=None):

        df = df.copy()

        if cat_columns is None:

            cat_columns = self._get_cat_columns(df)

        for col in cat_columns:

            _le = self.encoder_maps.get(col, None)

            if _le is None:

                raise ValueError(f'Column not encountered in training - {col}')

            else:

                df[col] = _le.transform(df[col])

        return df
_ce = DataFrameCategoryEncoder()

train_en, category_columns = _ce.fit_transform(train)

test_en = _ce.transform(test)
del _ce

gc.collect()
print(train_en.shape)

print(test_en.shape)
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train_en = reduce_mem_usage(train_en)

test_en = reduce_mem_usage(test_en)
from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, train_test_split
import os

import zipfile

from IPython.display import FileLink



def zipdir(path, ziph):

    # ziph is zipfile handle

    for root, dirs, files in os.walk(path):

        for file in files:

            ziph.write(os.path.join(root, file))



def eval_recorder(params, tep, teprob, trp, trprob, scores, fi, name, mode='local'):

    scores = pd.DataFrame(scores, columns=['score'])

    mean_score = round(scores['score'].mean(), 6)

    std_score = round(scores['score'].std(), 4)

    name_score = str(mean_score) + '_' + str(std_score)

    if mode == 'local':

        if not os.path.isdir('results/' + name + '_' + name_score):

            os.mkdir('results/' + name + '_' + name_score)

        base_path = 'results/' + name + '_' + name_score + '/'

    if mode == 'kaggle':

        os.chdir(r'/kaggle/working')

        base_path = '/kaggle/working/'

        if not os.path.isdir(base_path + '/' + name + '_' + name_score):

            os.mkdir(name + '_' + name_score)

        base_path = base_path + '/' + name + '_' + name_score + '/'

    pd.Series(params).to_csv(base_path + 'params.csv', index=False)

    pd.DataFrame(tep).to_csv(base_path + 'test_predictions.csv', index=False)

    pd.DataFrame(teprob).to_csv(base_path + 'test_probablity.csv', index=False)

    if trp is not None:

        pd.DataFrame({'predition': trp, 'probablity': trprob}).to_csv(base_path + 'train_results.csv')

    pd.DataFrame(fi).to_csv(base_path + 'feature_importances.csv', index=False)

    if mode == 'kaggle':

        zipf = zipfile.ZipFile(name + '_' + name_score + '.zip', 'w', zipfile.ZIP_DEFLATED)

        zipdir(name + '_' + name_score +'/', zipf)

        zipf.close()

        return name + '_' + name_score + '.zip'

    

def split_eval(train, labels, x_val, y_val, test, clf, params, fit_params, name):

    scores = []

    feature_importances = np.zeros(len(train.columns))

    test_predictions = np.zeros(test.shape[0])

    test_probablity = np.zeros(test.shape[0])

    

    clf.fit(train, labels, eval_set=[(x_val, y_val)], **fit_params)

    if 'catboost' in name:

        scores.append(clf.best_score_['validation']['AUC'])

    if 'xgboost' in name:

        try:

            scores.append(clf.best_score)

        except:

            scores.append({'valid_0': {'auc': clf.evals_result()['validation_0']['auc'][-1]}})

    if 'lightgbm' in name:

        scores.append(clf.best_score_)

    test_predicts = clf.predict_proba(test)

    test_predictions = test_predicts[:, 1]

    test_probablity = test_predicts[:, 0]

    feature_importances = clf.feature_importances_

    print('-'*60)

    if 'lightgbm' in name:

        scores = [dict(s)['valid_0']['auc'] for s in scores]

    del clf

    filename = eval_recorder(params, test_predictions, test_probablity, None, None, scores, feature_importances, name, 'kaggle')

    return test_predictions, test_probablity, None, None, scores, feature_importances, filename



def plot_feature_importances(fe, cols):

    fe = pd.DataFrame(fe, index=cols)

    if fe.shape[1] > 1:

        fe = fe.apply(sum, axis=1)

    else:

        fe = fe[0]

    fe.sort_values(ascending=False)[:20].plot(kind='bar')



def cv_eval(train, labels, test, clf, cv, params, fit_params, name):

    scores = []

    feature_importances = np.zeros((len(train.columns), cv.n_splits))

    train_predictions = np.zeros(train.shape[0])

    train_probablity = np.zeros(train.shape[0])

    test_predictions = np.zeros((test.shape[0], cv.n_splits))

    test_probablity = np.zeros((test.shape[0], cv.n_splits))

    for i, (train_index, val_index) in enumerate(cv.split(train, labels)):

        print(f'starting {i} split')

        x_train = train.iloc[train_index]

        y_train = labels[train_index]

        x_val = train.iloc[val_index]

        y_val = labels[val_index]

        clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], **fit_params)

        if 'catboost' in name:

            scores.append(clf.best_score_['validation']['AUC'])

        if 'xgboost' in name:

            try:

                scores.append(clf.best_score)

            except:

                scores.append({'valid_0': {'auc': clf.evals_result()['validation_0']['auc'][-1]}})

        if 'lightgbm' in name:

            scores.append(clf.best_score_)

        val_predictions = clf.predict_proba(x_val)

        train_predictions[val_index] = val_predictions[:, 1]

        train_probablity[val_index] = val_predictions[:, 0]

        test_predicts = clf.predict_proba(test)

        test_predictions[:, i] = test_predicts[:, 1]

        test_probablity[:, i] = test_predicts[:, 0]

        feature_importances[:, i] = clf.feature_importances_

        print('-'*60)

        del clf

    filename = eval_recorder(params, test_predictions, test_probablity, train_predictions, train_probablity, scores, feature_importances, name, 'kaggle')

    return test_predictions, test_probablity, train_predictions, train_probablity, scores, feature_importances, filename



def eval_catboost(train, labels, test, cv, params, cat_features, name, eval_set=None):

    clf = CatBoostClassifier(**params)

    fit_params = {

        'cat_features': cat_features,

        'plot':False

    }

    if cv is not None:

        return cv_eval(train, labels, test, clf, cv, params, fit_params, 'catboost_' + name)

    return split_eval(train, labels, eval_set[0], eval_set[1], test, clf, params, fit_params, 'catboost_' + name)



def eval_xgboost(train, labels, test, cv, params, name, eval_set=None):

    clf = XGBClassifier(**params)

    fit_params = {

        'verbose':100, 

        'eval_metric':'auc',

        'early_stopping_rounds': 300

    }

    if cv is not None:

        return cv_eval(train, labels, test, clf, cv, params, fit_params, 'xgboost_' + name)

    return split_eval(train, labels, eval_set[0], eval_set[1], test, clf, params, fit_params, 'xgboost_' + name)



def eval_lightgbm(train, labels, test, cv, params, cat_features, name, eval_set=None):

    clf = LGBMClassifier(**params)

    fit_params = {

        'verbose': 100,

        'eval_metric': 'auc',

        #'categorical_feature':cat_features,        

        'early_stopping_rounds': 300

    }

    if cv is not None:

        return cv_eval(train, labels, test, clf, cv, params, fit_params, 'lightgbm_' + name)

    return split_eval(train, labels, eval_set[0], eval_set[1], test, clf, params, fit_params, 'lightgbm_' + name)
catboost_params = {

    'iterations': 5000,

    'loss_function': 'Logloss',

    'task_type': 'GPU',

    'eval_metric': 'AUC',

    'random_seed': 42,

    'od_type': 'Iter',

    'early_stopping_rounds': 300,

    'learning_rate': 0.07,

    'depth': 8,

    'random_strength': 0.5,

    'verbose': 100,

    'metric_period': 50

}



xgb_params = {

    'n_estimators': 5000,

    'n_job': 6,

    'max_depth': 8,

    'learning_rate': 0.05,

    'colsample_bytree': 0.5,

    'tree_method': 'gpu_hist'

}



lgb_params = {

    'n_estimators': 5000,

    'learning_rate': 0.05,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'reg_lambda': 0.2

}
train = train.sort_values('TransactionDT')

test = test.sort_values('TransactionDT')

train_en = train_en.sort_values('TransactionDT')

test_en = test_en.sort_values('TransactionDT')

target = target.reindex(train.index)
train.drop(['TransactionDT', 'TransactionID'], axis=1, inplace=True)

test.drop(['TransactionDT', 'TransactionID'], axis=1, inplace=True)

train_en.drop(['TransactionDT', 'TransactionID'], axis=1, inplace=True)

test_en.drop(['TransactionDT', 'TransactionID'], axis=1, inplace=True)
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

train_x, val_x, train_y, val_y = train_test_split(train, target, shuffle=False)
test_predictions_cat, test_probablity_cal, train_predictions, train_probablity, scores, feature_importances, result_zip = eval_catboost(

    train_x,

    train_y,

    test,

    None,

    catboost_params,

    category_columns,

    'catboost_tts',

    (val_x, val_y)

)
plot_feature_importances(feature_importances, train.columns)
del test_probablity_cal, train_predictions, train_probablity, scores, feature_importances
gc.collect()
FileLink(result_zip)
del train

del test

del train_x

del train_y

del val_x

del val_y
gc.collect()
train_x_en, val_x_en, train_y_en, val_y_en = train_test_split(train_en, target, shuffle=False)
test_predictions_lgb, test_probablity_lgb, train_predictions, train_probablity, scores, feature_importances, result_zip = eval_lightgbm(

    train_x_en,

    train_y_en,

    test_en,

    None,

    lgb_params,

    category_columns,

    'lightgbm_tts',

    (val_x_en, val_y_en)

)
plot_feature_importances(feature_importances, train_x_en.columns)
del test_probablity_lgb, train_predictions, train_probablity, scores, feature_importances
gc.collect()
train_x_en = train_x_en.fillna(-1)

val_x_en = val_x_en.fillna(-1)

test_en = test_en.fillna(-1)
test_predictions_xgb, test_probablity_xgb, train_predictions, train_probablity, scores, feature_importances, result_zip = eval_xgboost(

    train_x_en,

    train_y_en,

    test_en,

    None,

    xgb_params,

    'xgboost_tts',

    (val_x_en, val_y_en)

)
plot_feature_importances(feature_importances, train_en.columns)
def get_submission_df(predictions):

    sample_submission = pd.read_csv('../input/ieee-fraud-detection/' +'sample_submission.csv')

    print(sample_submission.shape)

    sample_submission['isFraud'] = predictions

    return sample_submission



def ensemble_predictions(preds):

    ensemble_prediction = np.zeros(preds[0].shape)

    for pred in preds:

        ensemble_prediction += pred

    ensemble_prediction /= len(preds)

    return get_submission_df(ensemble_prediction)



def ensemble_predictions_files(files):

    ensemble_prediction = np.zeros(506691)

    for f in files:

        _s = pd.read_csv('./submissions/'+f)

        ensemble_prediction += _s['prediction']

    ensemble_prediction /= len(files)

    return get_submission_df(ensemble_prediction)



def ensemble_predictions_files_gmean(files):

    ensemble_prediction = np.ones(506691)

    for f in files:

        _s = pd.read_csv('./submissions/'+f)

        ensemble_prediction *= _s['prediction']

    ensemble_prediction = np.power(ensemble_prediction, 1/len(files))

    return get_submission_df(ensemble_prediction)
ensemble_predictions([test_predictions_cat, test_predictions_lgb, test_predictions_xgb]).to_csv('submission.csv', index=False)