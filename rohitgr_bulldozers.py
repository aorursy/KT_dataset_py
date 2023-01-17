%load_ext autoreload
%autoreload 2

%matplotlib inline
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor, forest
import re
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

print(os.listdir('../input'))
# Learned these techniques from fast.ai ml course and implemented these functions myself with some changes.

class LabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, y):
        self.classes_ = ['NaN'] + list(set(y) - set(['NaN']))
        self.class_maps_ = {label: i for i, label in enumerate(self.classes_)}
        self.inverse_maps_ = {i: label for label, i in self.class_maps_.items()}

        return self

    def transform(self, y):
        y = np.array(y)
        new_labels = list(set(y) - set(self.classes_))
        y[np.isin(y, new_labels)] = 'NaN'

        return np.array([self.class_maps_[v] for v in y]).astype(np.int32)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        y = np.array(y)
        return np.array([self.inverse_maps_[v] for v in y])

    def add_label(self, label, ix):
        self.classes_ = self.classes_[:ix] + [label] + self.classes_[ix:]
        self.class_maps_ = {label: i for i, label in enumerate(self.classes_)}
        self.inverse_maps_ = {i: label for label, i in self.class_maps_.items()}


def replace_nan(df, nan_cols):
    df.replace(nan_cols, np.nan, inplace=True)


def conv_contncat(df, cont_cols=None, cat_cols=None):
    if cont_cols is not None:
        for n in cont_cols:
            df[n] = pd.to_numeric(df[n], errors='coerce').astype(np.float64)

    if cat_cols is not None:
        for n in cat_cols:
            df[n] = df[n].astype(str)

    df[df == 'nan'] = np.nan


def add_datecols(df, col, time=True, drop=True):
    fld_dtype = df[col].dtype

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True)

    target_pre = re.sub('[Dd]ate$', '', col)
    attr = ['year', 'month', 'week', 'dayofweek', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end', ]

    if time:
        attr += ['hour', 'minute', 'second']

    for at in attr:
        df[target_pre + at + '_cat'] = getattr(df[col].dt, at)

    df[target_pre + 'elapsed'] = np.int64(df[col])

    if drop:
        df.drop(col, axis=1, inplace=True)


def get_nn_mappers(df, cat_cols, cont_cols):
    cat_maps = [(o, LabelEncoder()) for o in cat_cols]
    cont_maps = [([o], StandardScaler()) for o in cont_cols]

    conv_mapper = DataFrameMapper(cont_maps).fit(df)
    cat_mapper = DataFrameMapper(cat_maps).fit(df)
    return cat_mapper, conv_mapper


def fix_missing(df, na_dict=None, cont_cols=None, cat_cols=None):
    if na_dict is not None:
        for n in na_dict.keys():
            col_null = df[n].isnull()
            df[n + '_na'] = col_null
            df.loc[col_null, n] = na_dict[n]

        if cont_cols is not None:
            for n in cont_cols:
                col_null = df[n].isnull()
                if col_null.sum():
                    df.loc[col_null, n] = df[n].median()

        if cat_cols is not None:
            for n in cat_cols:
                col_null = df[n].isnull()
                if col_null.sum():
                    df.loc[col_null, n] = 'NaN'

    else:
        na_dict = {}
        if cont_cols is not None:
            for n in cont_cols:
                col_null = df[n].isnull()
                if col_null.sum():
                    df[n + '_na'] = col_null
                    na_dict[n] = df[n].median()
                    df.loc[col_null, n] = na_dict[n]

        if cat_cols is not None:
            for n in cat_cols:
                col_null = df[n].isnull()
                if col_null.sum():
                    df[n + '_na'] = col_null
                    na_dict[n] = 'NaN'
                    df.loc[col_null, n] = na_dict[n]

    return na_dict


def get_one_hot(df, max_n_cat, drop_first, cat_mapper):
    one_hot_cols = []

    for n, c in df.items():
        if n.endswith('_cat') and len(set(df[n])) <= max_n_cat:
            one_hot_cols.append(n)

    for n, encoder, _ in cat_mapper.built_features:
        if len(encoder.classes_) <= max_n_cat:
            one_hot_cols.append(n)

    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=drop_first)
    return df


def proc_df(df, y_col, subset=None, drop_cols=None, do_scale=False, cat_mapper=None, cont_mapper=None, max_n_cat=None, drop_first=False):
    if subset is not None:
        df = df[-subset:]

    if drop_cols is not None:
        df.drop(drop_cols, axis=1, inplace=True)

    if do_scale:
        df[cont_mapper.transformed_names_] = cont_mapper.transform(df)

    df[cat_mapper.transformed_names_] = cat_mapper.transform(df)

    if max_n_cat is not None:
        df = get_one_hot(df, max_n_cat, drop_first, cat_mapper)

    X = df.drop(y_col, axis=1)
    y = df[y_col]

    return X, y


def rf_feat_importance(m, df):
    return pd.DataFrame({'cols': df.columns, 'imp': m.feature_importances_}).sort_values('imp', ascending=False)


def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples: forest.check_random_state(rs).randint(0, n_samples, n))


def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples: forest.check_random_state(rs).randint(0, n_samples, n_samples))
os.makedirs('tmp', exist_ok=True)

TRAIN_PATH = './data/Train.csv'
VALID_PATH = './data/Valid.csv'
VALID_SOL = './data/ValidSolution.csv'
TEST_PATH = './data/Test.csv'
def display_all(df):
    with pd.option_context('display.max_rows', 100, 'display.max_columns', 1000):
        display(df)
# train_df = pd.read_csv(TRAIN_PATH, low_memory=False, parse_dates=['saledate'])
# valid_df = pd.read_csv(VALID_PATH, low_memory=False, parse_dates=['saledate'])
# valid_sol = pd.read_csv(VALID_SOL, low_memory=False)
# test_sol = pd.read_csv(TEST_PATH, low_memory=False, parse_dates=['saledate'])
# train_df.to_feather('tmp/train_raw')
# valid_df.to_feather('tmp/valid_raw')
# valid_sol.to_feather('tmp/valid_sol_raw')
# test_sol.to_feather('tmp/test_raw')
# valid_df = pd.read_feather('tmp/valid_raw')
# valid_sol = pd.read_feather('tmp/valid_sol_raw')

# valid_df = pd.merge(left=valid_df, right=valid_sol, how='left', left_on='SalesID', right_on='SalesID')
# valid_df.drop('Usage', axis=1, inplace=True)

# valid_df.to_feather('tmp/valid_raw')
train_df = pd.read_feather('../input/bulldozerstmp/train_raw')
train_temp_df = train_df.copy()
valid_df = pd.read_feather('../input/bulldozerstmp/valid_raw')
valid_temp_df = valid_df.copy()

print(len(train_df))
print(len(valid_df))
%time add_datecols(train_df, col='saledate')
%time add_datecols(valid_df, col='saledate')
def get_n_unique(df):
    for col in df.columns:
        print(f'{col}: {len(df[col].unique())}')

# get_n_unique(train_df)
# Replace None and other null values with np.nan
nan_cols = [None, 'None or Unspecified', 'None', 'NaN', 'nan']
replace_nan(train_df, nan_cols)
replace_nan(valid_df, nan_cols)
drop_cols = None
cat_cols = ['Tire_Size', 'UsageBand', 'fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor', 'ProductSize', 'fiProductClassDesc', 'state', 'ProductGroup', 'ProductGroupDesc', 'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control', 'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension', 'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Hydraulics', 'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow', 'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb', 'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls', 'Differential_Type', 'Steering_Controls']
cont_cols = ['SalesID', 'MachineHoursCurrentMeter', 'auctioneerID', 'MachineID', 'ModelID', 'datasource', 'YearMade']
# Convert y to log values
train_df['SalePrice'] = np.log(train_df['SalePrice'])
valid_df['SalePrice'] = np.log(valid_df['SalePrice'])
# Convert categorical data to categoriy dtype and continous data to float32 dtype
conv_contncat(train_df, cont_cols, cat_cols)
conv_contncat(valid_df, cont_cols, cat_cols)
# Fill missing values
na_dict = fix_missing(train_df, None, cont_cols, cat_cols)
na_dict = fix_missing(valid_df, na_dict, cont_cols, cat_cols)
# cat_cols.extend([col for col in train_df.columns if col.endswith('_na')])
# Get mappers
cat_mapper, cont_mapper = get_nn_mappers(train_df, cat_cols, cont_cols)
# Proc_df
%time X_train, y_train = proc_df(train_df, 'SalePrice', drop_cols=drop_cols, cat_mapper=cat_mapper, cont_mapper=cont_mapper)
%time X_valid, y_valid = proc_df(valid_df, 'SalePrice', drop_cols=drop_cols, cat_mapper=cat_mapper, cont_mapper=cont_mapper)
X_train = X_train.astype(np.float32)
X_valid = X_valid.astype(np.float32)
def rmse(y_pred, y_true):
    return np.mean((y_pred - y_true)**2) ** 0.5

def get_scores(m, X_train, y_train, X_valid, y_valid):
    scores = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid), m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if m.oob_score is True:
        scores.append(m.oob_score_)

    return scores
m2 = RandomForestRegressor(n_estimators = 40, n_jobs=-1, oob_score=True)
%time m2.fit(X_train, y_train)
%time get_scores(m2, X_train, y_train, X_valid, y_valid)
m = RandomForestRegressor(n_estimators=80, n_jobs=-1, oob_score=True)
%time m.fit(X_train, y_train)
%time get_scores(m, X_train, y_train, X_valid, y_valid)
m3 = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
%time m3.fit(X_train, y_train)
%time get_scores(m3, X_train, y_train, X_valid, y_valid)
fi = rf_feat_importance(m3, X_train)
fi[:30].plot('cols', 'imp', 'barh', figsize=(12, 8), legend=False)
cols_to_keep = fi[fi['imp'] > 0.005]['cols']
len(cols_to_keep)
X_train_keep = X_train[cols_to_keep].copy()
X_valid_keep = X_valid[cols_to_keep].copy()
m4 = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
%time m4.fit(X_train_keep, y_train)
%time get_scores(m4, X_train_keep, y_train, X_valid_keep, y_valid)
import scipy
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(X_train_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=X_train_keep.columns, orientation='left', leaf_font_size=16)
plt.show()
dn_cols = ['saleyear_cat', 'saleelapsed']

for col in dn_cols:
    X_train_temp = X_train_keep.drop(col, axis=1).copy()
    X_valid_temp = X_valid_keep.drop(col, axis=1).copy()
    m = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, max_features=0.6, oob_score=True)
    m.fit(X_train_temp, y_train)
    print(col)
    print(get_scores(m, X_train_temp, y_train, X_valid_temp, y_valid))
X_train_ext = X_train_keep.copy()
X_valid_ext = X_valid_keep.copy()
X_train_ext['is_valid'] = 0
X_valid_ext['is_valid'] = 1
X_ext = pd.concat([X_train_ext, X_valid_ext], ignore_index=True)
X_ext = X_ext.sample(frac=1).reset_index(drop=True)

y_ext = X_ext['is_valid']
X_ext.drop('is_valid', axis=1, inplace=True)

X_ext.head()
m6 = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, max_features=0.6, oob_score=True)
%time m6.fit(X_ext, y_ext)
print(m6.oob_score_)
ft_ext = rf_feat_importance(m6, X_ext)
ft_ext[:10]
t_feats = ['saleyear_cat', 'saleelapsed', 'SalesID', 'saleweek_cat', 'MachineID']
X_ext.drop(t_feats, axis=1, inplace=True)
m6 = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, max_features=0.6, oob_score=True)
%time m6.fit(X_ext, y_ext)
print(m6.oob_score_)
m7 = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, max_features=0.6, oob_score=True)
%time m7.fit(X_train_keep, y_train)
get_scores(m7, X_train_keep, y_train, X_valid_keep, y_valid)
for col in t_feats:
    X_train_temp = X_train_keep.drop(col, axis=1)
    X_valid_temp = X_valid_keep.drop(col, axis=1)
    
    m0 = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, max_features=0.6, oob_score=True)
    m0.fit(X_train_temp, y_train)
    print(col)
    print(get_scores(m0, X_train_temp, y_train, X_valid_temp, y_valid))
X_train_temp = X_train_keep.drop(['SalesID', 'MachineID'], axis=1)
X_valid_temp = X_valid_keep.drop(['SalesID', 'MachineID'], axis=1)

m9 = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, max_features=0.6, oob_score=True)
%time m9.fit(X_train_temp, y_train)
%time get_scores(m9, X_train_temp, y_train, X_valid_temp, y_valid)
X_train_temp.columns
X_train_final = X_train_keep.drop(['MachineID', 'SalesID'], axis=1)
X_valid_final = X_valid_keep.drop(['MachineID', 'SalesID'], axis=1)

m9 = RandomForestRegressor(n_estimators=160, min_samples_leaf = 3, max_features=0.5, oob_score=True)
%time m9.fit(X_train_final, y_train)
%time get_scores(m9, X_train_final, y_train, X_valid_final, y_valid)