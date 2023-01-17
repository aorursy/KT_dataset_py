# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%load_ext autoreload
%autoreload 2

%matplotlib inline
import os
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.ensemble import forest
from sklearn.tree import export_graphviz
import IPython, graphviz, re
def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    """ Draws a representation of a random forest in IPython.

    Parameters:
    -----------
    t: The tree you wish to draw
    df: The data used to train the tree. This is used to get the names of the features.
    """
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}', s)))
def get_sample(df,n):
    """ Gets a random sample of n rows from df, without replacement.

    Parameters:
    -----------
    df: A pandas data frame, that you wish to sample from.
    n: The number of rows you wish to sample.

    Returns:
    --------
    return value: A random sample of n rows of df.

    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a

    >>> get_sample(df, 2)
       col1 col2
    1     2    b
    2     3    a
    """
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()
def train_cats(df):
    """Change any columns of strings in a panda's dataframe to a column of
    categorical values. This applies the changes inplace.

    Parameters:
    -----------
    df: A pandas dataframe. Any columns of strings will be changed to
        categorical values.

    Examples:
    ---------

    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a

    note the type of col2 is string

    >>> train_cats(df)
    >>> df

       col1 col2
    0     1    a
    1     2    b
    2     3    a

    now the type of col2 is category
    """
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()

def apply_cats(df, trn):
    """Changes any columns of strings in df into categorical variables using trn as
    a template for the category codes.

    Parameters:
    -----------
    df: A pandas dataframe. Any columns of strings will be changed to
        categorical values. The category codes are determined by trn.

    trn: A pandas dataframe. When creating a category for df, it looks up the
        what the category's code were in trn and makes those the category codes
        for df.

    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a

    note the type of col2 is string

    >>> train_cats(df)
    >>> df

       col1 col2
    0     1    a
    1     2    b
    2     3    a

    now the type of col2 is category {a : 1, b : 2}

    >>> df2 = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['b', 'a', 'a']})
    >>> apply_cats(df2, df)

           col1 col2
        0     1    b
        1     2    a
        2     3    a

    now the type of col is category {a : 1, b : 2}
    """
    for n,c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name=='category'):
            df[n] = c.astype('category').cat.as_ordered()
            df[n].cat.set_categories(trn[n].cat.categories, ordered=True, inplace=True)

def fix_missing(df, col, name, na_dict):
    """ Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing.

    Parameters:
    -----------
    df: The data frame that will be changed.

    col: The column of data to fix by filling in missing data.

    name: The name of the new filled column in df.

    na_dict: A dictionary of values to create na's of and the value to insert. If
        name is not a key of na_dict the median will fill any missing data. Also
        if name is not a key of na_dict and there is no missing data in col, then
        no {name}_na column is not created.


    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2

    >>> fix_missing(df, df['col1'], 'col1', {})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1     2    2    True
    2     3    2   False


    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2

    >>> fix_missing(df, df['col2'], 'col2', {})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2


    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2

    >>> fix_missing(df, df['col1'], 'col1', {'col1' : 500})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1   500    2    True
    2     3    2   False
    """
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

def numericalize(df, col, name, max_n_cat):
    """ Changes the column col from a categorical type to it's integer codes.

    Parameters:
    -----------
    df: A pandas dataframe. df[name] will be filled with the integer codes from
        col.

    col: The column you wish to change into the categories.
    name: The column name you wish to insert into df. This column will hold the
        integer codes.

    max_n_cat: If col has more categories than max_n_cat it will not change the
        it to its integer codes. If max_n_cat is None, then col will always be
        converted.

    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a

    note the type of col2 is string

    >>> train_cats(df)
    >>> df

       col1 col2
    0     1    a
    1     2    b
    2     3    a

    now the type of col2 is category { a : 1, b : 2}

    >>> numericalize(df, df['col2'], 'col3', None)

       col1 col2 col3
    0     1    a    1
    1     2    b    2
    2     3    a    1
    """
    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = col.cat.codes+1

def scale_vars(df, mapper):
    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper

def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    """ proc_df takes a data frame df and splits off the response variable, and
    changes the df into an entirely numeric dataframe. For each column of df 
    which is not in skip_flds nor in ignore_flds, na values are replaced by the
    median value of the column.

    Parameters:
    -----------
    df: The data frame you wish to process.

    y_fld: The name of the response variable

    skip_flds: A list of fields that dropped from df.

    ignore_flds: A list of fields that are ignored during processing.

    do_scale: Standardizes each column in df. Takes Boolean Values(True,False)

    na_dict: a dictionary of na columns to add. Na columns are also added if there
        are any missing values.

    preproc_fn: A function that gets applied to df.

    max_n_cat: The maximum number of categories to break into dummy values, instead
        of integer codes.

    subset: Takes a random subset of size subset from df.

    mapper: If do_scale is set as True, the mapper variable
        calculates the values used for scaling of variables during training time (mean and standard deviation).

    Returns:
    --------
    [x, y, nas, mapper(optional)]:

        x: x is the transformed version of df. x will not have the response variable
            and is entirely numeric.

        y: y is the response variable

        nas: returns a dictionary of which nas it created, and the associated median.

        mapper: A DataFrameMapper which stores the mean and standard deviation of the corresponding continuous
        variables which is then used for scaling of during test-time.

    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a

    note the type of col2 is string

    >>> train_cats(df)
    >>> df

       col1 col2
    0     1    a
    1     2    b
    2     3    a

    now the type of col2 is category { a : 1, b : 2}

    >>> x, y, nas = proc_df(df, 'col1')
    >>> x

       col2
    0     1
    1     2
    2     1

    >>> data = DataFrame(pet=["cat", "dog", "dog", "fish", "cat", "dog", "cat", "fish"],
                 children=[4., 6, 3, 3, 2, 3, 5, 4],
                 salary=[90, 24, 44, 27, 32, 59, 36, 27])

    >>> mapper = DataFrameMapper([(:pet, LabelBinarizer()),
                          ([:children], StandardScaler())])

    >>>round(fit_transform!(mapper, copy(data)), 2)

    8x4 Array{Float64,2}:
    1.0  0.0  0.0   0.21
    0.0  1.0  0.0   1.88
    0.0  1.0  0.0  -0.63
    0.0  0.0  1.0  -0.63
    1.0  0.0  0.0  -1.46
    0.0  1.0  0.0  -0.63
    1.0  0.0  0.0   1.04
    0.0  0.0  1.0   0.21
    """
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))

def get_nn_mappers(df, cat_vars, contin_vars):
    # Replace nulls with 0 for continuous, "" for categorical.
    for v in contin_vars: df[v] = df[v].fillna(df[v].max()+100,)
    for v in cat_vars: df[v].fillna('#NA#', inplace=True)

    # list of tuples, containing variable and instance of a transformer for that variable
    # for categoricals, use LabelEncoder to map to integers. For continuous, standardize
    cat_maps = [(o, LabelEncoder()) for o in cat_vars]
    contin_maps = [([o], StandardScaler()) for o in contin_vars]
    return DataFrameMapper(cat_maps).fit(df), DataFrameMapper(contin_maps).fit(df)
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
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
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                #el
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
PATH = "../input/"
%%time 
df_raw_train = pd.read_csv(f'{PATH}train_V2.csv', low_memory=False)
df_raw_train = reduce_mem_usage(df_raw_train)
df_raw_test = pd.read_csv(f'{PATH}test_V2.csv', low_memory = False)
df_raw_test = reduce_mem_usage(df_raw_test)
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
display_all(df_raw_train.describe(include='all').T)
null_cnt = df_raw_train.isnull().sum().sort_values()
print('null count:', null_cnt[null_cnt > 0])
df_raw_train.dropna(inplace=True)
df_raw_train = df_raw_train.sort_values(by = ['matchId'])
mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'
df_raw_train['matchType'] = df_raw_train['matchType'].apply(mapper)
train_cats(df_raw_train)
df, y, nas = proc_df(df_raw_train, 'winPlacePerc')
def split_vals(a,n): return a[:n].copy(), a[n:].copy()
n_valid = 1000000
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape
def rmse(x,y): return np.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train[features]), y_train), rmse(m.predict(X_valid[features]), y_valid),
                m.score(X_train[features], y_train), m.score(X_valid[features], y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
# Training on subset of the data initially to do analysis fast.
X_train = X_train[:1000000]
y_train = y_train[:1000000]
X_train.head()
# Training on simple model without removing matchId, Id and groupId. Just to visualise the decision tree
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
rmse(m.predict(X_train), y_train)
draw_tree(m.estimators_[0], X_train, precision=3)
def fillInf(df, val):
    numcols = df.select_dtypes(include='number').columns
    cols = numcols[numcols != 'winPlacePerc']
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    for c in cols: df[c].fillna(val, inplace=True)
features = list(X_train.columns)
features
features.remove('Id')
features.remove('groupId')
features.remove('matchId')
# Fitting using a basic Random Forest
m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train[features], y_train)
print_score(m)
fi = rf_feat_importance(m, X_train[features]); fi
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:30])
X_train['headshot_percentage'] = X_train['headshotKills']/X_train['kills']
X_valid['headshot_percentage'] = X_valid['headshotKills']/X_valid['kills']

X_train['totalDistance'] = X_train['rideDistance'] + X_train['walkDistance'] + X_train['swimDistance']
X_valid['totalDistance'] = X_valid['rideDistance'] + X_valid['walkDistance'] + X_valid['swimDistance']

X_train['health_items'] = X_train['heals'] + X_train['boosts']
X_train['killPlaceOverMaxPlace'] = X_train['killPlace'] / X_train['maxPlace']
X_train['killsOverWalkDistance'] = X_train['kills'] / X_train['walkDistance']

X_valid['health_items'] = X_valid['heals'] + X_train['boosts']
X_valid['killPlaceOverMaxPlace'] = X_valid['killPlace'] / X_train['maxPlace']
X_valid['killsOverWalkDistance'] = X_valid['kills'] / X_train['walkDistance']

X_train['killStreakRate'] = X_train['killStreaks']/X_train['kills']
X_valid['killStreakRate'] = X_valid['killStreaks']/X_valid['kills']

X_train['killMinute'] = X_train['kills'] / X_train['matchDuration']
X_valid['killMinute'] = X_valid['kills'] / X_valid['matchDuration']

X_train['damageDealtMinute'] = X_train['damageDealt'] / X_train['matchDuration']
X_valid['damageDealtMinute'] = X_valid['damageDealt'] / X_valid['matchDuration']

X_train['participateKills'] = X_train['kills'] + X_train['assists'] + X_train['DBNOs']
X_valid['participateKills'] = X_valid['kills'] + X_valid['assists'] + X_valid['DBNOs']

X_train['vehicleDestroysMinute'] = X_train['vehicleDestroys'] / X_train['matchDuration']
X_valid['vehicleDestroysMinute'] = X_valid['vehicleDestroys'] / X_valid['matchDuration']

X_train['killsMiter'] = X_train['roadKills'] / X_train['rideDistance']
X_valid['killsMiter'] = X_valid['roadKills'] / X_valid['rideDistance']

X_train['playersJoined'] = X_train.groupby('matchId')['matchId'].transform('count')

X_train['killsNorm'] = X_train['kills']*((100-X_train['playersJoined'])/100)
X_train['damageDealtNorm'] = X_train['damageDealt']*((100-X_train['playersJoined'])/100)

X_train['boostsPerWalkDistance'] = X_train['boosts']/(X_train['walkDistance']) 

X_train['healsPerWalkDistance'] = X_train['heals']/(X_train['walkDistance'])

X_train['healsAndBoosts'] = X_train['heals']+X_train['boosts']

X_train['healsAndBoostsPerWalkDistance'] = X_train['healsAndBoosts']/(X_train['walkDistance'])

X_valid['playersJoined'] = X_valid.groupby('matchId')['matchId'].transform('count')

X_valid['killsNorm'] = X_valid['kills']*((100-X_valid['playersJoined'])/100)
X_valid['damageDealtNorm'] = X_valid['damageDealt']*((100-X_valid['playersJoined'])/100)

X_valid['boostsPerWalkDistance'] = X_valid['boosts']/(X_train['walkDistance']) 

X_valid['healsPerWalkDistance'] = X_valid['heals']/(X_train['walkDistance'])

X_valid['healsAndBoosts'] = X_valid['heals'] + X_valid['boosts']

X_valid['healsAndBoostsPerWalkDistance'] = X_valid['healsAndBoosts']/(X_valid['walkDistance'])
train_group_agg = X_train.groupby(['matchId', 'groupId', 'matchType'])
val_group_agg =  X_valid.groupby(['matchId', 'groupId', 'matchType'])

train_match_agg = X_train.groupby(['matchId'])
val_match_agg = X_valid.groupby(['matchId'])

X_train.columns
features = list(X_train.columns)
features.remove('matchId')
features.remove('groupId')
features.remove('Id')
agg_col = features
agg_col.remove('matchType')
agg_col.remove('numGroups')
agg_col.remove('maxPlace')
agg_col
features
fillInf(X_train, 0)
fillInf(X_valid, 0)
for col in agg_col :
    X_train['percentage_match_' + col] = train_match_agg[col].rank(pct=True).values
    X_valid['percentage_match_' + col] = val_match_agg[col].rank(pct=True).values
X_train = X_train.merge(train_match_agg[agg_col].max().rename(columns=lambda s: 'match_max_' + s), on = 'matchId', how = 'left')
X_valid = X_valid.merge(val_match_agg[agg_col].max().rename(columns=lambda s: 'match_max_' + s), on = 'matchId', how = 'left')
X_train = X_train.merge(train_match_agg[agg_col].min().rename(columns=lambda s: 'match_min_' + s), on = 'matchId', how = 'left')
X_valid = X_valid.merge(val_match_agg[agg_col].min().rename(columns=lambda s: 'match_min_' + s), on = 'matchId', how = 'left')
X_train = X_train.merge(train_match_agg[agg_col].mean().rename(columns=lambda s: 'match_mean_' + s), on = 'matchId', how = 'left')
X_valid = X_valid.merge(val_match_agg[agg_col].mean().rename(columns=lambda s: 'match_mean_' + s), on = 'matchId', how = 'left')
display_all(X_train.head().T)
features = list(X_train.columns)
features.remove('matchId')
features.remove('groupId')
features.remove('Id')
features
# Chooses random set for each tree. Helpful in reducing overfitting as it increases variance per each treee 
set_rf_samples(100000)
m = RandomForestRegressor(n_jobs=-1, n_estimators=40, max_features=0.5,min_samples_leaf=3)
m.fit(X_train[features], y_train)
print_score(m)
fi = rf_feat_importance(m, X_train[features]); 
display_all(fi)
plot_fi(fi[:30])
to_keep = fi[fi.imp > 0.0005].cols
len(to_keep)
X_train = X_train[to_keep]
X_valid = X_valid[to_keep]
features = list(X_train.columns)
features
m = RandomForestRegressor(n_jobs=-1, n_estimators=40, max_features=0.5,min_samples_leaf=3)
m.fit(X_train[features], y_train)
print_score(m)
m = RandomForestRegressor(n_jobs=-1, n_estimators=40, max_features=0.5,min_samples_leaf=10)
m.fit(X_train[features], y_train)
print_score(m)
m = RandomForestRegressor(n_jobs=-1, n_estimators=40, max_features=0.5,min_samples_leaf=25)
m.fit(X_train[features], y_train)
print_score(m)
fi = rf_feat_importance(m, X_train[features]); 
display_all(fi)
plot_fi(fi[:15])
to_keep = fi[fi.imp > 0.0005].cols
len(to_keep)
X_train = X_train[to_keep]
X_valid = X_valid[to_keep]
features = X_train.columns
m = RandomForestRegressor(n_jobs=-1, n_estimators=40, max_features=0.6,min_samples_leaf=3)
m.fit(X_train[features], y_train)
print_score(m)
from scipy.cluster import hierarchy as hc
import scipy
from matplotlib import pyplot as plt
corr = np.round(scipy.stats.spearmanr(X_train[features]).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=X_train[features].columns, orientation='left', leaf_font_size=16)
plt.show()
def get_score(X_train, y_train, X_valid, y_valid) :
    m = RandomForestRegressor(n_jobs=-1, n_estimators=40, max_features=0.6,min_samples_leaf=3)
    m.fit(X_train, y_train)
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]    
    return res
for c in ('percentage_match_killPlaceOverMaxPlace', 'percentage_match_killPlace','percentage_match_health_items', 'percentage_match_killMinute'):
    print(c, get_score(X_train.drop(c, axis=1), y_train, X_valid.drop(c, axis = 1), y_valid))
X_train = X_train.drop(['percentage_match_killPlaceOverMaxPlace'], axis = 1)
X_valid = X_valid.drop(['percentage_match_killPlaceOverMaxPlace'], axis = 1)
features = list(X_train.columns)
m = RandomForestRegressor(n_jobs=-1, n_estimators=40, max_features=0.6,min_samples_leaf=3)
m.fit(X_train, y_train)
print_score(m)
fi = rf_feat_importance(m, X_train[features]); 
display_all(fi)
plot_fi(fi[:15])
