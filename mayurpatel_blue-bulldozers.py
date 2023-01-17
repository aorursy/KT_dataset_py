%load_ext autoreload

%autoreload 2



%matplotlib inline
from fastai.tabular import *

from fastai.imports import *



from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display



from sklearn import metrics

from sklearn.ensemble import forest

from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
# additional packages

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype

import IPython, graphviz, sklearn_pandas, sklearn, warnings, pdb

from sklearn.tree import export_graphviz
def train_cats(df):

    for n,c in df.items():

        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()

            

def get_sample(df,n):

    idxs = sorted(np.random.permutation(len(df))[:n])

    return df.iloc[idxs].copy()



def fix_missing(df, col, name, na_dict):

    if is_numeric_dtype(col):

        if pd.isnull(col).sum() or (name in na_dict):

            df[name+'_na'] = pd.isnull(col)

            filler = na_dict[name] if name in na_dict else col.median()

            df[name] = col.fillna(filler)

            na_dict[name] = filler

    return na_dict



def numericalize(df, col, name, max_n_cat):

    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):

        df[name] = pd.Categorical(col).codes+1



def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,

            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):

    if not ignore_flds: ignore_flds=[]

    if not skip_flds: skip_flds=[]

    if subset: df = get_sample(df,subset)

    else: df = df.copy()

    ignored_flds = df.loc[:, ignore_flds]

    df.drop(ignore_flds, axis=1, inplace=True)

    if preproc_fn: preproc_fn(df)

    if y_fld is None: y = None

    else:

        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes

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



def draw_tree(t, df, size=10, ratio=0.6, precision=0):

    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,

                      special_characters=True, rotate=True, precision=precision)

    IPython.display.display(graphviz.Source(re.sub('Tree {',

       f'Tree {{ size={size}; ratio={ratio}', s)))

    

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
PATH = "../input"
!ls {PATH}
df_raw = pd.read_csv(f'{PATH}/train/Train.csv', low_memory=False, parse_dates=['saledate'])
df_raw.head().T
add_datepart(df_raw, 'saledate')
df_raw.SalePrice = np.log(df_raw.SalePrice)
df_raw.head()
train_cats(df_raw)
df_raw.Hydraulics.cat.codes
df_raw.head()
df_raw.describe(include='all').T
df_raw.UsageBand = df_raw.UsageBand.cat.codes
df_raw.isnull().sum().sort_index() / len(df_raw)
!mkdir tmp

df_raw.to_feather('tmp/test')
df, y, nas = proc_df(df_raw, 'SalePrice')
df.columns
df.shape, y.shape, df_raw.shape
m = RandomForestRegressor(n_jobs=-1)

m.fit(df, y)

m.score(df,y)
!wc -l {PATH}/Test.csv
def split_vals(a,n): return a[:n].copy(), a[n:].copy()



n_valid = 12000  # same as Kaggle's test set size

n_trn = len(df)-n_valid

raw_train, raw_valid = split_vals(df_raw, n_trn)

X_train, X_valid = split_vals(df, n_trn)

y_train, y_valid = split_vals(y, n_trn)



X_train.shape, y_train.shape, X_valid.shape
x = m.predict(X_train)
y = y_train
((x - y)**2).mean()
def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
m = RandomForestRegressor(n_jobs=-1, n_estimators=40)

%time m.fit(X_train, y_train)

print_score(m)
reset_rf_samples()
m = RandomForestRegressor(n_jobs=-1, n_estimators=10)

%time m.fit(X_train, y_train)

print_score(m)
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice', subset=30000, na_dict=nas)

X_train, _ = split_vals(df_trn, 20000)

y_train, _ = split_vals(y_trn, 20000)

df_raw.shape, X_train.shape, y_train.shape
m = RandomForestRegressor(n_jobs=-1, n_estimators=10, bootstrap=True)

%time m.fit(X_train, y_train)

print_score(m)
# trying single tree

m = RandomForestRegressor(n_jobs=-1, n_estimators=10, max_depth=4, bootstrap=False)

%time m.fit(X_train, y_train)

print_score(m)
draw_tree(m.estimators_[0], df_trn, precision=3)
m = RandomForestRegressor(n_jobs=-1, n_estimators=10, bootstrap=True)

%time m.fit(X_train, y_train)

print_score(m)
X_train.shape, y_train.shape
X_valid.shape, y_valid.shape
predictions = np.stack([t.predict(X_valid) for t in m.estimators_])

predictions[:, 0], np.mean(predictions[:, 0]), y_valid[0]
predictions.shape, y_valid.shape
predictions[:2]
plt.plot([metrics.r2_score(y_valid, np.mean(predictions[:i+1], axis=0)) for i in range(10)])
[predictions[:i+1] for i in range(10)][0].shape
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')

X_train, X_valid = split_vals(df_trn, n_trn)

y_train, y_valid = split_vals(y_trn, n_trn)
df_raw.shape, df_trn.shape, n_trn
set_rf_samples(20000)
m = RandomForestRegressor(n_jobs=-1, oob_score=False)

%time m.fit(X_train, y_train)

print_score(m)
reset_rf_samples()


def dectree_max_depth(tree):

    children_left = tree.children_left

    children_right = tree.children_right



    def walk(node_id):

        if (children_left[node_id] != children_right[node_id]):

            left_max = 1 + walk(children_left[node_id])

            right_max = 1 + walk(children_right[node_id])

            return max(left_max, right_max)

        else: # leaf

            return 1



    root_node_id = 0

    return walk(root_node_id)
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
t=m.estimators_[0].tree_
dectree_max_depth(t)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=5, n_jobs=-1, oob_score=True)

%time m.fit(X_train, y_train)

print_score(m)
t=m.estimators_[0].tree_

dectree_max_depth(t)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
t=m.estimators_[0].tree_

dectree_max_depth(t)
df_trn.shape
def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)
fi = rf_feat_importance(m, df_trn); fi[:10]
fi.plot('cols', 'imp', figsize=(10,6), legend=True);