%load_ext autoreload

%autoreload 2



%matplotlib inline
from fastai.imports import *

from fastai.tabular import *



from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.ensemble import forest

from sklearn.tree import export_graphviz

import IPython, graphviz

from IPython.display import display



from sklearn import metrics
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
PATH = "../input/bluebook-for-bulldozers/train/"
!ls {PATH}
df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory=False, 

                     parse_dates=["saledate"])
## temp cell to make a copy of DataFrame

d = df_raw.copy()

e = df_raw[:]
df_raw = d.copy()
def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)
display_all(df_raw.tail().T)
display_all(df_raw.describe(include='all').T)
df_raw.SalePrice = np.log(df_raw.SalePrice)
m = RandomForestRegressor(n_jobs=-1)

# The following code is supposed to fail due to string values in the input data

m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)
add_datepart(df_raw, 'saledate')

df_raw.saleYear.head()
dep_var = 'SalePrice'



object_cols = df_raw.select_dtypes(include = 'object')

cat_names = list(object_cols.columns)



cont_names = list(set(df_raw.columns)-set(cat_names)-{dep_var})
# we can use following function instead of above code block if we want

dep_var = 'SalePrice'

cont_names, cat_names = cont_cat_split(df_raw, dep_var = 'SalePrice')
# convert string type columns to category type columns



tfm = Categorify(cat_names, cont_names)

tfm(df_raw)
df_raw.UsageBand.cat.categories
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
# Replace all categories with their codes in all categorical columns

# We added 1 to all codes as Categorify method will assign -1 for NaN (null values)

# By adding 1, null values are indicated by 0, and categories' codes start from 0



for col in cat_names:

    df_raw[col] = df_raw[col].cat.codes + 1
display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
# fill missing values of columns present in cont_names

# After we replace categories with their codes, there will be no missing values in categorical columns



tfm = FillMissing(cat_names, cont_names)

tfm(df_raw)
os.makedirs('tmp', exist_ok=True)

df_raw.to_feather('tmp/bulldozers-raw')
df_raw = pd.read_feather('tmp/bulldozers-raw')
# all columns are numerical 

# no missing values



df_raw.info()
# Now we will split dependant variable from df_raw



y = df_raw.SalePrice

df = df_raw.drop(['SalePrice'], axis=1)
m = RandomForestRegressor(n_jobs=-1)

m.fit(df, y)

m.score(df,y)
def split_vals(a,n): return a[:n].copy(), a[n:].copy()



n_valid = 12000  # same as Kaggle's test set size

n_trn = len(df_raw)-n_valid

raw_train, raw_valid = split_vals(df_raw, n_trn)

X_train, X_valid = split_vals(df, n_trn)

y_train, y_valid = split_vals(y, n_trn)



X_train.shape, y_train.shape, X_valid.shape
def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
m = RandomForestRegressor(n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
# randomly select 30000 rows from df and y



n = 30000

idxs = sorted(np.random.permutation(len(df))[:n])  # generate 30000 indices randomly



df_trn = df.iloc[idxs].copy()

y_trn = y.iloc[idxs].copy()
X_train, _ = split_vals(df_trn, 20000)

y_train, _ = split_vals(y_trn, 20000)
m = RandomForestRegressor(n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
draw_tree(m.estimators_[0], df_trn, precision=3)
m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
preds = np.stack([t.predict(X_valid) for t in m.estimators_])

preds[:,0], np.mean(preds[:,0]), y_valid[0]
preds.shape
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)]);
m = RandomForestRegressor(n_estimators=20, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=40, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=80, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
y_trn = df_raw.SalePrice

df_trn = df_raw.drop(['SalePrice'], axis=1)



X_train, X_valid = split_vals(df_trn, n_trn)

y_train, y_valid = split_vals(y_trn, n_trn)
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
set_rf_samples(20000)
m = RandomForestRegressor(n_jobs=-1, oob_score=True)

%time m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

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

m.fit(X_train, y_train)

print_score(m)
t=m.estimators_[0].tree_
dectree_max_depth(t)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)