%load_ext autoreload
%autoreload 2 

%matplotlib inline
from fastai.structured import * 
from fastai.imports import *
from pandas_summary import DataFrameSummary
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics
PATH = '/kaggle/input/train/'
df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory=False, parse_dates=['saledate'])
def display_all(df):
    with pd.option_context('display.max_rows', 1000, 'display.max_columns', 1000):
        display(df)
display_all(df_raw.tail().T)
display_all(df_raw.describe(include='all').T)
df_raw.SalePrice = np.log(df_raw.SalePrice)
# Extracting date fields from complete date time for making categoricals
add_datepart(df_raw, 'saledate')
df_raw.saleYear.head()
# convert categorical variables(strings) to pandas categories
train_cats(df_raw)
df_raw.UsageBand.cat.categories
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
# converting text categories to numbers 
df_raw.UsageBand = df_raw.UsageBand.cat.codes
# summing null values
display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
# handling missing values, replacing categories with numeric codes
df, y, nas = proc_df(df_raw, 'SalePrice')
m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)
m.score(df, y)
def split_vals(a, n): return a[:n].copy(), a[n:].copy()

n_valid = 12000 # same as kaggle 
n_trn = len(df) - n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape
def rmse(x, y): return math.sqrt(((x - y)** 2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
          m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
m = RandomForestRegressor(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
prediction = m.predict(X_train)
kaggle = pd.DataFrame({'SalesID' : X_train.SalesID, 'SalePrice' : prediction})
kaggle.to_csv('/kaggle.submission.csv', encoding='utf-8', index=False)
