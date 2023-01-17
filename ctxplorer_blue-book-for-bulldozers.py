%load_ext autoreload

%autoreload 2



%matplotlib inline
!pip install fastai==0.7.0

from fastai.imports import *

from fastai.structured import *



from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display



from sklearn import metrics
PATH = "../input/"

!ls {PATH}
df = pd.read_csv(f'{PATH}train/Train.csv', low_memory=False, parse_dates=['saledate'])
def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)
df.shape
display_all(df.head().T)
display_all(df.describe(include='all').T)
# df.SalePrice = np.log(df.SalePrice)
add_datepart(df, 'saledate')

df.columns
train_cats(df)

display_all(df.head().T)
df.UsageBand.cat.categories
df.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
df.UsageBand = df.UsageBand.cat.codes
display_all(df.head().T)
display_all(df.isnull().sum().sort_index()/len(df))
df, y, nas = proc_df(df, 'SalePrice')
os.makedirs('tmp', exist_ok=True)

df.to_feather('tmp/bulldozers-raw')
df_raw = pd.read_feather('tmp/bulldozers-raw')
rf = RandomForestRegressor(n_jobs=-1)

rf.fit(df, y)

rf.score(df, y)
def split_val(df, n):

    return df[:n].copy(), df[n:].copy()



n_val = 12000

n_trn = len(df) - n_val

x_trn, x_val = split_val(df, n_trn)

y_trn, y_val = split_val(y, n_trn)

x_trn.shape, x_val.shape, y_trn.shape, y_val.shape
def rmse(x, y):

    return math.sqrt(((np.log(x)-np.log(y))**2).mean())



def print_score(m):

    res = [rmse(m.predict(x_trn), y_trn), rmse(m.predict(x_val), y_val),

                m.score(x_trn, y_trn), m.score(x_val, y_val)]

    print(res)
rf = RandomForestRegressor(n_jobs=-1)

%time rf.fit(x_trn, y_trn)

print_score(rf)
test_df = pd.read_csv(f'{PATH}Test.csv', low_memory=False, parse_dates=['saledate'])
add_datepart(test_df, 'saledate')
train_cats(test_df)
test_df.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)

test_df.UsageBand = test_df.UsageBand.cat.codes
display_all(test_df.isnull().sum().sort_index()/len(test_df))
test_df, _, _ = proc_df(test_df, na_dict=nas)
output_df = pd.DataFrame({

    'SalesId': test_df.SalesID,

    'SalePrice': rf.predict(test_df)

})
output_df.tail()
output_df.to_csv('bulldozer_test_result_00.csv')