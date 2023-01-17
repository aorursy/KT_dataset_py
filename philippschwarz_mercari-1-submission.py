%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.imports import *
from fastai.column_data import *
from fastai.structured import *

from scipy import stats

from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path('../input/')
def split_df(df, test_mask):
    df_train, df_test = df[~test_mask], df[test_mask]
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    return df_train, df_test

# na category names are just replaced with 'missing'
def split_cat(text):
    try:
        return text.split('/')
    except AttributeError:
        return tuple(['missing'] * 3)

# replace na or no description values with 'missing'
def fix_desc(text):
    return 'missing' if not isinstance(text, str) or text == 'No description yet' else text
split_cat('Men/Coats & Jackets/Flight/Bomber')
tuple(['missing'] * 3)
train = pd.read_csv(DATA_PATH/'train.tsv', sep='\t')
test = pd.read_csv(DATA_PATH/'test_stg2.tsv', sep='\t')
test.rename(columns={'test_id': 'train_id'}, inplace=True)
train.category_name.str.count('/').max()
train[train.category_name.str.count('/') == 3].category_name.unique()
train = train.drop(train[train['price'] < 3].index)
train['main_cat'], train['sub_cat1'], train['sub_cat2'] = zip(*train['category_name'].apply(split_cat))                                                              
test['main_cat'], test['sub_cat1'], test['sub_cat2'] = zip(*test['category_name'].apply(split_cat))

train.drop('category_name', inplace=True, axis=1)
test.drop('category_name', inplace=True, axis=1)

train['brand_name'].fillna(value='missing', inplace=True)
test['brand_name'].fillna(value='missing', inplace=True)
train['name'].fillna(value='missing', inplace=True)
test['name'].fillna(value='missing', inplace=True)
train['shipping'] = train['shipping'].astype('str')
test['shipping'] = test['shipping'].astype('str')

train['item_condition_id'] = train['item_condition_id'].astype('str')
test['item_condition_id'] = test['item_condition_id'].astype('str')
train['item_description'] = train['item_description'].apply(fix_desc)
test['item_description'] = test['item_description'].apply(fix_desc)
train['full_desc'] = train['name'].str.cat(train['item_description'], sep='\n')
test['full_desc'] = test['name'].str.cat(test['item_description'], sep='\n')
train.drop('name', axis=1, inplace=True)
train.drop('item_description', axis=1, inplace=True)

test.drop('name', axis=1, inplace=True)
test.drop('item_description', axis=1, inplace=True)
train['price'] = np.log1p(train['price'])
train.reset_index(inplace=True, drop=True)
train.columns
print(train['full_desc'][np.random.randint(0, len(train))])
dep = ['price']
rid = ['train_id']
struct_vars = ['item_condition_id', 'brand_name', 'shipping', 'main_cat', 'sub_cat1', 'sub_cat2']
test.columns
for s in struct_vars: print (len(train[s].unique()))
price = train[dep].as_matrix().flatten()
train = train[rid + struct_vars + dep]
test =  test[rid +  struct_vars]
test_mask = train.index.isin(get_cv_idxs(n = len(train), val_pct=0.1))
y_test = price[test_mask]
my_train, my_test = split_df(train, test_mask)
my_test.drop('price', axis=1, inplace=True)
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
        
def rmsle(y_pred, targ):
    '''Root Mean Squared Logarithmic Error'''
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(targ))**2))
        
def RMSE(preds, targs):
    assert(len(preds) == len(targs))
    return np.sqrt(mean_squared_error(targs, preds))    
def get_epochs(n_cycle, cycle_len, cycle_mult):
    n_epochs = 0
    for cycle in range(n_cycle):
        n_epochs += cycle_mult ** cycle
    
    return cycle_len * n_epochs
X_train = train.copy()
X_test = test.copy()
X_train.set_index('train_id', inplace=True)
X_test.set_index('train_id', inplace=True)
train_cats(X_train) 
apply_cats(X_test, X_train)
df_train, y_train, nas = proc_df(X_train, 'price')
df_test, _, nas = proc_df(X_test, na_dict=nas)
val_idxs = get_cv_idxs(len(df_train), val_pct=0.15, seed=None)
y_range = (0, np.max(y_train) * 1.5)
cat_vars = ['item_condition_id', 'brand_name', 'shipping', 'main_cat', 'sub_cat1', 'sub_cat2']

cat_sz = [(c, len(X_train[c].cat.categories)+1) for c in cat_vars]
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
print (emb_szs)
PATH = '../working/'
md = ColumnarModelData.from_data_frame(PATH,
                                       val_idxs, 
                                       df_train,
                                       y_train.astype(np.float32),
                                       cat_flds=cat_vars,
                                       bs=128, 
                                       test_df=df_test)
m = md.get_learner(emb_szs,
                   n_cont=0,
                   emb_drop=0.04,
                   out_sz=1,
                   szs=[1000, 500],
                   drops=[0.001, 0.01],
                   y_range=y_range)
# %%time
# m.lr_find()
# m.sched.plot(1000)
lr=1e-3
# bk = PlotDLTraining(m)
%%time
m.fit(lr, n_cycle=4, metrics=[RMSE], best_save_name='mercari_best')
x,y=m.predict_with_targs()
RMSE(x,y)
pred_test=m.predict(is_test=True)
submission = pd.DataFrame(np.exp(pred_test)).reset_index()
submission.columns = ['test_id', 'price']
submission.to_csv('submission.csv', index=False)
