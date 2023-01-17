print('harsh')
%load_ext autoreload
%autoreload 2

%matplotlib inline
import time
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns

from fastai.imports import *
from fastai.tabular import *
from fbprophet import Prophet

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold
from scipy import stats
from plotly.offline import init_notebook_mode, iplot
from plotly import graph_objs as go

import statsmodels.api as sm
# Initialize plotly
init_notebook_mode(connected=True)
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

pd.option_context("display.max_rows", 1000);
pd.option_context("display.max_columns", 1000);
os.getcwd()
PATH ='../input/demand-forecasting'
print(os.listdir(PATH))
train = pd.read_csv(f'{PATH}/train.csv',parse_dates=['week'])
test = pd.read_csv(f'{PATH}/test.csv',parse_dates=['week'])
sub = pd.read_csv(f'{PATH}/sample_submission.csv')
train.shape , test.shape
train.head()
test.head()
train.nunique()
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True,annot=True)
train.describe()
train.isnull().sum()
train.dropna(inplace=True)
train.isnull().sum()
def date_features(df):
    df['Year'] = pd.to_datetime(df['week']).dt.year
    df['Month'] = pd.to_datetime(df['week']).dt.month
    df['Day'] = pd.to_datetime(df['week']).dt.day
    df['Dayofweek'] = pd.to_datetime(df['week']).dt.dayofweek
    df['DayOfyear'] = pd.to_datetime(df['week']).dt.dayofyear
    df['Week'] = pd.to_datetime(df['week']).dt.week
    df['Quarter'] = pd.to_datetime(df['week']).dt.quarter 
    df['Is_month_start'] = pd.to_datetime(df['week']).dt.is_month_start
    df['Is_month_end'] = pd.to_datetime(df['week']).dt.is_month_end
    df['Is_quarter_start'] = pd.to_datetime(df['week']).dt.is_quarter_start
    df['Is_quarter_end'] = pd.to_datetime(df['week']).dt.is_quarter_end
    df['Is_year_start'] = pd.to_datetime(df['week']).dt.is_year_start
    df['Is_year_end'] = pd.to_datetime(df['week']).dt.is_year_end
    df['weekofyear'] = pd.to_datetime(df['week']).dt.weekofyear
    df['Semester'] = np.where(df['Quarter'].isin([1,2]),1,2)
    df['Is_weekend'] = np.where(df['Dayofweek'].isin([5,6]),1,0)
    df['Is_weekday'] = np.where(df['Dayofweek'].isin([0,1,2,3,4]),1,0)
    df['Days_in_month'] = pd.to_datetime(df['week']).dt.days_in_month
    return df
train = date_features(train)
test = date_features(test)
train.head()
def new_features(df):
    df['discount'] = df['base_price'] - df['total_price']
    df['discount%'] = (df['base_price'] - df['total_price'])/df['base_price']*100
    return df
train = new_features(train)
test = new_features(test)
pivoted = pd.pivot_table(train, values='units_sold',
                         columns='Year', index='Month')
pivoted.plot(figsize=(12,12));
pivoted = pd.pivot_table(train, values='units_sold',
                         columns='Year', index='Week')
pivoted.plot(figsize=(12,12));
pivoted = pd.pivot_table(train, values='units_sold',
                         columns='Month', index='Day')
pivoted.plot(figsize=(12,12));
temp_1 = train.groupby(['Year','Month'])['units_sold'].mean().reset_index()
plt.figure(figsize=(12,8));
sns.lmplot('Month','units_sold',data = temp_1, hue='Year', fit_reg= False);
temp_1 = train.groupby(['Year','Month','sku_id'])['units_sold'].mean().reset_index()
plt.figure(figsize=(30,10))
sns.swarmplot('sku_id', 'units_sold', data=temp_1, hue = 'Month');
# Place legend to the right
plt.legend(bbox_to_anchor=(1, 1), loc=2);
def inverse_boxcox(y, lambda_):
    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)

original_target = train.units_sold.values
target, lambda_prophet = stats.boxcox(train['units_sold'] + 1)
len_train=target.shape[0]
print('train values ',len_train)
merged_df = pd.concat([train, test])
%%time
merged_df["median-store_id_sku_id"] = merged_df.groupby(["sku_id", "store_id"])["units_sold"].transform("median")
merged_df["mean-store_id_sku_id"] = merged_df.groupby(["sku_id", "store_id"])["units_sold"].transform("mean")
merged_df["mean-Month_sku_id"] = merged_df.groupby(["Month", "sku_id"])["units_sold"].transform("mean")
merged_df["median-Month_sku_id"] = merged_df.groupby(["Month", "sku_id"])["units_sold"].transform("median")
merged_df["median-Month_store_id"] = merged_df.groupby(["Month", "store_id"])["units_sold"].transform("median")
merged_df["median-sku_id"] = merged_df.groupby(["sku_id"])["units_sold"].transform("median")
merged_df["median-store_id"] = merged_df.groupby(["store_id"])["units_sold"].transform("median")
merged_df["mean-sku_id"] = merged_df.groupby(["sku_id"])["units_sold"].transform("mean")
merged_df["mean-store_id"] = merged_df.groupby(["store_id"])["units_sold"].transform("mean")

merged_df["median-store_id_sku_id-Month"] = merged_df.groupby(['Month', "sku_id", "store_id"])["units_sold"].transform("median")
merged_df["mean-store_id_sku_id-week"] = merged_df.groupby(["sku_id", "store_id",'weekofyear'])["units_sold"].transform("mean")
merged_df["sku_id-Month-mean"] = merged_df.groupby(['Month', "sku_id"])["units_sold"].transform("mean")# mean units_sold of that sku_id  for all store_ids scaled
merged_df["store_id-Month-mean"] = merged_df.groupby(['Month', "store_id"])["units_sold"].transform("mean")# mean units_sold of that store_id  for all sku_ids scaled


lags = [12,30,90]
for i in lags:
#     print("Done For Lag {}".format(i))
    merged_df['_'.join(['sku_id-week_shifted-', str(i)])] = merged_df.groupby(['weekofyear',"sku_id"])["units_sold"].transform(lambda x:x.shift(i).sum()) 
    merged_df['_'.join(['sku_id-week_shifted-', str(i)])] = merged_df.groupby(['weekofyear',"sku_id"])["units_sold"].transform(lambda x:x.shift(i).mean()) 
    merged_df['_'.join(['sku_id-week_shifted-', str(i)])].fillna(merged_df['_'.join(['sku_id-week_shifted-', str(i)])].mode()[0], inplace=True)
    ##### units_sold for that sku_id i days in the past
    merged_df['_'.join(['store_id-week_shifted-', str(i)])] = merged_df.groupby(['weekofyear',"store_id"])["units_sold"].transform(lambda x:x.shift(i).sum())
    merged_df['_'.join(['store_id-week_shifted-', str(i)])] = merged_df.groupby(['weekofyear',"store_id"])["units_sold"].transform(lambda x:x.shift(i).mean()) 
    merged_df['_'.join(['store_id-week_shifted-', str(i)])].fillna(merged_df['_'.join(['store_id-week_shifted-', str(i)])].mode()[0], inplace=True)
train.drop('units_sold', axis=1, inplace=True)
merged_df.drop(['record_ID','week','units_sold'], axis=1, inplace=True)
merged_df.shape
merged_df.tail()
merged_df.columns
merged_df.dtypes
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in train:
    if train[col].dtype == 'bool':
        le.fit(train[col])
        # Transform both training and valing data
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        merged_df[col] = le.transform(merged_df[col])
            
            # Keep track of how many columns were label encoded
        le_count += 1
            
print('%d columns were label encoded.' % le_count)
'''for col in cols:
    dataset[col] = [1 if i == 'True' else 0 for sentiment in df[''].values]
    '''
m = merged_df *1
m.head(3)
merged_df.head(3)
params = {
    'nthread': 4,
    'categorical_feature' : [0,1,7,9,10,12,13,14,20], # Day, DayOfWeek, Month, Week, Item, Store, WeekOfYear
    'max_depth': 16,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': 'rmse', # this is abs(a-e)/max(1,a)
    'num_leaves': 150,
    'learning_rate': 0.15,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 30,
    'lambda_l1': 0.06,
    'lambda_l2': 0.1,
    'verbose': 1
}
num_folds = 3
test_x = merged_df[len_train:].values
all_x = merged_df[:len_train].values
all_y = target # removing what we did earlier

oof_preds = np.zeros([all_y.shape[0]])
sub_preds = np.zeros([test_x.shape[0]])

feature_importance_df = pd.DataFrame()
folds = KFold(n_splits=num_folds, shuffle=True, random_state=345665)

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(all_x)):
    
    train_x, train_y = all_x[train_idx], all_y[train_idx]
    valid_x, valid_y = all_x[valid_idx], all_y[valid_idx]
    lgb_train = lgb.Dataset(train_x,train_y)
    lgb_valid = lgb.Dataset(valid_x,valid_y)
        
    # train
    gbm = lgb.train(params, lgb_train, 1000, 
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=200, verbose_eval=200)
    
    oof_preds[valid_idx] = gbm.predict(valid_x, num_iteration=gbm.best_iteration)
    sub_preds[:] += gbm.predict(test_x, num_iteration=gbm.best_iteration) / folds.n_splits
    valid_idx += 1
    importance_df = pd.DataFrame()
    importance_df['feature'] = merged_df.columns
    importance_df['importance'] = gbm.feature_importance()
    importance_df['fold'] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, importance_df], axis=0)
    
e = 2 * abs(all_y - oof_preds) / ( abs(all_y)+abs(oof_preds) )
e = e.mean()
print('Full validation score With Box Cox %.4f' %e)
print('Inverting Box Cox Transformation')
print('Done!!')

sub_preds = inverse_boxcox(sub_preds , lambda_prophet) - 1
oof_preds = inverse_boxcox(oof_preds , lambda_prophet) - 1
e = 2 * abs(all_y - oof_preds) / ( abs(all_y)+abs(oof_preds) )
e = e.mean()
print('Full validation score Re-Box Cox Transformation is %.4f' %e)
#Don't Forget to apply inverse box-cox
sub_preds
sub['units_sold'] = sub_preds
sub.to_csv('sub3.csv',index=False)
feature_importance_df.head()
importance_df.sort_values(['importance'], 
                          ascending=False, inplace=True);
def plot_fi(fi): 
    return fi.plot('feature', 'importance', 'barh', figsize=(12,12), legend=False)
plot_fi(importance_df[:]);
merged_df.get_dtype_counts()
print("Before OHE", merged_df.shape)
merged_df = pd.get_dummies(merged_df, columns=['Day', 'Dayofweek', 'Month', 'Week', 'sku_id', 'weekofyear'])
print("After OHE", merged_df.shape)
test_x = merged_df[len_train:].values
all_x = merged_df[:len_train].values
all_y = target;
def XGB_regressor(train_X, train_y, test_X, 
    test_y= None, feature_names=None, seed_val=2018, num_rounds=500):

    param = {}
    param['objective'] = 'reg:linear'
    param['eta'] = 0.1
    param['max_depth'] = 5
    param['silent'] = 1
    param['eval_metric'] = 'rmse'
    param['min_child_weight'] = 4
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)
    
    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist,
                          early_stopping_rounds=20,verbose_eval=True)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds,verbose_eval=True)
        
    return model   
    
model = XGB_regressor(train_X = all_x,
                      train_y = all_y, test_X = test_x)
y_test = model.predict(xgb.DMatrix(test_x),
                       ntree_limit = model.best_ntree_limit)

y_test
sub['units_sold'] = y_test
sub.to_csv('xgboost.csv',index=False)
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


%matplotlib inline
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)


from numpy.random import seed
seed(42)
print('min date ',train['week'].min())
print('max date ',train['week'].max())

df_raw_cats = train[cat_cols].copy()
df_test_cats = test[cat_cols].copy()
df_raw_cats.head()
epochs = 40
batch = 256
lr = 0.0003
adam = optimizers.Adam(lr)
model_mlp = Sequential()
model_mlp.add(Dense(100, activation='relu', input_dim=train.shape[1]))
model_mlp.add(Dense(1))
model_mlp.compile(loss='mse', optimizer='adam')
model_mlp.summary()
original_target
target
mlp_history = model_mlp.fit(X_train.values, Y_train, validation_data=(df_raw_cats.values, ), epochs=epochs, verbose=2)
from xgboost import XGBRegressor,XGBClassifier
model = XGBRegressor(
    max_depth=8,
    n_estimators=100,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,
    verbose=3,
    seed=42)

model.fit(
    all_x, 
    all_y,verbose=1)
col=x_train.columns
from sklearn.preprocessing import StandardScaler
st=StandardScaler()
st.fit(x_train)
x_train=st.transform(x_train)
test=st.transform(test)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
%%time
GBoost = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5,verbose=1)
GBoost.fit(x_train,y_train)
pred=GBoost.predict(test)

len(test)
pred
type(pred)
t=np.expm1(pred)
t
sub['units_sold'] = t
sub.to_csv('sub1.csv',index=False)
