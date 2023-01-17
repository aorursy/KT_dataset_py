import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
sub=pd.read_csv("../input/sample_submission_V2.csv")

train.dropna(inplace=True)
def feature_engineering(df):
    # Sum of all distances
    df['allDistance'] = df['rideDistance']+df['swimDistance']+df['walkDistance']
    # Sum of all kills and assists
    df['allKills'] = df['headshotKills']+df['kills']+df['roadKills']+df['teamKills']+df['assists']
    # Special Kills
    df["specialKills"] = df["headshotKills"] + df["roadKills"]
    # % of head shot kills
    df['headshotKillRate'] = df['headshotKills'] / df['kills']
    # Kills per walk distance
    df['killsOverWalkDistance'] = df['kills'] / df['walkDistance']
    # kills per total distance
    df['killsOverDistance'] = df['kills'] / df['allDistance']
    # kill place over max place
    df['killPlaceOverMaxPlace'] = df['killPlace'] / df['maxPlace']
    # Sum of boosts and heals
    df['boosters'] = df['heals'] + df['boosts']
    # Kill Place Percentile
    df['killPlacePerc'] = df['kills'].rank(pct=True).values
    # Find Zombis
    df['zombi'] = ((df['allDistance'] == 0) & 
                   (df['kills'] == 0) & 
                   (df['weaponsAcquired'] == 0) &
                   (df['matchType'].str.contains('solo'))
                  ).astype(int)
    # Find Cheaters
    df['cheater'] = ((df['kills'] / df['allDistance'] >= 1) | 
                     (df['kills'] > 30) | (df['roadKills'] > 10)).astype(int)
    return df

train = feature_engineering(train)
test = feature_engineering(test)
def fillInf(df, val):
    numcols = df.select_dtypes(include='number').columns
    cols = numcols[numcols != 'winPlacePerc']
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    for c in cols:
        df[c].fillna(val, inplace=True)

fillInf(train,0)
fillInf(test,0)
# Thanks and credited to https://www.kaggle.com/gemartin who created this wonderful mem reducer
# iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.0f} bytes'.format(start_mem))
    
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
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.0f} bytes'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
columns = test.select_dtypes(include='number').columns

y_train = train[['winPlacePerc']]
x_train, x_valid, y_train, y_valid = train_test_split(train[columns], y_train, test_size = 0.2, random_state = 42)
x_test = test[columns]
def run_lgb(x_train, y_train, x_valid, y_valid, x_test):
    params = {
        "objective" : "regression",
        "metric" : "mae",
        "num_leaves" : 40,
        "learning_rate" : 0.004,
        "bagging_fraction" : 0.6,
        "feature_fraction" : 0.6,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42
    }
    
    lgb_train = lgb.Dataset(x_train, label=y_train)
    lgb_valid = lgb.Dataset(x_valid, label=y_valid)
    evals_result = {}
    model = lgb.train(params, 
                      lgb_train, 
                      1000, 
                      valid_sets=[lgb_train, lgb_valid], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result)
    
    pred_y_test = model.predict(x_test, num_iteration=model.best_iteration)
    return pred_y_test, model, evals_result

# Training LGB model
pred_y_test, model, evals_result = run_lgb(x_train, y_train, x_valid, y_valid, x_test)
print("LightGBM Training Completed...")
sub['winPlacePerc'] = pred_y_test
sub['winPlacePerc'] = sub['winPlacePerc'].apply(lambda x:1 if x>1 else x)
sub['winPlacePerc'] = sub['winPlacePerc'].apply(lambda x:0 if x<0 else x)
sub.to_csv('LGBM_submission.csv',index=False)
print('Plotting feature importances...')
ax = lgb.plot_importance(model, max_num_features=10)
plt.show()
dtg_model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=0.1
)
dtg_model.fit(x_train, y_train)
sub['winPlacePerc'] = dtg_model.predict(x_test)
sub.to_csv('DTR_submission.csv',index=False)
dtg_model.feature_importances_