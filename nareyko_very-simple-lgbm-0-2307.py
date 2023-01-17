import numpy as np 
import pandas as pd 
import gc
import lightgbm as lgb
def readdf(filename, dataset, usecols=None, parse_dates=None):
    print(f'Reading {filename}...', end=' ')
    df = pd.read_csv(f'../input/{filename}', parse_dates=parse_dates, usecols=usecols)
    df['dataset'] = dataset
    gc.collect()
    print('Done. Rows:', df.shape[0])
    return df
usecols = ['category_name', 'city', 
       'description', 'image', 'image_top_1', 'item_id',
       'item_seq_number', 'parent_category_name', 'price', 'region', 'title', 
       'user_id', 'user_type']

train_df = readdf('train.csv', 0, usecols = usecols + ['deal_probability'])
test_df = readdf('test.csv', 1, usecols = usecols)
train_df['dataset'] = 0
test_df['dataset'] = 1

train_len = train_df.shape[0]
# validation set = 5% of train
val_len = train_len // 20
train_df['train_probability'] = train_df.deal_probability
train_df.loc[np.random.randint(0, train_len, train_len) < val_len, 'train_probability'] = np.nan
train_df.loc[pd.isnull(train_df.train_probability), 'dataset'] = -1
df = train_df.append(test_df, ignore_index=True)
del train_df, test_df
gc.collect()
df.loc[pd.isnull(df.image), 'image_top_1'] = -1
df.image_top_1 = df.image_top_1.astype('int')
df['has_image'] = pd.isnull(df.image).astype('int')
df['desc_len'] = df.description.str.len()
df['title_len'] = df.title.str.len()

df['desc_words'] = df.description.str.split().str.len()
df['title_words'] = df.description.str.split().str.len()

df['desc_word_avg'] = (df.desc_len / df.desc_words)
df['title_word_avg'] = (df.title_len / df.title_words)
def add_column(agg_name, column):
    global df, predictors
    df[agg_name] = column
    predictors.append(agg_name)
predictors = ['image_top_1', 'item_seq_number', 'price', 'has_image', 
       'desc_len', 'title_len', 'desc_words', 'title_words', 'desc_word_avg',
       'title_word_avg']
add_column('user_id_a_has_image', df.groupby('user_id')['has_image'].transform('mean'))
categorical = ['has_image', 'image_top_1']

for attr in ['city', 'category_name', 'user_type']:
    df[attr] = df[attr].astype('category').cat.codes
    predictors.append(attr)
    categorical.append(attr)
gc.collect()
test_df = df[df.dataset == 1]
len(test_df)
train_df = df[df.dataset == 0]
len(train_df)
val_df = df[df.dataset == -1]
len(val_df)
train_target = 'train_probability'
val_target = 'deal_probability'
xgtrain = lgb.Dataset(
    train_df[predictors].values,
    label=train_df[train_target].values,
    feature_name=predictors,
    categorical_feature=categorical,
)
xgvalid = lgb.Dataset(
    val_df[predictors].values,
    label=val_df[val_target].values,
    feature_name=predictors,
    categorical_feature=categorical,
    reference = xgtrain
)
gc.collect()
evals_results = {}

bst = lgb.train(
    {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 127,
        'learning_rate': 0.02,
        'verbose': 50,
        'nthread': 4,
        'seed': 1,
        'data_random_seed': 1
    },
    xgtrain,
    valid_sets=[xgvalid],
    valid_names=['valid'],
    evals_result=evals_results,
    num_boost_round=2000,
    early_stopping_rounds=100,
    verbose_eval=50,
    feval=None)

print("\nModel Report")
print("Best_iteration: ", bst.best_iteration)
print('rmse' + ":", evals_results['valid']['rmse'][bst.best_iteration - 1])

lgb.plot_importance(bst, figsize=(15,15))
print('Preparing DataFrame...')
sub = pd.DataFrame()
sub['item_id'] = test_df['item_id'].values
print('Predicting...')
sub['deal_probability'] = bst.predict(test_df[predictors].values, num_iteration=bst.best_iteration)
sub.deal_probability.clip(0, 1, inplace=True)
print('Writing...')
sub.to_csv('submission.csv', index=False)
print('done')