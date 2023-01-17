!pip install -q lightgbm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

%matplotlib inline
plt.rcParams["figure.figsize"] = (15,6)
# Importing training, test data
train_raw = pd.read_csv('../input/train.csv')
valid_raw = pd.read_csv('../input/test.csv')
all_raw = train_raw.append(valid_raw, sort=True)
def transform(df, cat_features, num_features, target_feature):
    # Moving item_id to index for easier handling of ids
    df = df.set_index('item_id')
    df['activation_dayofweek'] = pd.to_datetime(df['activation_date']).dt.dayofweek
    # Transforming categorical variables to numerical ones
    le = LabelEncoder()
    for col in CAT_FEATURES:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
    # Saving item_ids for laters
    to_out = set(df.columns) - set(cat_features) - set(num_features) - set(target_feature)
    df = df.drop(to_out, axis=1) # dropping irrelevant features
    return df
ALL_FEATURES = ['activation_date', 'category_name', 'city', 'deal_probability',
                'description', 'image', 'image_top_1', 'item_id', 'item_seq_number',
                'param_1', 'param_2', 'param_3', 'parent_category_name', 'price',
                'region', 'title', 'user_id', 'user_type', 'activation_dayofweek']

TARGET_FEATURE = ['deal_probability']

NUM_FEATURES = ['price']

CAT_FEATURES = ['city', 'user_type', 'region', 'category_name', 'param_1', 'param_2', 'param_3', 'image_top_1', 'activation_dayofweek']
all = transform(all_raw, CAT_FEATURES, NUM_FEATURES, TARGET_FEATURE)
# Splitting training and validation set after transform
train, valid = all[all.deal_probability.notna()], all[all.deal_probability.isnull()]
# Creating testing set out of the training set
X_train, y_train = train.drop(['deal_probability'], axis=1), train['deal_probability']
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
# Dropping null probabilities for validation set
X_valid = valid.drop(['deal_probability'], axis=1)
# Generating codes for categorical features
CAT_CODES = sorted([X_train.columns.get_loc(col) for col in CAT_FEATURES])
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
                       
evals_result = {}  # to record eval results for plotting

params = {
    'num_leaves': 60,
    'bagging_fraction': 0.5,
    'bagging_freq': 1,
    'min_data_in_leaf': 1000,
    'learning_rate': 0.005,
    'num_boost_round': 100,
    'metric': ('l2'),
    'verbose': 0,
}
    
for learning_rate in [0.1, 0.01, 0.005]:
    for num_boost_round in [100, 500, 1000]:
        # Setting hyper-parameters
        params['learning_rate'] = learning_rate
        # Training gbm
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=1000,
                        valid_sets=[lgb_train, lgb_test],
                        categorical_feature=CAT_CODES,
                        evals_result=evals_result,
                        verbose_eval=0)
        '''
        ax = lgb.plot_metric(evals_result, metric='l2')
        plt.show()
        ax = lgb.plot_importance(gbm, max_num_features=10)
        plt.show()
        '''
        y_pred = gbm.predict(X_test, gbm.best_iteration)
        rmse = round(mean_squared_error(y_test, y_pred) ** 0.5, 5)
        train_test_diff = round(abs(gbm.best_score["training"]["l2"]-gbm.best_score["valid_1"]["l2"]), 5)
        print(f'learning_rate {learning_rate} num_boost_round {num_boost_round} RMSE {rmse} train_test_diff {train_test_diff}')
sorted(CAT_CODES)
X_train.head()
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
                       
evals_result = {}  # to record eval results for plotting

params = {
    'num_leaves': 60,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'min_data_in_leaf': 1000,
    'learning_rate': 0.01,
    'num_boost_round': 1000,
    'metric': ('l2'),
    'verbose': 0,
}

# Training gbm
gbm = lgb.train(params,
                lgb_train,
                valid_sets=[lgb_train, lgb_test],
                categorical_feature=CAT_CODES,
                evals_result=evals_result,
                verbose_eval=10)

ax = lgb.plot_metric(evals_result, metric='l2')
plt.show()
ax = lgb.plot_importance(gbm, max_num_features=20)
plt.show()

y_pred = gbm.predict(X_test, gbm.best_iteration)
rmse = round(mean_squared_error(y_test, y_pred) ** 0.5, 5)
print(f'RMSE {rmse}')
def plot(scores):
    scores.set_index('parameters')
    ax1 = scores.plot('parameters', 'training_error', 'line', xticks=scores.index)
    scores.plot('parameters', 'testing_error', 'line', ax=ax1);
    ax1.set_ylabel('Error')
    ax1.legend(shadow=True)
    ax2 = ax1.twinx()
    scores.plot('parameters', 'kaggle', 'line', title='Training/Testing/Kaggle scores', style=['--'], colormap='summer', ax=ax2);
    ax2.set_ylabel('Kaggle score')
    ax2.legend(loc='lower left', shadow=True);
scores = pd.DataFrame([
    {'parameters': 'num_leaves=5', 'lmse': 0.2381, 'training_error': 0.056674, 'testing_error': 0.056648, 'kaggle': 0.244},
    {'parameters': 'num_leaves=31', 'lmse': 0.2359, 'training_error': 0.054945, 'testing_error': 0.055673, 'kaggle': 0.2455},
    {'parameters': 'bag_freq', 'lmse': 0.2359, 'training_error': 0.05466, 'testing_error': 0.05607, 'kaggle': 0.2419},
    {'parameters': 'param_X', 'lmse': 0.2299, 'training_error': 0.05162, 'testing_error': 0.05285, 'kaggle': 0.2343},
    {'parameters': 'min_data=1k', 'lmse': 0.2296, 'training_error': 0.051316, 'testing_error': 0.052732, 'kaggle': 0.2343},
    {'parameters': 'rounds=500,lr=0.01', 'lmse': 0.22963, 'training_error': 0.0518084, 'testing_error': 0.0527287, 'kaggle': 0.2341},
    {'parameters': 'image_top_1', 'lmse': 0.22642, 'training_error': 0.0501103, 'testing_error': 0.0512681, 'kaggle': 0.2305},
    {'parameters': 'rounds=1000', 'lmse': 0.22538, 'training_error': 0.04856, 'testing_error': 0.0507949, 'kaggle': 0.2305}
])
plot(scores)