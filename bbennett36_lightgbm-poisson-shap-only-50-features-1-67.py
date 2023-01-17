%matplotlib inline
from matplotlib import pyplot as plt
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, r2_score, median_absolute_error, mean_absolute_error, mean_squared_error, mean_squared_log_error
import seaborn as sns
import shap
shap.initjs()
dataset = pd.read_csv('../input/train.csv')
dataset = dataset.drop(['ID'], axis=1)
top100 = ['547d3135b', '30b3daec2', '899dbe405', 'd48c08bda', 'e78e3031b', '27461b158', 'bee629024', '9df4daa99', '236cc1ff5', 'a00adf70e', '8337d1adc', 'b6fa5a5fd', '29c059dd2', 'cd24eae8a', 'e7c0cfd0f', '7a7da3079', '3a48a2cd2', '191e21b5f', 'e176a204a', '91f701ba2', 'e13b0c0aa', '1af4d24fa', '22c7b00ef', 'fb49e4212', '1184df5c2', '20604ed8f', '8675bec0b', '2eeadde2b', 'cdfc2b069', '02861e414', '26fc93eb7', 'ed8951a75', '8e4d0fe45', '402bb0761', '939f628a7', '54723be01', '23310aa6f', '963a49cdc', '3cea34020', 'aa164b93b', '6786ea46d', '1931ccfdd', '22ed6dba3', '542f770e5', 'dbfa2b77f', '50e4f96cf', '186b87c05', '5c6487af1', 'ce3d7595b', 'ced6a7e91', '73687e512', '342e7eb03', '1db387535', '4da206d28', 'dd771cb8e', '0c9462c08', 'adb64ff71', 'fb387ea33', 'fc99f9426', '703885424', '0ff32eb98', '13bdd610a', '62e59a501', '45f6d00da', '324921c7b', '1c71183bb', 'dcc269cfe', '092271eb3', '29ab304b9', '491b9ee45', '9280f3d04', '0d51722ca', '5324862e4', '70feb1494', 'f1e0ada11', 'edc84139a', '190db8488', '5a1589f1a', '22501b58e', 'c928b4b74', '66ace2992', '1702b5bf0', '83c3779bf', 'c47340d97', '87ffda550', '58232a6fb', 'c5a231d81', '6cf7866c1', '58e2e02e6', '024c577b9', '20aa07010', 'd6bb78916', '15ace8c9f', '9fd594eec', 'fb0f5dbfe', '58e056e12', 'eeb9cd3aa', '0572565c2', '6eef030c1', 'b43a7cfd5', 'target']
top50 = ['73687e512', '342e7eb03', '1db387535', '4da206d28', 'dd771cb8e', '0c9462c08', 'adb64ff71', 'fb387ea33', 'fc99f9426', '703885424', '0ff32eb98', '13bdd610a', '62e59a501', '45f6d00da', '324921c7b', '1c71183bb', 'dcc269cfe', '092271eb3', '29ab304b9', '491b9ee45', '9280f3d04', '0d51722ca', '5324862e4', '70feb1494', 'f1e0ada11', 'edc84139a', '190db8488', '5a1589f1a', '22501b58e', 'c928b4b74', '66ace2992', '1702b5bf0', '83c3779bf', 'c47340d97', '87ffda550', '58232a6fb', 'c5a231d81', '6cf7866c1', '58e2e02e6', '024c577b9', '20aa07010', 'd6bb78916', '15ace8c9f', '9fd594eec', 'fb0f5dbfe', '58e056e12', 'eeb9cd3aa', '0572565c2', '6eef030c1', 'b43a7cfd5', 'target']
dataset = dataset[top50]
def apply_log(value):
    return np.log2(value)
# dataset['target_log'] = dataset['target'].apply(apply_log)
# dataset = dataset.drop(['target'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
     dataset[[c for c in dataset if 'target' != c]], dataset['target'], test_size=0.33, random_state=42)
model_cols = list(dataset[[c for c in dataset if 'target' != c]])
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
params = {'objective':'poisson',
#          'boosting_type': 'goss',
#          'top_rate': 0.3,
#          'other_rate': 0.3,
         'num_leaves':60, 
         'learning_rate': 0.01,
         'feature_fraction': 0.5,
         'bagging_fraction': 0.9,
         'bagging_freq': 1,
         'bagging_seed': 1,
         'poisson_max_delta_step': 0.8,
         'min_data': 5,
         'metric': ['rmse'],
#          'min_gain_to_split': 100,
         'num_threads': 4,
         'max_bin': 63
         }
lgb_model = lgb.train(params, lgb_train, 10000, valid_sets=[lgb_test], early_stopping_rounds=250, verbose_eval=50)
probs = lgb_model.predict(X_test, num_iteration=-1)
explained_variance_score(y_test, probs)
r2_score(y_test, probs)  
f, ax = plt.subplots(figsize=(16, 8))
sns.kdeplot(probs, shade=True, label='predicted');
sns.kdeplot(y_test, shade=True, label='y_test');
sns.kdeplot(y_train, shade=True, label='y_train');
sns.kdeplot(dataset['target'], shade=True, label='entire mds dataset');
plt.show()
lgb.plot_importance(lgb_model, importance_type='gain', max_num_features=20)
lgb.plot_importance(lgb_model, importance_type='split', max_num_features=20)
shap_values = shap.TreeExplainer(lgb_model).shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type='violin')
global_shap_vals = np.abs(shap_values).mean(0)[:-1]
inds = np.argsort(global_shap_vals)
top20 = list(inds[len(inds)-20:len(inds)])
y_pos = np.arange(X_train.iloc[:, lambda X_train: top20].shape[1])
plt.barh(y_pos, global_shap_vals[top20], color="#1E88E5")
plt.yticks(y_pos, X_train.columns[top20])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlabel("mean SHAP value magnitude (change in log odds)")
plt.gcf().set_size_inches(12, 8)
plt.show()
def score_records(df, counter):
    
    results = pd.DataFrame()
    results['ID'] = df['ID']
    
    score_df = df[model_cols]
    
#     top50.remove('target')
#     score_df = df[top50]
#     score_df = score_df.drop(['target'], aixs=1)

    preds = lgb_model.predict(score_df, num_iteration=-1)
    results['target'] = preds
    
    if counter == 1:
        results.to_csv('062218-sub_v5.csv', header=True, index=False, sep=',')
#         print('Initial Scoriung File Created')
    else:
        results.to_csv('062218-sub_v5.csv', mode='a', header=False, index=False)
#         print('Scores Appended')
    return len(results.loc[results['target'] < 0]) < 1
counter = 1
batch_count = 0
for batch in pd.read_csv('../input/test.csv', chunksize=1000):
    good = score_records(batch, counter)
    if good:
        counter += 1
        batch_count += len(batch)
        if counter % 5 == 0:
            print(counter, ' ', batch_count)
    else:
        print('negative number')
        break
