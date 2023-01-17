#Import needed libraries
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import textstat
from scipy import stats
%matplotlib inline
#Import data
print('Importing data...')
df_train = pd.read_csv("../input/train.csv")
#meta features
df_train['length'] = df_train['question_text'].str.len()

df_train['numbers'] = df_train['question_text'].apply(lambda x: len([s for s in x if s.isdigit()]))
df_train['words'] = df_train['question_text'].apply(lambda x: len([s for s in x if s.isalpha()]))
df_train['spaces'] = df_train['question_text'].apply(lambda x: len([s for s in x if s.isspace()]))
df_train['other_chars'] = df_train['length'] - df_train['numbers'] - df_train['words'] - df_train['spaces']

df_train['numbers_ratio'] = df_train['numbers'] / df_train['length']
df_train['words_ratio'] = df_train['words'] / df_train['length']
df_train['spaces_ratio'] = df_train['spaces'] / df_train['length']
df_train['other_chars_ratio'] = df_train['other_chars'] / df_train['length']
#Sentence readability
print('Flesch index..')
df_train['flesch'] = df_train['question_text'].apply(lambda x: textstat.flesch_reading_ease(x))
print('gunning index..')
df_train['gunning'] = df_train['question_text'].apply(lambda x: textstat.gunning_fog(x))
print('smog index..')
df_train['smog'] = df_train['question_text'].apply(lambda x: textstat.smog_index(x))
print('auto index..')
df_train['auto'] = df_train['question_text'].apply(lambda x: textstat.automated_readability_index(x))
print('coleman index..')
df_train['coleman'] = df_train['question_text'].apply(lambda x: textstat.coleman_liau_index(x))
print('linear index..')
df_train['linsear'] = df_train['question_text'].apply(lambda x: textstat.linsear_write_formula(x))
print('dale index..')
df_train['dale'] = df_train['question_text'].apply(lambda x: textstat.dale_chall_readability_score(x))
#Separate in sincere and insicere dataframes 
sincere_df = df_train[df_train['target']==0]
insincere_df = df_train[df_train['target']==1]
#Make sure we have enough samples for statistical significance
print('Sentences in sincere dataframe:', sincere_df.shape[0])
print('Sentences in insincere dataframe:', insincere_df.shape[0])
#Perform students t-test
check_columns = ['length', 'numbers', 'words', 'spaces', 'other_chars', 'numbers_ratio', 
                 'words_ratio', 'spaces_ratio','other_chars_ratio', 'flesch', 'gunning', 
                 'smog', 'auto', 'coleman', 'linsear', 'dale']

for col in check_columns:
    t2, p2 = stats.ttest_ind(sincere_df[col],insincere_df[col])
    print('t-stat:', t2, '. p-value:', p2)
    if p2<0.05: 
        print('For feature', col, 'means are: DIFFERENT')
    elif p2>=0.05: 
        print('For feature', col, 'means are: SAME')

#Split in train and test
train, valid = train_test_split(df_train, test_size=0.15)
train_x = train.drop(['qid', 'question_text', 'target', 'words', 'length', 
                      'words_ratio', 'spaces'], axis = 1)
train_y = train['target']

valid_x = valid.drop(['qid', 'question_text', 'target', 'words', 'length', 
                      'words_ratio', 'spaces'], axis = 1)
valid_y = valid['target']
#LGB model
lgb_train = lgb.Dataset(train_x, train_y)
lgb_valid = lgb.Dataset(valid_x, valid_y)

# Specify hyper-parameters as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 18,
    'max_depth': 4,
    'learning_rate': 0.05,
    #'feature_fraction': 0.95,
    #'bagging_fraction': 0.8,
    #'bagging_freq': 5,
    #'reg_alpha': 0.1,
    #'reg_lambda': 0.1,
    'is_unbalance': True,
    'num_class': 1,
    #'scale_pos_weight': 3.2,
    'verbose': 1,
}

num_iter = 500

# Train LightGBM model
print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=num_iter,
                valid_sets= lgb_valid,
                early_stopping_rounds=40,
                verbose_eval=20
                )
# Plot Importances
print('Plot feature importances...')
importances = gbm.feature_importance(importance_type='gain')  # importance_type='split'
model_columns = pd.DataFrame(train_x.columns, columns=['features'])
feat_imp = model_columns.copy()
feat_imp['importance'] = importances
feat_imp = feat_imp.sort_values(by='importance', ascending=False)
feat_imp.reset_index(inplace=True)

plt.figure()
plt.barh(np.arange(feat_imp.shape[0] - 1, -1, -1), feat_imp.importance)
plt.yticks(np.arange(feat_imp.shape[0] - 1, -1, -1), (feat_imp.features))
plt.title("Feature Importances")
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
pred_lgb = gbm.predict(valid_x,
                       num_iteration=40
                       )
#Sensitivity analysis
steps = np.arange(0.1,0.7, 0.01)
validation_pred = []
for i in steps:
    valid_pred_01_lstm = np.where(pred_lgb > i, 1, 0)
    valid_pred_01_lstm = [int(item) for item in valid_pred_01_lstm]
    f1_quora = f1_score(valid_y, valid_pred_01_lstm)
    validation_pred.append(f1_quora)

plt.figure()
plt.plot(steps, validation_pred)
plt.grid()
plt.show()