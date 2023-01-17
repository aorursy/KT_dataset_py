# Imports
%matplotlib inline
%reload_ext autoreload
%autoreload 2
import xgboost as xgb
from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)
from IPython.display import HTML, display
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
from sklearn.model_selection import StratifiedKFold
# Don't know why I can't access the data like that.. anyway the code runs localy.
train_df = pd.read_csv('../input/churnTrain.csv')
test_df = pd.read_csv('../input/churnTest.csv')
train_df.head()
cat_vars = ['State', 'Area_Code', 'International_Plan', 'Voice_Mail_Plan']

contin_vars = ['Account_Length', 'No_Vmail_Messages', 'Total_Day_minutes', 'Total_Day_Calls', 'Total_Day_charge',
               'Total_Eve_Minutes', 'Total_Eve_Calls', 'Total_Eve_Charge', 'Total_Night_Minutes', 'Total_Night_Calls',
               'Total_Night_Charge', 'Total_Intl_Minutes', 'Total_Intl_Calls', 'Total_Intl_Charge', 'No_CS_Calls']

id_var = 'Phone_No'

objective = ['Churn']
train_df['International_Plan'] = train_df['International_Plan'].replace((' yes', ' no'), (True, False)).astype(bool)
train_df['Voice_Mail_Plan'] = train_df['Voice_Mail_Plan'].replace((' yes', ' no'), (True, False)).astype(bool)
train_df['Premium'] = (train_df['International_Plan'] & train_df['Voice_Mail_Plan']).astype(bool)
test_df['International_Plan'] = test_df['International_Plan'].replace((' yes', ' no'), (True, False)).astype(bool)
test_df['Voice_Mail_Plan'] = test_df['Voice_Mail_Plan'].replace((' yes', ' no'), (True, False)).astype(bool)
test_df['Premium'] = (test_df['International_Plan'] & test_df['Voice_Mail_Plan']).astype(bool)
cat_vars += ['Premium']
# Apply categorical type:
for v in cat_vars:
    print(v)
    train_df[v] = train_df[v].astype('category').cat.as_ordered()

apply_cats(test_df, train_df)
# Contin_vars as floats:
for v in contin_vars:
    train_df[v] = train_df[v].fillna(0).astype('float32')
    test_df[v] = test_df[v].fillna(0).astype('float32')
# Process the training data using the awesome fastai function proc_df:
train_df = train_df.set_index(id_var)
train_df = train_df[cat_vars+contin_vars+objective]

df, y, nas, mapper = proc_df(train_df, objective[0], do_scale=True)
# Process the testing data using the awesome fastai function proc_df:
test_df = test_df.set_index(id_var)

# Just a dummy column so that the column exists.
test_df[objective[0]] = 0
test_df = test_df[cat_vars+contin_vars+objective]

df_test, _, nas, mapper = proc_df(test_df, objective[0], do_scale=True,
                                  mapper=mapper, na_dict=nas)
# Create a K-fold instance
k = 4
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
# Train/validate on 5 different training/validation sets thanks to K-folds and predict on the testing set:
predicts = []
test_X = np.array(df_test)
for train_index, test_index in kf.split(df, y):
    print("###")
    X_train, X_val = np.array(df)[train_index], np.array(df)[test_index]
    y_train, y_val = y[train_index], y[test_index]
    
    xgb_params = {
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'max_depth': 7,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 1,
        'colsample_bytree': 1,
        'objective': 'binary:logistic',
        'scale_pos_weight': 1,
        'eval_metric': 'logloss',
        'silent': 1,
        'seed': 27}
    
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_val, y_val)
    d_test = xgb.DMatrix(test_X)
    
    model = xgb.train(xgb_params, d_train, num_boost_round = 10000, evals=[(d_valid, 'eval')], verbose_eval=100, 
                     early_stopping_rounds=100)
                        
    xgb_pred = model.predict(d_test)
    predicts.append(list(xgb_pred))
# Average the 5 predictions sets:
preds=[]
for i in range(len(predicts[0])):
    sum=0
    for j in range(k):
        sum+=predicts[j][i]
    preds.append(sum / k)
# Did not have the time to finetune the thresh, but I think I could have improved my score with thresh smaller than 0.5:
preds = np.array(preds)
thresh = 0.45
results = np.where(preds > thresh, 1, 0)
# Create the results datframe:
df_test['Churn'] = results
sub = df_test[['Churn']].copy()
sub['Churn'].sum()
sub.reset_index(inplace=True, drop=True)
sub = sub.replace(0, 'FALSE')
sub = sub.replace(1, 'TRUE')
sub.index += 1 
sub.head()
sub.to_csv('sample_submission.csv', index=True, index_label='Id')