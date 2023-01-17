#Importing some libraries
import numpy as np
import pandas as pd
import datetime
import lightgbm as lgbm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
#Reading data
df = pd.read_csv("../input/bank-classification.csv")
#Creating new dataframe for learning using raw data
new_df = pd.DataFrame()
new_df['id'] = df.id
new_df['campaign'] = df.campaign
new_df['pdays'] = df.pdays
new_df['if_pdays'] = df.pdays != 999
new_df['previous'] = df.previous
new_df['y'] = df.y
new_df.y[new_df.y == 'no'] = False 
new_df.y[new_df.y == 'yes'] = True

new_df['month'] =  pd.to_datetime(df.contact_date).dt.month
new_df['day_of_the_week'] =  pd.to_datetime(df.contact_date).dt.dayofweek

start_date = pd.Timestamp(datetime.datetime(2013, 1, 1))
new_df['life_length'] = (pd.Series([start_date]*df.shape[0]) - pd.to_datetime(df.birth_date)).dt.days
new_df['contact_date'] = (pd.Series([start_date]*df.shape[0]) - pd.to_datetime(df.contact_date)).dt.days
#One hot encoding features
result = pd.DataFrame(np.ndarray((df.shape[0],0)).astype(np.bool))

for c in ['job','marital','education','default','housing','loan','contact','poutcome']:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(df[c])
    ohe = OneHotEncoder(sparse=False, categories='auto')
    result = pd.concat([result,pd.DataFrame(ohe.fit_transform(encoded.reshape(-1,1)).astype(np.bool), columns=[c +'_'+ s for s in encoder.classes_])], axis=1)
    
new_df = pd.concat([new_df,result], axis=1)
#some feature engineering magic
for c in ['contact_date', 'life_length', 'campaign']:
    new_df['log_'+c] = np.log(new_df[c])
poly = PolynomialFeatures(3, interaction_only=True)
new_df = pd.concat([new_df,pd.DataFrame(poly.fit_transform(new_df[['contact_date', 'life_length', 'campaign']]))], axis=1)
#Dropping some features that lgbm says are not important
new_df.drop([1, 'log_campaign', 'loan_unknown', 'education_illiterate', 'marital_unknown', 'education_basic.6y', 'default_unknown', 'log_life_length', 
             'default_no', 2, 'job_technician','education_unknown','log_contact_date', 0,],axis=1,inplace=True)
#splitting sets
test_set = new_df[new_df.y == 'unknown']
train_set = new_df[new_df.y != 'unknown']
train_set.y = train_set.y.astype(np.bool)
test_set.drop(['y'], axis=1, inplace=True)
lgb_train = lgbm.Dataset(train_set.drop(['y'], axis=1), train_set.y, free_raw_data=False)
# specifying LightGBM parameters
params = {
    'boosting_type': 'dart',
    'num_iterations': 254,
    'objective': 'cross_entropy_lambda',
    'metric': {'auc'},
    'num_leaves': 33,
    'n_estimators': 256,
    'learning_rate': 0.05,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.8,
    'bagging_freq': 8,
    'max_bin': 511,
    'min_sum_hessian_in_leaf': 1.49,
    'verbose': 2,
    'seed': None,
}
print('Starting CV...')
r = lgbm.cv(params, lgb_train, categorical_feature=['day_of_the_week', 'month'])
print(sum(r['auc-mean'])/len(r['auc-mean']))
gbm = lgbm.train(params,lgb_train)
y_pred = gbm.predict(test_set, num_iteration=gbm.best_iteration, categorical_feature=['day_of_the_week', 'month'])
sub = pd.DataFrame()
sub['id'] = test_set['id']
sub['y'] = y_pred
sub.to_csv("submission.csv", index=False)
print('Plotting feature importances...')
ax = lgbm.plot_importance(gbm, max_num_features=10)