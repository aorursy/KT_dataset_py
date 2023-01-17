import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import matthews_corrcoef

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV

from sklearn.metrics import make_scorer

from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/train.csv')

train = train.sample(frac=1).reset_index()

test = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/test.csv')

users = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/users.csv')
train.shape
train.head()
users.head()
train.user_id.value_counts()
# Add in feature engineering on date

# full_df['date.weekday']=full_df['date'].dt.weekday

# full_df['date.day']=full_df['date'].dt.day

# full_df['date.month']=full_df['date'].dt.month

# full_df['date.year']=full_df['date'].dt.year
train = pd.merge(train, users, left_on =train.user_id, right_on = users.user_id)

train = train.drop(['key_0', 'index', 'user_id_x', 'grass_date'], axis = 1)



test = pd.merge(test, users, left_on =test.user_id, right_on = users.user_id)

test = test.drop(['key_0', 'user_id_x', 'grass_date'], axis = 1)



train.head()
train.open_flag.value_counts()
train['domain'].value_counts()
train['domain'][~train['domain'].isin(['@gmail.com', '@yahoo.com', '@hotmail.com', '@icloud.com' , '@qq.com', '@outlook.com'])] = 'other'

test['domain'][~test['domain'].isin(['@gmail.com', '@yahoo.com', '@hotmail.com', '@icloud.com' , '@qq.com', '@outlook.com'])] = 'other'
train['domain'].value_counts()
train[['attr_1', 'attr_2', 'attr_3']].sample(10)
train.attr_1.value_counts(dropna = False)
train.attr_2.value_counts(dropna = False)
train.attr_3.value_counts(dropna = False)
train.age.value_counts(dropna = False)
train['attr_1'] = train['attr_1'].fillna(2)

train['attr_2'] = train['attr_2'].fillna(2)



test['attr_1'] = test['attr_1'].fillna(2)

test['attr_2'] = test['attr_2'].fillna(2)
train['age'] = train['age'].fillna(0)



train['age'] = train.age.astype(int)
train['age_group'] = ''



train.loc[train.age == 0, 'age_group'] = 'Null age'

train.loc[(train.age > 0) & (train.age < 16), 'age_group'] = '0-16'

train.loc[(train.age >= 16) & (train.age < 26), 'age_group'] = '16-26'

train.loc[(train.age >= 26) & (train.age < 36), 'age_group'] = '26-36'

train.loc[(train.age >= 36) & (train.age < 50), 'age_group'] = '36-50'

train.loc[(train.age >= 50), 'age_group'] = 'Above 50'



test['age_group'] = ''

test.loc[test.age == 0, 'age_group'] = 'Null age'

test.loc[(test.age > 0) & (test.age < 16), 'age_group'] = '0-16'

test.loc[(test.age >= 16) & (test.age < 26), 'age_group'] = '16-26'

test.loc[(test.age >= 26) & (test.age < 36), 'age_group'] = '26-36'

test.loc[(test.age >= 36) & (test.age < 50), 'age_group'] = '36-50'

test.loc[(test.age >= 50), 'age_group'] = 'Above 50'



train = train.drop('age', axis = 1)

test = test.drop('age', axis = 1)
train.sample(100)
train['last_open_day']=train['last_open_day'].replace({'Never open':999})

train['last_checkout_day']=train['last_checkout_day'].replace({'Never checkout':999})

train['last_login_day']=train['last_checkout_day'].replace({'Never login':999})



test['last_open_day']=test['last_open_day'].replace({'Never open':999})

test['last_checkout_day']=test['last_checkout_day'].replace({'Never checkout':999})

test['last_login_day']=test['last_checkout_day'].replace({'Never login':999})
train['open_count_10_to_30'] = train['open_count_last_30_days'] - train['open_count_last_10_days']

train['open_count_30_to_60'] = train['open_count_last_60_days'] - train['open_count_last_30_days']

train['login_count_10_to_30'] = train['login_count_last_30_days'] - train['login_count_last_10_days']

train['login_count_30_to_60'] = train['login_count_last_60_days'] - train['login_count_last_30_days']

train['checkout_count_10_to_30'] = train['checkout_count_last_30_days'] - train['checkout_count_last_10_days']

train['checkout_count_30_to_60'] = train['checkout_count_last_60_days'] - train['checkout_count_last_30_days']



test['open_count_10_to_30'] = test['open_count_last_30_days'] - test['open_count_last_10_days']

test['open_count_30_to_60'] = test['open_count_last_60_days'] - test['open_count_last_30_days']

test['login_count_10_to_30'] = test['login_count_last_30_days'] - test['login_count_last_10_days']

test['login_count_30_to_60'] = test['login_count_last_60_days'] - test['login_count_last_30_days']

test['checkout_count_10_to_30'] = test['checkout_count_last_30_days'] - test['checkout_count_last_10_days']

test['checkout_count_30_to_60'] = test['checkout_count_last_60_days'] - test['checkout_count_last_30_days']



# train = train.drop(['open_count_last_30_days','open_count_last_60_days','login_count_last_30_days','login_count_last_60_days','checkout_count_last_30_days','checkout_count_last_60_days'], axis = 1)

# test = test.drop(['open_count_last_30_days','open_count_last_60_days','login_count_last_30_days','login_count_last_60_days','checkout_count_last_30_days','checkout_count_last_60_days'], axis = 1)
train.head()
train = train.join(pd.get_dummies(train['domain'])).drop('domain', axis=1)

train = train.join(pd.get_dummies(train['age_group'])).drop('age_group', axis=1)



test = test.join(pd.get_dummies(test['domain'])).drop('domain', axis=1)

test = test.join(pd.get_dummies(test['age_group'])).drop('age_group', axis=1)



train.head()
pd.get_dummies(train['attr_1'], prefix='attr_1')
train = train.join(pd.get_dummies(train['attr_1'], prefix='attr_1')).drop('attr_1', axis=1)

train = train.join(pd.get_dummies(train['attr_2'], prefix='attr_2')).drop('attr_2', axis=1)

train = train.join(pd.get_dummies(train['attr_3'], prefix='attr_3')).drop('attr_3', axis=1)

#train = train.drop('row_id', axis=1)



test = test.join(pd.get_dummies(test['attr_1'], prefix='attr_1')).drop('attr_1', axis=1)

test = test.join(pd.get_dummies(test['attr_2'], prefix='attr_2')).drop('attr_2', axis=1)

test = test.join(pd.get_dummies(test['attr_3'], prefix='attr_3')).drop('attr_3', axis=1)

#test = test.drop('row_id', axis=1)

train.columns
X = train.drop(['open_flag'],axis=1)

y = train['open_flag']



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=101)
from imblearn.under_sampling import NearMiss

from imblearn.over_sampling import SMOTE





ros = SMOTE()

ros.fit(X_train, y_train)

X_train_sampled, y_train_sampled = ros.fit_sample(X_train, y_train)
print(sum(y)/len(y))

print(sum(y_resampled)/len(y_resampled))
## Random Forest
matt_scorer = make_scorer(matthews_corrcoef)



parameters = {'criterion': ['gini'],

               'max_depth': [20, 25, 40, 60, 80],

               'min_samples_leaf': [2, 4, 8],

               'min_samples_split': [2, 4, 6, 10, 12, 14, 18],

               'n_estimators': [20, 40, 80, 160, 320],

               'class_weight': ['balanced', 'balanced_subsample', None]

               }



rs_rf = RandomizedSearchCV(RandomForestClassifier(random_state=1),

                                            parameters,

                                            cv=StratifiedKFold(

                                                n_splits=10, shuffle=True, random_state=2019),

                                           scoring = matt_scorer,

                                            verbose=2, n_jobs=-1)



rs_rf.fit(X_train, y_train)

print('best parameters: ', rs_rf.best_params_)
rfc = RandomForestClassifier(n_estimators= 160, min_samples_split= 18, min_samples_leaf= 4, max_depth= 80, criterion= 'gini', class_weight= 'balanced')



rfc.fit(X_train_sampled, y_train_sampled)

rfc_val_prediction_undersampled = rfc.predict(X_val)



print(matthews_corrcoef(y_val, rfc_val_prediction_undersampled))

# 0.6920714135485448



rfc.fit(X_train,y_train)

rfc_val_prediction = rfc.predict(X_val)



print(matthews_corrcoef(y_val, rfc_val_prediction))

# 0.5308942057408922
rfc = RandomForestClassifier(n_estimators= 160, min_samples_split= 18, min_samples_leaf= 4, max_depth= 80, criterion= 'gini', class_weight= 'balanced')



rfc.fit(X_train_sampled, y_train_sampled)
val_proba = rfc.predict_proba(X_val)
#threshold search

highest_score = 0

best_threshold = 0

for i in range(100):

    threshold = i/100



    predicted = (val_proba [:,1] >= threshold).astype('int')

    

    score = matthews_corrcoef(y_val, predicted)

    if score > highest_score:

        highest_score = score

        best_threshold = threshold

    #print(score, threshold)    

        

print(best_threshold, highest_score)    
train_predictions = rfc.predict_proba(test)

prediction_after_threshold = (train_predictions [:,1] >= best_threshold).astype('int')

prediction_df = pd.DataFrame(prediction_after_threshold)



submission = test[['row_id','country_code']].join(prediction_df)

submission = submission.rename({0:'open_flag'}, axis=1)

submission['row_id'] = submission.index

submission = submission[['row_id', 'open_flag']]

submission.tail()



submission.to_csv('submission_rfc.csv', index=False)
rfc_importance= pd.DataFrame.from_dict({'feature':list(X.columns), 'importance': rfc.feature_importances_ })

print(rfc_importance.sort_values('importance',ascending=False).head(20))
#old 0.5228940686243029
## Gradient Boosting
# from sklearn.ensemble import GradientBoostingClassifier



# matt_scorer = make_scorer(matthews_corrcoef)



# parameters = {'loss': ['deviance', 'exponential'],

#                'learning_rate': [0.1,0.2, 0.3, 0.4],

#                'n_estimators': [50, 100, 150, 200, 250],

#                'subsample': [0.9, 1.0]

#                }



# rs_xgb = RandomizedSearchCV(GradientBoostingClassifier(random_state=1),

#                                             parameters,

#                                             cv=StratifiedKFold(

#                                                 n_splits=10, shuffle=True, random_state=2019),

#                                            scoring = matt_scorer,

#                                             verbose=2, n_jobs=-1)



# rs_xgb.fit(X, y)

# print('best parameters: ', rs_xgb.best_params_)
# # gbc = GradientBoostingClassifier(subsample= 1.0, n_estimators= 100, min_samples_split= 2, min_samples_leaf= 4, max_leaf_nodes= None, max_depth= 6, loss= 'exponential', learning_rate= 0.2)

# gbc = GradientBoostingClassifier(subsample= 0.9, n_estimators= 250, loss= 'deviance', learning_rate= 0.2)



# gbc.fit(X,y)



# gbc_predictions = gbc.predict(X_val)

# matthews_corrcoef(y_val, gbc_predictions)
# train_predictions = gbc.predict(test)



# prediction_df = pd.DataFrame(train_predictions)



# submission = test[['row_id','country_code']].join(prediction_df)

# submission = submission.rename({0:'open_flag'}, axis=1)

# submission['row_id'] = submission.index

# submission = submission[['row_id', 'open_flag']]

# submission.tail()



# submission.to_csv('submission_xgb.csv', index=False)
# from sklearn.ensemble import AdaBoostClassifier

# from sklearn.tree import DecisionTreeClassifier
# matt_scorer = make_scorer(matthews_corrcoef)



# parameters = {'base_estimator__criterion' : ["gini", "entropy"], 

#               'base_estimator__splitter' :   ["best", "random"],

#               'base_estimator__max_depth': [15, 20, 25, 30], 

#               'base_estimator__min_samples_split': list(range(15, 30, 5)),

#               'algorithm': ['SAMME', 'SAMME.R'], 'learning_rate': [0.001, 0.01, 0.1, 0.5, 1], 

#               'n_estimators': list(range(50, 200, 50))}



# DTC = DecisionTreeClassifier(random_state = 2019, max_features = "auto", class_weight = "balanced")



# rs_ab = RandomizedSearchCV(AdaBoostClassifier(random_state = 2019, base_estimator = DTC), 

#                                   parameters, scoring = matt_scorer, 

#                                   cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2019),

#                                        iid=False, verbose=2, n_jobs=-1)



# rs_ab.fit(X, y)

# print('best parameters: ', rs_ab.best_params_)
# #adaboost = AdaBoostClassifier(n_estimators= 150, learning_rate= 0.001, splitter= 'best', base_estimator__min_samples_split= 20, base_estimator__max_depth= 15, base_estimator__criterion= 'entropy', algorithm= 'SAMME.R')



# adaboost = AdaBoostClassifier(algorithm='SAMME.R',

#                    base_estimator=DecisionTreeClassifier(criterion='entropy',

#                                                          max_depth=15,

#                                                          min_samples_leaf=1,

#                                                          min_samples_split=20,

#                                                          random_state=2019,

#                                                          splitter='best'),

#                    learning_rate=0.001, n_estimators=150, random_state=2019)



# adaboost.fit(X,y)



# adaboost_predictions = adaboost.predict(X_val)

# matthews_corrcoef(y_val, adaboost_predictions)
# train_predictions = adaboost.predict(test)



# prediction_df = pd.DataFrame(train_predictions)



# submission = test[['row_id','country_code']].join(prediction_df)

# submission = submission.rename({0:'open_flag'}, axis=1)

# submission['row_id'] = submission.index

# submission = submission[['row_id', 'open_flag']]

# submission.tail()



# submission.to_csv('submission_ada.csv', index=False)
# after getting 3 models, to combine them and start a threshold search