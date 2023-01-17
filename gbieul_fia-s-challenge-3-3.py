#Importing everything we'll need for this notebook 

import pandas as pd

import numpy as np



#mlxtend -> for stacking



# If we want to pipeline things

from sklearn.pipeline import Pipeline, make_pipeline



# Helpers, transformers, etc

from sklearn.metrics import auc, make_scorer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import (KFold, train_test_split, cross_val_score, 

                                     cross_val_predict, GridSearchCV, RandomizedSearchCV)



# Under and oversampling

from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler



# Models

from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)



# Base (for inheritance)

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, ClassifierMixin, clone



# Visualizations

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt



import random

# Seed

random_State_Seed = random.seed(42)



# passar dentro do onehotencoder e labelencoder as colunas que sao categoricas
# Path of the file to read

train_path = '../input/train_data.csv'

train_orig = pd.read_csv(train_path)





# path to file used for predictions

predict_path = '../input/data_no_label.csv'

predict_data = pd.read_csv(predict_path)
#Train Columns

print('Train data columns\n', train_orig.columns.tolist(), '\n')

print('Train data columns length\n', len(train_orig.columns.tolist()), '\n\n')



#Test Columns

print('Predict data columns\n', predict_data.columns.tolist(), '\n')

print('Predict data columns length\n', len(predict_data.columns.tolist()), '\n')
# What's the type of each column?

train_orig.dtypes
# Which unique values do we have for each column?

for column in train_orig:

    print(column, 'column unique values:\n')

    print(train_orig[column].unique(), '\n\n')
# Is there any class imbalance? We expect to, since clients tends to pay their debts (usually)



plt.figure()

ax = sns.countplot(data=train_orig, x='default');

ax.set_title("Countplot for *default* values");
# What's the numerical correlations?

plt.figure(figsize=(20,10))

corr = train_orig.select_dtypes(exclude ='object').corr()

corr.index = train_orig.select_dtypes(exclude ='object').columns

sns.heatmap(corr, annot = True, cmap='RdYlGn', vmin=-1, vmax=1)

plt.title("Numerical Correlation Heatmap", fontsize=16)

plt.show()
# Let's make a new category for the data that has just a few values per category

score_2_count = train_orig['score_2'].value_counts()[train_orig['score_2'].value_counts() <100]

state_count = train_orig['state'].value_counts()[train_orig['state'].value_counts() <100]

zip_count = train_orig['zip'].value_counts()[train_orig['zip'].value_counts() <120]

real_state_count = train_orig['real_state'].value_counts()[train_orig['real_state'].value_counts() <50]

job_name_count = train_orig['job_name'].value_counts()[train_orig['job_name'].value_counts() <80] # best was 80

reason_count = train_orig['reason'].value_counts()[train_orig['reason'].value_counts() <80] # best was 80
# Changing values!



train_orig.loc[train_orig['score_2']

               .isin((score_2_count).index), 'score_2'] = 'other'



train_orig.loc[train_orig['state']

               .isin((state_count).index), 'state'] = 'other'



train_orig.loc[train_orig['zip']

               .isin((zip_count).index), 'zip'] = 'other'





train_orig.loc[train_orig['real_state']

               .isin((real_state_count).index), 'real_state'] = 'other'



train_orig.loc[train_orig['job_name']

               .isin((job_name_count).index), 'job_name'] = 'other'



train_orig.loc[train_orig['reason']

               .isin((reason_count).index), 'reason'] = 'other'
predict_data.loc[predict_data['score_2']

               .isin((score_2_count).index), 'score_2'] = 'other'



predict_data.loc[predict_data['state']

               .isin((state_count).index), 'state'] = 'other'



predict_data.loc[predict_data['zip']

               .isin((zip_count).index), 'zip'] = 'other'





predict_data.loc[predict_data['real_state']

               .isin((real_state_count).index), 'real_state'] = 'other'



predict_data.loc[predict_data['job_name']

               .isin((job_name_count).index), 'job_name'] = 'other'



predict_data.loc[predict_data['reason']

               .isin((reason_count).index), 'reason'] = 'other'
train_orig['zip'].value_counts().describe()
for column in train_orig.select_dtypes(include=['object']):

    print(column, 'column unique values:')

    print(len(train_orig[column].unique()), '\n')
#Categorical engineering

train_orig['absent_credit_limit'] = np.where(train_orig['credit_limit'].isna(), 'yes', 'no')

predict_data['absent_credit_limit'] = np.where(predict_data['credit_limit'].isna(), 'yes', 'no')





bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 60000, 80000, 100000,

        150000, 220000, 260000, 300000, 340000, np.inf]



labels = [f'{i}+' if j==np.inf else f'{i}-{j}' for i, j in zip(bins, bins[1:])]



train_orig['credit_limit_bin'] = pd.cut(train_orig['credit_limit'], bins, labels)

train_orig['credit_limit_bin'] = train_orig['credit_limit_bin'].astype(str)

train_orig['credit_limit_bin'] = train_orig['credit_limit_bin'].replace('nan', 'Was_NaN')



predict_data['credit_limit_bin'] = pd.cut(predict_data['credit_limit'], bins, labels)

predict_data['credit_limit_bin'] = predict_data['credit_limit_bin'].astype(str)

predict_data['credit_limit_bin'] = predict_data['credit_limit_bin'].replace('nan', 'Was_NaN')
# Numerical engineering

predict_data['dif'] = predict_data['income'] - predict_data['amount_borrowed']

predict_data['income_credit_limit_dif'] = predict_data['income'] -  predict_data['credit_limit']

predict_data['income_borinmonths_dif'] = predict_data['income'] -  predict_data['borrowed_in_months']



predict_data['dif'] = predict_data['income'] - predict_data['amount_borrowed']

predict_data['income_credit_limit_dif'] = predict_data['income'] -  predict_data['credit_limit']

predict_data['income_borinmonths_dif'] = predict_data['income'] -  predict_data['borrowed_in_months']


train_orig['income_range'] = pd.qcut(train_orig['income'], 10)





_, bins = pd.qcut(train_orig['income'], 10, retbins = True)

bins = np.concatenate(([-np.inf], bins[1:-1], [np.inf]))

predict_data['income_range'] = pd.cut(predict_data['income'], bins)





train_orig['income_range'] = train_orig['income_range'].astype(str)

predict_data['income_range'] = predict_data['income_range'].astype(str)
train_orig['amount_borrowed_bins'] = pd.qcut(train_orig['amount_borrowed'], 10)





_, bins = pd.qcut(train_orig['amount_borrowed'], 10, retbins = True)

bins = np.concatenate(([-np.inf], bins[1:-1], [np.inf]))



predict_data['amount_borrowed_bins'] = pd.cut(predict_data['amount_borrowed'], bins)

#pd.cut(predict_data['amount_borrowed'], bins, labels)



train_orig['amount_borrowed_bins'] = train_orig['amount_borrowed_bins'].astype(str)

predict_data['amount_borrowed_bins'] = predict_data['amount_borrowed_bins'].astype(str)
train_orig.dtypes
percent_missing = train_orig.isnull().sum() * 100 / len(train_orig)

missing_value_df = pd.DataFrame({'column_name': train_orig.columns,

                                 'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', inplace=True)
missing_value_df
# Removing the columns which we don't want to process

object_columns = train_orig.select_dtypes(include='object').columns.tolist()



object_columns = [element for element in object_columns if element not in ('ids', 'default')]
# Replace NaN values of the selected columns above with a new category

train_orig[object_columns] = train_orig[object_columns].fillna("was_NaN")
# Also for the prediction/test data

predict_data[object_columns] = predict_data[object_columns].fillna("was_NaN")
# Taking care of some column types over here (The .fillna may confuse Pandas sometimes!)

train_orig[object_columns] = train_orig[object_columns].astype(str)

predict_data[object_columns] = predict_data[object_columns].astype(str)
train_orig.isna().sum()
cat_atribs = train_orig.select_dtypes(include=['object', 'category']).columns.tolist()

cat_atribs = [e for e in cat_atribs if e not in ('ids', 'default')]

num_atribs = train_orig.select_dtypes(exclude=['object', 'category']).columns.tolist()

# commented row below is due to a possible further intention to exclude this column from the analysis

num_atribs = [e for e in num_atribs if e not in ('n_issues', 'credit_limit')]
# Here we're defining which col is the Y

Y_col = ['default']



# And here we're splitting the rows without the Y.

train_no_mis_y = train_orig.dropna(subset=['default'])



    # The missing Y rows are stored on this variable.

train_mis = train_orig[train_orig['default'].isna()]
# Now, we're effectively defining the categorical 

# and numerical features for both train and test data!



train_X_cat = train_no_mis_y[cat_atribs]

train_X_num = train_no_mis_y[num_atribs]

train_y = train_no_mis_y[Y_col]



test_X_cat = predict_data[cat_atribs]

test_X_num = predict_data[num_atribs]



# Our model doesn't accept values like 'True' or 'False'

# We have to replace it for 1 and 0

train_y = train_y.replace(True, 1)

train_y = train_y.replace(False, 0)
# One Hot Encoder: It's the responsible for encoding the categoricals.

enc = OneHotEncoder(handle_unknown='ignore', sparse = False)



# Imputer: the responsible for filling the missing numericals

imputer = SimpleImputer(strategy='mean', missing_values=np.nan)



# StandardScaler: the responsible for normalization

scaler = StandardScaler()



# Now, we have to fit these guys, then transform the dataframe

imputer.fit(train_X_num)

train_X_num_imp = pd.DataFrame(imputer.transform(train_X_num))

train_X_num_imp.columns = train_X_num.columns.tolist()



test_X_num_imp = pd.DataFrame(imputer.transform(test_X_num))

test_X_num_imp.columns = test_X_num.columns.tolist()
# The scaler is over here. We can choose whether to run or skip it. For this problem,

# it was better not to transform the original data. But we may check for outliers.

#scaler.fit(train_X_num_imp)

#train_X_num_scl = pd.DataFrame(scaler.transform(train_X_num_imp))

#train_X_num_scl.columns = train_X_num_imp.columns.tolist()

#train_X_num_imp = train_X_num_imp1
pd.set_option('display.float_format', lambda x: '%.4f' % x)

#train_X_num_scl.describe()
# Fitting the encoder

enc.fit(train_X_cat)



train_X_cat_enc = pd.DataFrame(enc.transform(train_X_cat))

test_X_cat_enc = pd.DataFrame(enc.transform(test_X_cat))
# So we use concatenation!

train_X_all = pd.concat([train_X_num_imp, train_X_cat_enc], axis = 1)



test_X_all = pd.concat([test_X_num_imp, test_X_cat_enc], axis = 1)
# What's the numerical correlations... again?

plt.figure(figsize=(20,10))

corr = train_X_num_imp.select_dtypes(exclude ='object').corr()

corr.index = train_X_num_imp.select_dtypes(exclude ='object').columns

sns.heatmap(corr, annot = True, cmap='RdYlGn', vmin=-1, vmax=1)

plt.title("Numerical Correlation Heatmap", fontsize=16)

plt.show()
ros = RandomOverSampler(random_state=0)

ros.fit(train_X_all, train_y)

train_X_all_resampled, train_y_resampled = ros.fit_resample(train_X_all, train_y)
# Random Forest Parameters

# This test can be run to find the best parameters used on the next cells

"""rf_grid = {

    #'n_jobs': -1,

    'n_estimators': [50, 100, 150],

     #'warm_start': True,

    'max_features': [2, 4, 6, 15],

    'max_depth': [6, 8, 12],

    'min_samples_leaf': [2, 5, 8],

    #'max_features' : ['sqrt'],

    #'verbose': 10    

}



rf_gridcv = GridSearchCV(estimator = RandomForestClassifier(random_state=random_State_Seed),

                          #n_iter = 30,

                         scoring = 'roc_auc',

                          param_grid = rf_grid,

                          #param_distributions = rf_grid, 

                          cv = 3,

                          n_jobs = -1,

                          verbose = 10)

rf_gridcv.fit(train_X_all_resampled, train_y_resampled) #.values.ravel()"""
# rf_gridcv.best_score_ # with all transforms 0.7629135

#rf_gridcv.best_estimator_ #0.7636621 | 0.7200122 | oversampling 0.81481
####Parameters - best estimators from the above test

pre_rf_params = {'max_depth': 12,

                 'max_features': 15,

                 'min_samples_leaf': 2,

                'n_estimators': 150,

                 'random_state': random_State_Seed}



pre_rf = RandomForestClassifier()



pre_rf.fit(train_X_all, train_y.values.ravel())
# Let's organize to see each feature's score

pd.set_option('display.float_format', lambda x: '%.6f' % x)

col_list = pd.DataFrame(train_X_all.columns.tolist())

importances = pd.DataFrame(pre_rf.feature_importances_)



importances_df = pd.concat([col_list, importances], axis = 1)

importances_df.columns = ['feature', 'importance']

importances_df = importances_df.sort_values(by='importance', ascending=False, axis = 0)

importances_df
# Some old tests

#xgb_params = {'subsample': 0.3, 'reg_lambda': 0.5, 'reg_alpha': 0.5, 

#              'n_estimators': 900, 'max_depth' : 60, 'learning_rate': 0.01, 

#              'gamma': 0.5, 'verbosity': 3, 'random_state': 7, 'n_jobs': -1}







# both engineered, fst bucket

#rf_params = {'max_depth': 18, 'max_features': 15, 'min_samples_leaf': 8, 'n_estimators': 150, 'random_state': 7}

#et_params = {'max_depth': 18, 'max_features': 12, 'min_samples_leaf': 6, 'n_estimators': 110, 'random_state': 7}

#ada_params = {'learning_rate': 0.01, 'n_estimators': 800, 'random_state': 7}

#lgbm_params =  {'learning_rate': 0.01, 'max_depth ': 15, 'n_estimators': 800, 

#                'num_leaves': 15, 'subsample': 0.08, 'random_state': 7}

#xgb_params = {'subsample': 0.3, 'reg_lambda': 0.5, 'reg_alpha': 0.5, 

#              'n_estimators': 900, 'max_depth' : 60, 'learning_rate': 0.01, 

#              'gamma': 0.5, 'verbosity': 3, 'random_state': 7, 'n_jobs': -1}





#Newer

#et_params = {'max_depth': 16, 'max_features': 15, 'min_samples_leaf': 4, 'n_estimators': 50}

#ada_params = {'learning_rate': 0.05, 'n_estimators': 1000}

#lgbm_params = {'learning_rate': 0.1, 'max_depth ': 60, 'n_estimators': 100, 'num_leaves': 25, 'subsample': 0.05}





# old lgbm {'learning_rate': 0.05, 'max_depth ': 60, 'n_estimators': 100, 'num_leaves': 25, 'subsample': 0.08}





#ada {'learning_rate': 0.01, 'max_depth ': 15, 'n_estimators': 800, 'num_leaves': 15, 'subsample': 0.08}

#lgbm {'learning_rate': 0.01, 'max_depth ': 15, 'n_estimators': 800, 'num_leaves': 15, 'subsample': 0.08}



#xgb score=0.7759465774621821

#{'subsample': 0.3, 'reg_lambda': 0.5, 'reg_alpha': 0.5, 

#'n_estimators': 900, 'max_depth' : 60, 'learning_rate': 0.01, 'gamma': 0.5}





#new prms, both engin. feat, fst bucket

#rf {'max_depth': 18, 'max_features': 15, 'min_samples_leaf': 8, 'n_estimators': 150}

#et {'max_depth': 18, 'max_features': 12, 'min_samples_leaf': 6, 'n_estimators': 110}

#lgbm {'learning_rate': 0.01, 'max_depth ': 15, 'n_estimators': 800, 'num_leaves': 15, 'subsample': 0.08}

#xgb_params = {'subsample': 0.3, 'reg_lambda': 0.5, 'reg_alpha': 0.5, 

#              'n_estimators': 900, 'max_depth' : 60, 'learning_rate': 0.01, 

#              'gamma': 0.5, 'verbosity': 3, 'random_state': 7, 'n_jobs': -1}
# Some models parameters that have been tested with GridSearchCV before

rf_params = {'max_depth': 18, 'max_features': 15, 'min_samples_leaf': 8, 'n_estimators': 150}

et_params = {'max_depth': 18, 'max_features': 18, 'min_samples_leaf': 10, 'n_estimators': 110}

ada_params = {'learning_rate': 0.1, 'n_estimators': 500}

lgbm_params = {'learning_rate': 0.01, 'max_depth ': 15, 'n_estimators': 800, 'num_leaves': 15, 'subsample': 0.08}
# Models

rf_clf = RandomForestClassifier(**rf_params)

et_clf = ExtraTreesClassifier(**et_params)

ada_clf = AdaBoostClassifier(**ada_params)

lgbm_clf = LGBMClassifier(**lgbm_params)
# Fitting

rf_clf.fit(train_X_all, train_y.values.ravel())
# Fitting

et_clf.fit(train_X_all, train_y.values.ravel())
# Fitting

ada_clf.fit(train_X_all, train_y.values.ravel())
# Fitting

lgbm_clf.fit(train_X_all, train_y.values.ravel())
# Create a structure with the predicted probas - train

rf_probas = rf_clf.predict_proba(train_X_all)

et_probas = et_clf.predict_proba(train_X_all)

ada_probas = ada_clf.predict_proba(train_X_all)

lgbm_probas = lgbm_clf.predict_proba(train_X_all)



struct1 = {'rf' : rf_probas[:,1],

         'et' : et_probas[:,1],

         'ada' : ada_probas[:,1],

         'lgbm' : lgbm_probas[:,1],

         #'xgb' : xgb_probas[:,1]

         }



struct = pd.DataFrame(struct1)

struct.head()
# Create a structure with the predicted probas - test

rf_probas = rf_clf.predict_proba(test_X_all)

et_probas = et_clf.predict_proba(test_X_all)

ada_probas = ada_clf.predict_proba(test_X_all)

lgbm_probas = lgbm_clf.predict_proba(test_X_all)



struct2 = {'rf' : rf_probas[:,1],

         'et' : et_probas[:,1],

         'ada' : ada_probas[:,1],

         'lgbm' : lgbm_probas[:,1],

         #'xgb' : xgb_probas[:,1]

         }



structtest = pd.DataFrame(struct2)

structtest.head()
# Concat dataframes

train_X_all_probs = pd.concat([train_X_all, struct], axis = 1)

test_X_all_probs = pd.concat([test_X_all, structtest], axis = 1)
# generate a model over the predicted probas

xgb_final_params = {'subsample': 1,

                    'reg_lambda': 1.5,

                    'reg_alpha': 0,

                    'n_estimators': 900,

                    'max_depth ': 60,

                    'learning_rate': 1,

                    'gamma': 0.5,

                    'n_jobs': -1,

                    'random_state': 7}



xgb_clf_final = XGBClassifier(**xgb_final_params)



xgb_clf_final.fit(train_X_all_probs, train_y.values.ravel())
# So... we could average these guys and send predictions with them!

avg_preds = np.mean(struct, axis = 1)
output1 = pd.DataFrame({'ids': predict_data.ids,

                       'prob': avg_preds})

                        

output1.head()
output1.to_csv('submission.csv', index = False) # with LGB learning rate 0.04 and all created columns scored  0.77453
# Trying a refitted model approach

refit_probs = xgb_clf_final.predict_proba(test_X_all_probs)
output2 = pd.DataFrame({'ids': predict_data.ids,

                       'prob': refit_probs[:,1]})



output2.head()
output2.to_csv('submission.csv', index = False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
# create a link to download the dataframe

create_download_link(output2)
"""xgb_pipeline_grid = {

    'Classifier__learning_rate': [0.05, 0.07, 0.10], 

    'Classifier__n_estimators': [800, 1000, 1500],

    'Classifier__seed': [random_State_Seed],

    'Classifier__max_depth ': [100, 300, 500]

}



fit_params={ #"Classifier__early_stopping_rounds": 5, 

            "Classifier__eval_metric" : "auc"}"""
"""xgb_pipeline_gridsearch = GridSearchCV(estimator = pipe,  # passando o objeto do Pipeline aqui

                              param_grid = xgb_pipeline_grid,

                                       fit_params = fit_params,

                                       cv = 5,

                                       n_jobs = -1,

                                       verbose = 1,

                                       scoring = 'roc_auc')



xgb_pipeline_gridsearch.fit(train_X, train_y) #**fit_params"""
"""rf_rand = GridSearchCV(estimator = RandomForestClassifier(random_state=random_State_Seed),

                          #n_iter = 30,

                       param_grid = rf_grid,

                       #param_distributions = rf_grid,

                       scoring = 'roc_auc',

                       cv = 3,

                       n_jobs = -1,

                       verbose = 10)

rf_rand.fit(train_X_all, train_y.values.ravel())"""



"""

lgbm_rand = GridSearchCV(estimator = LGBMClassifier(random_state=random_State_Seed),

                          #n_iter = 30,

                          param_grid = lgbm_grid,

                          #param_distributions = lgbm_grid,

                         scoring = 'roc_auc',

                          cv = 3,

                          n_jobs = -1,

                          verbose = 10)

lgbm_rand.fit(train_X_all_resampled, train_y_resampled)"""
"""et_rand = GridSearchCV(estimator = ExtraTreesClassifier(random_state=random_State_Seed),

                          #n_iter = 30,

                          param_grid = et_grid,

                          #param_distributions = et_grid,

                       scoring = 'roc_auc',

                          cv = 3,

                          n_jobs = -1,

                          verbose = 10)

et_rand.fit(train_X_all_resampled, train_y_resampled)"""
"""ada_rand = GridSearchCV(estimator = AdaBoostClassifier(random_state=random_State_Seed),

                          #n_iter = 30,

                          param_grid = ada_grid,

                          #param_distributions = ada_grid,

                        scoring = 'roc_auc',

                          cv = 3,

                          n_jobs = -1,

                          verbose = 10)

ada_rand.fit(train_X_all_resampled, train_y_resampled)"""
"""#rf_params = rf_rand.best_params_

et_params = et_rand.best_params_

ada_params = ada_rand.best_params_

lgbm_params = lgbm_rand.best_params_"""
"""#xgb_params = 

xgbgrid.best_params_"""
"""print('rf_params:',rf_params)

print('et_params:',et_params)

print('ada_params:',ada_params)

print('lgbm_params:',lgbm_params)



rf_params: {'max_depth': 12, 'max_features': 15, 'min_samples_leaf': 2, 'n_estimators': 50}

et_params: {'max_depth': 16, 'max_features': 15, 'min_samples_leaf': 4, 'n_estimators': 50}

ada_params: {'learning_rate': 0.05, 'n_estimators': 1000}

lgbm_params: {'learning_rate': 0.04, 'max_depth ': 60, 'n_estimators': 100, 'num_leaves': 25, 'subsample': 0.08}"""
"""#svc_rand.best_params_

#{'n_estimators': 100, 'min_samples_leaf': 2, 'max_depth': 12}

#{'n_estimators': 100, 'min_samples_leaf': 5, 'max_depth': 12}

#{'n_estimators': 900, 'learning_rate': 0.05}



#Old params

#rf_params = {'n_estimators': 100, 'min_samples_leaf': 2, 'max_depth': 12}

#et_params = {'n_estimators': 100, 'min_samples_leaf': 5, 'max_depth': 12}

#ada_params = {'n_estimators': 900, 'learning_rate': 0.05}

#lgbm_params = {'n_estimators': 100, 'min_samples_leaf': 2, 'max_depth': 12}



rf_params = {'max_depth': 12, 'max_features': 15, 'min_samples_leaf': 2, 'n_estimators': 50}

et_params = {'max_depth': 16, 'max_features': 15, 'min_samples_leaf': 4, 'n_estimators': 50}

ada_params = {'learning_rate': 0.05, 'n_estimators': 1000}

lgbm_params = {'learning_rate': 0.04, 'max_depth ': 60, 'n_estimators': 100, 'num_leaves': 25, 'subsample': 0.08}





rf_clf = RandomForestClassifier(**rf_params)

et_clf = ExtraTreesClassifier(**et_params)

ada_clf = AdaBoostClassifier(**ada_params)

lgbm_clf = LGBMClassifier(**lgbm_params)



rf_clf.fit(train_X_all, train_y.values.ravel())

et_clf.fit(train_X_all, train_y.values.ravel())

ada_clf.fit(train_X_all, train_y.values.ravel())

lgbm_clf.fit(train_X_all, train_y.values.ravel())"""
"""rf_probas = rf_clf.predict_proba(train_X_all)

et_probas = et_clf.predict_proba(train_X_all)

ada_probas = ada_clf.predict_proba(train_X_all)

lgbm_probas = lgbm_clf.predict_proba(train_X_all)

struct = {'rf' : rf_probas[:,1],

         'et' : et_probas[:,1],

         'ada' : ada_probas[:,1],

         'lgbm' : lgbm_probas[:,1]}"""
"""meta_data = pd.DataFrame(struct)"""
"""logistic = LogisticRegression(random_state = random_State_Seed, n_jobs = -1)"""
"""# LGBM parameters 

# LGBM parameters 

lgbm_g = {'learning_rate': [0.01, 0.1, 1, 10, 100],

             'n_estimators': [100, 200],

             'max_depth ': [60, 50],

             'num_leaves': [25, 31],

             'subsample': [0.05, 0.08, 0.10] }



lgbm_gcv = GridSearchCV(estimator = LGBMClassifier(random_state=random_State_Seed),

                          #n_iter = 30,

                          param_grid = lgbm_g,

                          #param_distributions = rf_grid, 

                          cv = 3,

                          n_jobs = -1,

                          verbose = 10)

#lgbm_gcv.fit(meta_data, train_y.values.ravel())

lgbm_gcv.fit(train_X_all, train_y.values.ravel())"""
"""lgbm_gcv.best_params_"""

# com todas as features criadas 0.8536453
"""rf_probast = rf_clf.predict_proba(test_X_all)

et_probast = et_clf.predict_proba(test_X_all)

ada_probast = ada_clf.predict_proba(test_X_all)

lgbm_probast = lgbm_clf.predict_proba(test_X_all)

structt = {'rf' : rf_probast[:,1],

         'et' : et_probast[:,1],

         'ada' : ada_probast[:,1],

         'lgbm' : lgbm_probast[:,1]}



structt = pd.DataFrame(structt)"""
"""predictions = lgbm_gcv.predict_proba(structt)"""
"""output = pd.DataFrame({'ids': predict_data.ids,

                       'prob': predictions[:,1]})



output.to_csv('submission.csv', index = False)"""