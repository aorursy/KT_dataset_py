import pandas as pd

import xgboost as xgb
dataset = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
null_columns=dataset.columns[dataset.isnull().any()]

dataset[null_columns].isnull().sum()
dataset.head()
from sklearn.utils import resample



df_marjority = dataset[dataset.target==0]

df_minority = dataset[dataset.target==1]

 

# Upsample minority class

df_marjority_sampled = resample(df_marjority, 

                                 replace=True,     # sample with replacement

                                 n_samples=dataset[dataset['target']==1].shape[0])  # to match minority class

 

# Combine majority class with upsampled minority class

dataset_balanced = pd.concat([df_minority, df_marjority_sampled])

 

# Display new class counts

dataset_balanced.target.value_counts()
X = dataset_balanced.drop('target', axis=1).drop('ID_code', axis=1)

Y = dataset_balanced[['target']]
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV



# A parameter grid for XGBoost

params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }



xgb_model = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic', silent=True, nthread=1)
folds = 3

param_comb = 5



skf = StratifiedKFold(n_splits=folds, shuffle = True)



random_search = RandomizedSearchCV(xgb_model, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3)

random_search.fit(X, Y)
xgb_model_final = random_search.best_estimator_
print('\n All results:')

print(random_search.cv_results_)
columns = X.columns

validation=pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')

df_validation = validation[columns]

df_validation.head()
y_hat = xgb_model_final.predict(df_validation)

y_hat = pd.DataFrame(y_hat)
validation['target'] = y_hat
validation[['ID_code','target']].to_csv('submission_final.csv',index=False)