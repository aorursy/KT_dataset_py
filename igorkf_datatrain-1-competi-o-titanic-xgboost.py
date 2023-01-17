import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder, scale

import xgboost as xgb

from sklearn.metrics import mean_squared_error, auc, roc_curve, roc_auc_score

from sklearn.model_selection import train_test_split
# load the dataset

train = pd.read_csv('/kaggle/input/data-train-competicao-ml-1-titanic/train.csv')



# create labelencoder

le = LabelEncoder()



train = train.drop(['Name', 'Cabin'], axis=1).fillna(train['Age'].mean())

cat_columns = train.select_dtypes('object').columns

train[cat_columns] = train[cat_columns].apply(lambda col: le.fit_transform(col.astype(str)))

train.describe()
# separate X and Y

X = train.drop(['PassengerId', 'Ticket', 'Survived'], axis=1)

y = train[['Survived']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)
plt.bar(x=[0, 1], height=y.groupby('Survived').size(), color=['red', 'green'])

plt.xticks([0, 1], ['Died', 'Survied']);
plt.hist(X['Age']);
# convert to DMatrix

dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)
# params to tune

params = {

    'max_depth': 6,

    'min_child_weight': 1,

    'eta': .3,

    'subsample': 1,

    'colsample_bytree': 1,

    # Other parameters

    'objective':'binary:logistic',

}



params['eval_metric'] = 'auc'



num_boost_round = 999
model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, 'Test')],

    early_stopping_rounds=10

)
cv_results = xgb.cv(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    seed=42,

    nfold=5,

    metrics={'auc'},

    early_stopping_rounds=10

)



cv_results
cv_results['test-auc-mean'].max()
gridsearch_params = [

    (max_depth, min_child_weight)

    for max_depth in range(1, 10)

    for min_child_weight in range(1, 10)

]
# Define initial best params

max_auc = float(0)

best_params = None



for max_depth, min_child_weight in gridsearch_params:

    print(f'CV with max_depth = {max_depth}, min_child_weight = {min_child_weight}')   # Update our parameters

    params['max_depth'] = max_depth

    params['min_child_weight'] = min_child_weight 

          

    # run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        nfold=5,

        metrics={'auc'},

        early_stopping_rounds=10

    )

          

    # update best AUC model    

    mean_auc = cv_results['test-auc-mean'].max()

    boost_rounds = cv_results['test-auc-mean'].argmax()

    print(f'\tAUC {mean_auc} for {boost_rounds} rounds')

    if mean_auc > max_auc:

        max_auc = mean_auc

        best_params = (max_depth, min_child_weight)

        print(f'Best params: {best_params[0]}, {best_params[1]}, AUC: {max_auc}')



print('\nBest params:', best_params)
params['max_depth'], params['min_child_weight'] = best_params

params
model2 = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, 'Test')],

    early_stopping_rounds=10

)
gridsearch_params = [

    (subsample, colsample)

    for subsample in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for colsample in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

]
# Define initial best params

max_auc = float(0)

best_params = None



# start reversed (max -> min)

for subsample, colsample in reversed(gridsearch_params):

    print(f'CV with subsample = {subsample}, colsample = {colsample}')   # Update our parameters

    params['subsample'] = subsample

    params['colsample_bytree'] = colsample 

          

    # run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        nfold=5,

        metrics={'auc'},

        early_stopping_rounds=10

    )

          

    # update best AUC model    

    mean_auc = cv_results['test-auc-mean'].max()

    boost_rounds = cv_results['test-auc-mean'].argmax()

    print(f'\tAUC {mean_auc} for {boost_rounds} rounds')

    if mean_auc > max_auc:

        max_auc = mean_auc

        best_params = (subsample, colsample)

        print(f'Best params: {best_params[0]}, {best_params[1]}, AUC: {max_auc}')



print('\nBest params:', best_params)
params['subsample'], params['colsample_bytree'] = best_params

params
model3 = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, 'Test')],

    early_stopping_rounds=10

)
max_auc = float(0)

best_params = None



# start reversed (max -> min)

for eta in [.31, .309, .308, .307, .306, .305, .304, .303, .302, .301, .3, .299]:

    print(f'CV with eta = {eta}')   # Update our parameters

    params['eta'] = eta

          

    # run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        nfold=5,

        metrics={'auc'},

        early_stopping_rounds=10

    )

          

    # update best AUC model    

    mean_auc = cv_results['test-auc-mean'].max()

    boost_rounds = cv_results['test-auc-mean'].argmax()

    print(f'\tAUC {mean_auc} for {boost_rounds} rounds')

    if mean_auc > max_auc:

        max_auc = mean_auc

        best_params = (eta)

        print(f'Best params: {best_params}, AUC: {max_auc}')



print('\nBest params:', best_params)
params['eta'] = best_params

params
model4 = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, 'Test')],

    early_stopping_rounds=10

)
num_boost_round = model4.best_iteration + 1



best_model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, 'Test')]

)
# predict

y_pred = best_model.predict(dtest)

y_pred = y_pred.reshape((y_pred.shape[0], 1))



auc = roc_auc_score(y_test, y_pred)

auc
params
# load test for submission

test = pd.read_csv('/kaggle/input/data-train-competicao-ml-1-titanic/test.csv')
plt.hist(test['Age']);
# id

id_passenger = test[['PassengerId']]



# use only train features

test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

test['Age'] = test['Age'].fillna(test['Age'].mean())



cat_columns = test.select_dtypes('object').columns

test[cat_columns] = test[cat_columns].apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))

test
plt.hist(test['Age']);
# generate predictions from submit data

dtest_submit = xgb.DMatrix(test)

pred_submit = best_model.predict(dtest_submit)



df_submit = pd.concat([id_passenger, pd.DataFrame(pred_submit, columns=['Survived'])], axis=1)

df_submit
plt.hist(df_submit['Survived']);
# df_submit.to_csv(f'auc{auc}.csv', index=False)