import pandas as pd 
import numpy as np
from collections import Counter

import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import xgboost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

OVERSAMPLE = False
PERFORM_PCA = False
all_cat_vars = ['Profession', 'Status', 'edu', 'Irregular', 'residence', 'prev_diagnosed', 'communication', 'Month', 'side_effects']
categorical_vars = ['Profession', 'Status', 'edu', 'communication', 'Month', 'side_effects']
continuous_vars = ['age', 'Money']

def load_data(csv_file, impute=False):
    df = pd.read_csv(csv_file)
    df = df.drop(columns=['ID'])

    # impute missing values
    if impute:
        for col in all_cat_vars:
            df[col] = df[col].fillna(df[col].value_counts().index[0])
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        df[continuous_vars] = imputer.fit_transform(df[continuous_vars].values)

    # encode categorical variables
    df = df.replace({'no': 0, 'yes': 1})
    categorical_vars = ['Profession', 'edu', 'Status', 'communication', 'Month', 'side_effects']
    df = pd.get_dummies(df, columns=categorical_vars)
    return df

train = load_data('../input/pasc-data-quest-20-20/doctor_train.csv', impute=True)
train.head()
# classes distribution
ratio = list(train['Y'].value_counts())
ratio
def split(df):
    y = df['Y']
    X = df[[x for x in df.columns if x != 'Y']]
    return X, y

X, y = split(train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print('train:', X_train.shape, y_train.shape)
print('test:', X_test.shape, y_test.shape)
# oversampling
if OVERSAMPLE:
    from imblearn.over_sampling import RandomOverSampler
    oversample = RandomOverSampler(sampling_strategy='minority', random_state=0)
    X_over, y_over = oversample.fit_resample(X, y)
if OVERSAMPLE:
    print(Counter(y_train))
    print(Counter(y_over))
# principal component analysis
if PERFORM_PCA:
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(.95)
    pca.fit(X_train)
    print(pca.n_components_)
if PERFORM_PCA:
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
# feature selection

dtrain = xgboost.DMatrix(X, y)

params = {
    'scale_pos_weight': max(ratio)/min(ratio),
    'n_estimators': 400
}
num_rounds = 5

bst = xgboost.train(params, dtrain, num_rounds)
ax = xgboost.plot_importance(bst, 
                             importance_type='gain', 
                             max_num_features=len(X_train.columns), 
                             xlabel='Gain')
fig = ax.figure
fig.set_size_inches(8, 8)
# Evaluation
def results(model):
    pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, pred))
    print('f1 score:', f1_score(y_test, pred))
    print('ROC-AUC:', roc_auc_score(y_test, pred))
    print('\nConfusion matrix:')
    return confusion_matrix(y_test, pred)
# LGBM

# Tuned LGBM model
# lgbm_model = LGBMClassifier(scale_pos_weight=10,
                            # n_estimators=140,
                            # learning_rate=0.17,
                            # max_depth=8,
                            # num_leaves=30,
                            # reg_alpha=0.2,
                            # reg_lambda=0.2,
                            # subsample=0.1,
                            # gamma=0
                            # )

# This configuration gives the best results on this dataset 
lgbm_model = LGBMClassifier(scale_pos_weight=10,
                            n_estimators=140
                            )

GRID_SEARCH = False
if GRID_SEARCH:
    param_dist = {
        'scale_pos_weight': [7, 7.1, 7.2, 7.3, max(ratio)/min(ratio), 8],
        }
    grid_search = GridSearchCV(lgbm_model, 
                            n_jobs=-1, 
                            param_grid=param_dist, 
                            cv = 3, 
                            scoring='f1', 
                            verbose=5)
    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_, grid_search.best_score_)
    results = pd.DataFrame(grid_search.cv_results_)
    results[['mean_test_score', 'std_test_score', 'params']]
lgbm_model.fit(X_train, y_train)
results(lgbm_model)
# XGBoost

# Tuned XGBoost model
# xgb_model = XGBClassifier(scale_pos_weight=max(ratio)/min(ratio),
                        #   n_estimators=439,
                        #   gamma=0.28,
                        #   reg_alpha=0.8,
                        #   reg_lambda=1.5,
                        #   learning_rate=0.117,
                        #   max_depth=5,
                        #   colsample_bytree=0.565,
                        #   subsample=0.8
                        #   )

# This configuration gives the best results on this dataset 
xgb_model = XGBClassifier(scale_pos_weight=max(ratio)/min(ratio),
                          n_estimators=439
                          )
GRID_SEARCH = False
if GRID_SEARCH:
    params = {
        'scale_pos_weight': [max(ratio)/min(ratio)+0.1, max(ratio)/min(ratio)+0.2, max(ratio)/min(ratio)+0.3]
        }

    grid_search = GridSearchCV(model, 
                            param_grid = params, 
                            scoring='f1',
                            n_jobs=-1,
                            cv=3,
                            verbose=5)
    grid_search.fit(X, y)
    print(grid_search.best_params_, grid_search.best_score_)
    results = pd.DataFrame(grid_search.cv_results_)
    results[['mean_test_score', 'std_test_score', 'params']]
xgb_model.fit(X_train, y_train)
results(xgb_model)
# combined xgb + lgbm
xgb_pred = xgb_model.predict(X_test)
lgbm_pred = lgbm_model.predict(X_test)
pred = []
for xgb, lgbm in zip(xgb_pred, lgbm_pred):
    pred.append(round(np.mean([xgb, lgbm])))

print('Accuracy:', accuracy_score(y_test, pred))
print('f1 score:', f1_score(y_test, pred))
print('ROC-AUC:', roc_auc_score(y_test, pred))
print('\nConfusion matrix:')
confusion_matrix(y_test, pred)
# comparing all models
def get_results(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return acc, f1, roc_auc

xgb_acc, xgb_f1, xgb_roc_auc = get_results(y_test, xgb_model.predict(X_test))
lgbm_acc, lgbm_f1, lgbm_roc_auc = get_results(y_test, lgbm_model.predict(X_test))

xgb_pred = xgb_model.predict(X_test)
lgbm_pred = lgbm_model.predict(X_test)
stacked_pred = []
for xgb, lgbm in zip(xgb_pred, lgbm_pred):
    stacked_pred.append(round(np.mean([xgb, lgbm])))
stacked_acc, stacked_f1, stacked_roc_auc = get_results(y_test, stacked_pred)

print('\t\t\tStacked\t\t\tXGBoost\t\t\tLGBM\n')
print('Accuracy\t', stacked_acc, '\t', xgb_acc, '\t', lgbm_acc)
print('F1 Score\t', stacked_f1, '\t', xgb_f1, '\t', lgbm_f1)
print('ROC AUC Score\t', stacked_roc_auc, '\t', xgb_roc_auc, '\t', lgbm_roc_auc)
# submission

def submit(model, output_file='submission'):
    test = load_data('../input/pasc-data-quest-20-20/doctor_test.csv')
    pred = model.predict(test.values)
    pred.tolist()

    submission = pd.DataFrame()
    submission['ID'] = range(0, len(test))

    submission['Y'] = pred
    submission['Y'] = submission['Y'].replace({0:'no', 1:'yes'})
    submission.to_csv(output_file + '.csv', index=False)

    print(submission.shape)
    print(output_file + '.csv saved!')
    return submission.head()

submit(lgbm_model, 'submission_lgbm')