#ferramentas basicas

import pandas as pd

import matplotlib.pyplot as plt



#feature engineering

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold



#CART models 

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb



#validação

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score







dataset = pd.read_csv('../input/train.csv')
X = dataset.drop('ID', axis=1).drop('TARGET', axis=1)

y = dataset[['TARGET']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
rf_selection = RandomForestClassifier(random_state = 42)



folds = 3

param_comb = 5



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)



param_search = { 

    'n_estimators': [500, 1000],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8,10],

    'criterion' :['gini', 'entropy']

}



random_search_selection = RandomizedSearchCV(rf_selection, param_distributions=param_search, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y_train), verbose=3, random_state=42)

random_search_selection.fit(X_train, y_train)
clf = random_search_selection.best_estimator_

selector = clf.fit(X_train, y_train)
# plot most important features

feat_imp = pd.Series(clf.feature_importances_, index = X_train.columns.values).sort_values(ascending=False)

feat_imp[:40].plot(kind='bar', title='Features Relevance', figsize=(12, 8))

plt.ylabel('% of Importance')

plt.subplots_adjust(bottom=0.3)

plt.show()
# clf.feature_importances_ 

fs = SelectFromModel(selector, prefit=True)



X_train = fs.transform(X_train)

X_test = fs.transform(X_test)
# A parameter grid for XGBoost

params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

}



xgb = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',

                    silent=True, nthread=1)



folds = 3

param_comb = 5



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)



random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y_train), verbose=3, random_state=42 )



random_search.fit(X_train, y_train)
print('\n All results:')

print(random_search.cv_results_)

print('\n Best estimator:')

print(random_search.best_estimator_)

print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))

print(random_search.best_score_ * 2 - 1)

print('\n Best hyperparameters:')

print(random_search.best_params_)
#seleciona o melhor modelo

xgb_model = random_search.best_estimator_
xgboost_yhat = xgb_model.predict(X_test)

print("Roc AUC: ", roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1],

              average='macro'))

print(confusion_matrix(y_test,xgboost_yhat))

print(classification_report(y_test,xgboost_yhat))
!pip install kaggle

dataset_val = pd.read_csv('../input/test.csv')



#usa o ID como indice pra que ele nao entre como coluna da predicao

dataset_val.set_index('ID', inplace=True)



#transformando as features 

X_val = fs.transform(dataset_val.values)



#montando a predicao

y_hat_val = xgb_model.predict(X_val)



predicoes = pd.DataFrame(data=y_hat_val)



#retorna o ID como coluna

dataset_val.reset_index(level=0, inplace=True)



#junta as colunas ID com as de predicao

submissao = pd.concat([dataset_val['ID'], predicoes], axis=1)



#adequando ao formato de envio do Kaggle

submissao.rename(columns={0:'TARGET'}, 

                 inplace=True)

submissao.set_index('ID', inplace = True)

submissao.to_csv('submission.csv')



#!kaggle competitions submit santander-customer-satisfaction -f "submission.csv" -m "Santander Customer Submission from API"