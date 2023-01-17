#Xgboost_TypeA
xgb_model = xgb.XGBClassifier()

parameters = {
              #'objective'       : ['binary:logistic'],
              'eta'             : [0.05],
              'max_depth'       : [2, 3,4, 5,6],
              'n_estimators'    : [150],
              'early stopping'  : [3,5, 7, 10],
              'learning_rate'   : [0.02,0.03, 0.05,0.1],
              'random_state'    : [0]
              }


#GridSearch
xgb_clf = GridSearchCV(xgb_model, parameters, cv=5, n_jobs=4, verbose=0).fit(X_train, y_train)

print(xgb_clf.best_params_)
print(xgb_clf.best_estimator_)
print(xgb_clf.best_score_)
print(xgb_clf.score(X_test, y_test))
print('-' * 30)
y_pred = xgb_clf.predict(X)
print(classification_report(y, y_pred))
print('-' * 30)
print(confusion_matrix(y, y_pred))


#Repeat and split training data and test data for verification
gsXgb = cross_val_score(xgb_clf, X, y, scoring='accuracy', cv=5)

print(gsXgb)
print(gsXgb.mean())
#LightGBM

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {'task': 'train',
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'learning_rate': 0.02,
          'num_leaves': 23,
          'min_data_in_leaf': 1,
          'num_iteration': 300
          }

lgb_clf2 = lgb.train(params,
               lgb_train,
               valid_sets=(lgb_train, lgb_eval),
               num_boost_round=100,
               early_stopping_rounds=100,
               verbose_eval=50)


y_pred = lgb_clf2.predict(X_test, num_iteration=lgb_clf2.best_iteration)
y_pred = np.where(y_pred < 0.5, 0, 1)


accuracy = accuracy_score(y_test, y_pred)
print('accuracy:', accuracy_score(y_test, y_pred))


lgb.plot_importance(lgb_clf2, height=0.5, figsize=(8,16))
#LightGBM_TypeB

lgb_model = lgb.LGBMClassifier()

params = {
              'n_estimators'     : [30, 50, 70, 100 ],
              'num_leaves'       : [5, 10, 20],
              'max_depth'        : [2, 3, 5, 7],
              'learning_rate'    : [0.01, 0.1, 1],
              'early stopping'   : [5, 10, 50],
              'random_state'     : [0]
              }


#GridSearchCV
lgb_clf3 = GridSearchCV(lgb_model, params, cv=5, n_jobs=4, verbose=0).fit(X_train, y_train)

print(lgb_clf3.best_params_)
print(lgb_clf3.best_estimator_)
print(lgb_clf3.best_score_)
print(lgb_clf3.score(X_test, y_test))
print('-' * 30)
y_pred = lgb_clf3.predict(X)
print(classification_report(y, y_pred))
print('-' * 30)
print(confusion_matrix(y, y_pred))
#Repeat and split training data and test data for verification
gsLgb = cross_val_score(lgb_clf3, X, y, scoring='accuracy', cv=5)

print(gsLgb)
print(gsLgb.mean())
#LightGBM
prediction = lgb_clf3.predict(test)
prediction = np.where(prediction < 0.5, 0, 1)

submission = pd.DataFrame({
  'PassengerId' : IDtest,
  'Survived' : prediction
})
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)

score_list = []

params = {'task': 'train',
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'learning_rate': 0.02,
          'num_leaves': 23,
          'min_data_in_leaf': 1,
          'num_iteration': 300
          }

for fold_, (train_index, valid_index) in enumerate(kf.split(X_train,y_train)):

    print(f'fold{fold_ + 1} start')

    train_x = X_train.iloc[train_index]
    valid_x = X_train.iloc[valid_index]
    train_y = y_train.iloc[train_index]
    valid_y = y_train.iloc[valid_index]

    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_valid = lgb.Dataset(valid_x, valid_y)
    
    lgbm_params = {"objective":"binary"}

    gbm = lgb.train(params = params,
        train_set = lgb_train,
        valid_sets= [lgb_train, lgb_valid],            
        num_boost_round=100,
        early_stopping_rounds=20,
        verbose_eval=-1
    ) 
    
    oof = (gbm.predict(valid_x) > 0.5).astype(int)
    score_list.append(round(accuracy_score(valid_y, oof)*100,2))
    
    print(f'fold{fold_ + 1} end\n' ) 

print(score_list, 'score', round(np.mean(score_list), 2))