import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
!ls ../input
train = pd.read_csv('../input/forest-cover-type-prediction/train.csv',
                   index_col='Id')
test = pd.read_csv('../input/forest-cover-type-prediction/test.csv',
                  index_col='Id')
train.head(1).T
train['Cover_Type'].value_counts()
def write_to_submission_file(predicted_labels, out_file,
                             target='Cover_Type', index_label="Id", init_index=15121):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(init_index, 
                                                  predicted_labels.shape[0] + init_index),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)
X_train, X_valid, y_train, y_valid = train_test_split(
    train.drop('Cover_Type', axis=1), train['Cover_Type'],
    test_size=0.3, random_state=17)
logit = LogisticRegression(C=1, solver='lbfgs', max_iter=500,
                           random_state=17, n_jobs=4,
                          multi_class='multinomial')
logit_pipe = Pipeline([('scaler', StandardScaler()), 
                       ('logit', logit)])
%%time
logit_pipe.fit(X_train, y_train)
logit_val_pred = logit_pipe.predict(X_valid)
accuracy_score(y_valid, logit_val_pred)
first_forest = RandomForestClassifier(
    n_estimators=100, random_state=17, n_jobs=4)
%%time
first_forest.fit(X_train, y_train)
forest_val_pred = first_forest.predict(X_valid)
accuracy_score(y_valid, forest_val_pred)
pd.DataFrame(first_forest.feature_importances_,
             index=X_train.columns, columns=['Importance']).sort_values(
    by='Importance', ascending=False)[:10]
lgb_clf = LGBMClassifier(random_state=17)
%%time
lgb_clf.fit(X_train, y_train)
accuracy_score(y_valid, lgb_clf.predict(X_valid))
param_grid = {'num_leaves': [7, 15, 31, 63], 
              'max_depth': [3, 4, 5, 6, -1]}
grid_searcher = GridSearchCV(estimator=lgb_clf, param_grid=param_grid, 
                             cv=5, verbose=1, n_jobs=4)
grid_searcher.fit(X_train, y_train)
grid_searcher.best_params_, grid_searcher.best_score_
accuracy_score(y_valid, grid_searcher.predict(X_valid))
num_iterations = 200
lgb_clf2 = LGBMClassifier(random_state=17, max_depth=-1, 
                          num_leaves=63, n_estimators=num_iterations,
                          n_jobs=1)

param_grid2 = {'learning_rate': np.logspace(-3, 0, 10)}
grid_searcher2 = GridSearchCV(estimator=lgb_clf2, param_grid=param_grid2,
                               cv=5, verbose=1, n_jobs=4)
grid_searcher2.fit(X_train, y_train)
print(grid_searcher2.best_params_, grid_searcher2.best_score_)
print(accuracy_score(y_valid, grid_searcher2.predict(X_valid)))
final_lgb = LGBMClassifier(n_estimators=200, num_leaves=63,
                           learning_rate=0.2, max_depth=-1,
                         n_jobs=4)
%%time
final_lgb.fit(train.drop('Cover_Type', axis=1), train['Cover_Type'])
%%time
lgb_final_pred = final_lgb.predict(test)
write_to_submission_file(lgb_final_pred, 
                         'lgb_forest_cover_type.csv')
pd.DataFrame(final_lgb.feature_importances_,
             index=X_train.columns, columns=['Importance']).sort_values(
    by='Importance', ascending=False)[:10]