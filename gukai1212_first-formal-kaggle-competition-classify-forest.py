import pandas as pd

import numpy as np

from sklearn.metrics import accuracy_score, classification_report

from sklearn.model_selection import cross_val_score,cross_validate,train_test_split, GridSearchCV, StratifiedKFold

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier

from mlxtend.classifier import StackingCVClassifier
X = pd.read_csv("../input/learn-together/train.csv", index_col='Id')

X_test = pd.read_csv("../input/learn-together/test.csv", index_col='Id')

X_train = X.iloc[:,:54]

y_train = X.loc[:,'Cover_Type']
lgbc = LGBMClassifier(n_estimators=500, learning_rate= 0.1, 

               objective= 'multiclass', num_class=7,

               random_state= 12345, n_jobs=-1)

lgbc.fit(X_train, y_train)

lgbc_feature_importances = pd.DataFrame(lgbc.feature_importances_,

                                   index = X_train.columns,

                                    columns=["importance"])

print(lgbc_feature_importances.sort_values("importance",ascending=False))

print(X_train.columns[lgbc_feature_importances["importance"] == 0])
def get_LGBC():

    return LGBMClassifier(n_estimators=500, learning_rate= 0.1, 

               objective= 'multiclass', num_class=7,

               random_state= 12345, n_jobs=-1)





for thre in [0,50,100,200,500]:

    print(np.mean(cross_val_score(get_LGBC(), 

                                  X_train.drop(X_train.columns[lgbc.feature_importances_<thre], axis=1), 

                                  y_train, cv=5)))

X_train.drop(X_train.columns[lgbc.feature_importances_ == 0], axis=1, inplace=True)

X_test.drop(X_test.columns[lgbc.feature_importances_ == 0], axis=1, inplace=True)
fr = DecisionTreeClassifier(random_state=12345).fit(X_train, y_train)

fr_feature_importances = pd.DataFrame(fr.feature_importances_, 

                            index = X_train.columns,

                             columns=['importance'])

print(fr_feature_importances.sort_values("importance",ascending=False))
print(X_train.columns[fr_feature_importances["importance"] == 0])
for thre in [0,0.0001,0.001,0.005,0.01]:

    print(np.mean(cross_val_score(DecisionTreeClassifier(random_state=12345), 

                                    X_train.drop(X_train.columns[fr.feature_importances_<thre], axis=1), 

                                    y_train, cv=5)))
X_train.drop(X_train.columns[fr.feature_importances_ == 0], axis=1, inplace=True)

X_test.drop(X_test.columns[fr.feature_importances_ == 0], axis=1, inplace=True)
X_train.columns
len(X_train.columns)
ab_clf = AdaBoostClassifier(n_estimators=200,

                            base_estimator=DecisionTreeClassifier(

                                min_samples_leaf=2,

                                random_state=12345),

                            random_state=12345)

   

rf_clf = RandomForestClassifier(n_estimators=300,

                                random_state=12345,

                                n_jobs=1)



xgb_clf = XGBClassifier(n_estimators = 500, 

                        booster='gbtree', 

                        colsample_bylevel=1, 

                        colsample_bynode=1, 

                        colsample_bytree=0.8, 

                        gamma=5,

                        nthread=1, 

                        learning_rate=0.1,

                        max_delta_step=0, 

                        max_depth=10,

                        min_child_weight=10, 

                        missing=None, 

                        random_state= 12345,

                        n_jobs=1)                     



et_clf = ExtraTreesClassifier(n_estimators=300,

                              min_samples_leaf=1,

                              min_samples_split=2,

                              max_depth=50,

                              max_features=0.3,

                              bootstrap = False,

                              random_state=12345,

                              n_jobs=1)



lg_clf = LGBMClassifier(n_estimators=300,

                        num_leaves=128,

                        learning_rate= 0.1,

                        verbose=-1,

                        num_class=7,

                        random_state=12345,

                        n_jobs=1)



ensemble = [("AdaBoostClassifier", ab_clf),

            ("RandomForestClassifier", rf_clf),

            ("XGBClassifier", xgb_clf),

            ("ExtraTreesClassifier", et_clf),

            ("LGBMClassifier", lg_clf)]
for label, clf in ensemble:

    score = cross_val_score(clf, X_train, y_train,

                            cv=5,

                            scoring='accuracy',

                            verbose=0,

                            n_jobs=-1)



    print('  -- {: <24} : {:.3f} : {}'.format(label, np.mean(score), np.around(score, 3)))
stack = StackingCVClassifier(classifiers=[ab_clf, rf_clf, xgb_clf, et_clf, lg_clf],

                             meta_classifier=rf_clf,

                             cv=5,

                             stratify=True,

                             shuffle=True,

                             use_probas=True,

                             use_features_in_secondary=True,

                             verbose=1,

                             random_state=12345,

                             n_jobs=-1)

X_train = np.array(X_train)

y_train = np.array(y_train)

stack = stack.fit(X_train, y_train)
X_test = np.array(X_test)

pred = stack.predict(X_test)
pred[:10]
X_test = pd.read_csv("../input/learn-together/test.csv", index_col='Id')

predictions = pd.Series(pred, index=X_test.index, dtype=y_train.dtype)

predictions.to_csv('submission.csv', header=['Cover_Type'], index=True, index_label='Id')