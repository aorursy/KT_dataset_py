import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



# preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV



# models

from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 

from sklearn.ensemble import BaggingClassifier, VotingClassifier 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

import xgboost as xgb

from xgboost import XGBClassifier

import lightgbm as lgb

from lightgbm import LGBMClassifier



# NN models

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras import optimizers

from keras.wrappers.scikit_learn import KerasClassifier

from keras.callbacks import EarlyStopping, ModelCheckpoint



# model tuning

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval



# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)
traindf = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')

testdf = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')

submission = pd.read_csv('../input/titanic/gender_submission.csv')
#Thanks to:

# https://www.kaggle.com/mauricef/titanic

# https://www.kaggle.com/vbmokin/titanic-top-3-one-line-of-the-prediction-code

#

df = pd.concat([traindf, testdf], axis=0, sort=False)

df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))

df['LastName'] = df.Name.str.split(',').str[0]

family = df.groupby(df.LastName).Survived

df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())

df['WomanOrBoyCount'] = df.mask(df.IsWomanOrBoy, df.WomanOrBoyCount - 1, axis=0)

df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())

df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount - \

                                    df.Survived.fillna(0), axis=0)

df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)

df.WomanOrBoyCount = df.WomanOrBoyCount.replace(np.nan, 0)

df['Alone'] = (df.WomanOrBoyCount == 0)



#Thanks to https://www.kaggle.com/kpacocha/top-6-titanic-machine-learning-from-disaster

#"Title" improvement

df['Title'] = df['Title'].replace('Ms','Miss')

df['Title'] = df['Title'].replace('Mlle','Miss')

df['Title'] = df['Title'].replace('Mme','Mrs')

# Embarked

df['Embarked'] = df['Embarked'].fillna('S')

# Cabin, Deck

df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

df.loc[(df['Deck'] == 'T'), 'Deck'] = 'A'



# Thanks to https://www.kaggle.com/erinsweet/simpledetect

# Fare

med_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]

df['Fare'] = df['Fare'].fillna(med_fare)

#Age

df['Age'] = df.groupby(['Sex', 'Pclass', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))

# Family_Size

df['Family_Size'] = df['SibSp'] + df['Parch'] + 1



# Thanks to https://www.kaggle.com/vbmokin/titanic-top-3-cluster-analysis

cols_to_drop = ['Name','Ticket','Cabin']

df = df.drop(cols_to_drop, axis=1)



df.WomanOrBoySurvived = df.WomanOrBoySurvived.fillna(0)

df.WomanOrBoyCount = df.WomanOrBoyCount.fillna(0)

df.FamilySurvivedCount = df.FamilySurvivedCount.fillna(0)

df.Alone = df.Alone.fillna(0)
target = df.Survived.loc[traindf.index]

df = df.drop(['Survived'], axis=1)

train, test = df.loc[traindf.index], df.loc[testdf.index]
train.head(3)
test.head(3)
target[:3]
# Determination categorical features

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

categorical_columns = []

features = train.columns.values.tolist()

for col in features:

    if train[col].dtype in numerics: continue

    categorical_columns.append(col)

categorical_columns
# Encoding categorical features

for col in categorical_columns:

    if col in train.columns:

        le = LabelEncoder()

        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))

        train[col] = le.transform(list(train[col].astype(str).values))

        test[col] = le.transform(list(test[col].astype(str).values))   
train.info()
test.info()
#%% split training set to validation set

SEED = 100

Xtrain, Xval, Ztrain, Zval = train_test_split(train, target, test_size=0.3, random_state=SEED)
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(train, target)

Y_pred = logreg.predict(test).astype(int)

acc_log = round(logreg.score(train, target) * 100, 2)

acc_log
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_logreg.csv', index=False)

LB_log_all = 0.79904
coeff_df = pd.DataFrame(train.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines



svc = SVC()

svc.fit(train, target)

Y_pred = svc.predict(test).astype(int)

acc_svc = round(svc.score(train, target) * 100, 2)

acc_svc
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_svm.csv', index=False)

LB_svc_all = 0.62200
# Linear SVC



linear_svc = LinearSVC(dual=False)  # dual=False when n_samples > n_features.

linear_svc.fit(train, target)

Y_pred = linear_svc.predict(test).astype(int)

acc_linear_svc = round(linear_svc.score(train, target) * 100, 2)

acc_linear_svc
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_linear_svc.csv', index=False)

LB_linear_svc_all = 0.81339
# k-Nearest Neighbors algorithm



knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={'n_neighbors': [2, 3]}, cv=10).fit(train, target)

Y_pred = knn.predict(test).astype(int)

acc_knn = round(knn.score(train, target) * 100, 2)

print(acc_knn, knn.best_params_)
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_knn.csv', index=False)

LB_knn_all = 0.62679
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(train, target)

Y_pred = gaussian.predict(test).astype(int)

acc_gaussian = round(gaussian.score(train, target) * 100, 2)

acc_gaussian
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_GaussianNB.csv', index=False)

LB_gaussian_all = 0.73205
# Perceptron



perceptron = Perceptron()

perceptron.fit(train, target)

Y_pred = perceptron.predict(test).astype(int)

acc_perceptron = round(perceptron.score(train, target) * 100, 2)

acc_perceptron
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_perceptron.csv', index=False)

LB_perceptron_all = 0.46889
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(train, target)

Y_pred = sgd.predict(test).astype(int)

acc_sgd = round(sgd.score(train, target) * 100, 2)

acc_sgd
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_sgd.csv', index=False)

LB_sgd_all = 0.64593
# Decision Tree Classifier



decision_tree = DecisionTreeClassifier()

decision_tree.fit(train, target)

Y_pred = decision_tree.predict(test).astype(int)

acc_decision_tree = round(decision_tree.score(train, target) * 100, 2)

acc_decision_tree
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_decision_tree.csv', index=False)

LB_decision_tree_all = 0.77990
# Random Forest



random_forest = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [100, 300]}, cv=5).fit(train, target)

random_forest.fit(train, target)

Y_pred = random_forest.predict(test).astype(int)

random_forest.score(train, target)

acc_random_forest = round(random_forest.score(train, target) * 100, 2)

print(acc_random_forest,random_forest.best_params_)
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_random_forest.csv', index=False)

LB_random_forest_all = 0.81339
def hyperopt_xgb_score(params):

    clf = XGBClassifier(**params)

    current_score = cross_val_score(clf, train, target, cv=10).mean()

    print(current_score, params)

    return current_score 

 

space_xgb = {

            'learning_rate': hp.quniform('learning_rate', 0, 0.05, 0.0001),

            'n_estimators': hp.choice('n_estimators', range(100, 1000)),

            'eta': hp.quniform('eta', 0.025, 0.5, 0.005),

            'max_depth':  hp.choice('max_depth', np.arange(2, 12, dtype=int)),

            'min_child_weight': hp.quniform('min_child_weight', 1, 9, 0.025),

            'subsample': hp.quniform('subsample', 0.5, 1, 0.005),

            'gamma': hp.quniform('gamma', 0.5, 1, 0.005),

            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.005),

            'eval_metric': 'auc',

            'objective': 'binary:logistic',

            'booster': 'gbtree',

            'tree_method': 'exact',

            'silent': 1,

            'missing': None

        }

 

best = fmin(fn=hyperopt_xgb_score, space=space_xgb, algo=tpe.suggest, max_evals=10)

print('best:')

print(best)
params = space_eval(space_xgb, best)

params
XGB_Classifier = XGBClassifier(**params)

XGB_Classifier.fit(train, target)

Y_pred = XGB_Classifier.predict(test).astype(int)

XGB_Classifier.score(train, target)

acc_XGB_Classifier = round(XGB_Classifier.score(train, target) * 100, 2)

acc_XGB_Classifier
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_XGB_Classifier.csv', index=False)

LB_XGB_Classifier_all = 0.80861
fig =  plt.figure(figsize = (15,15))

axes = fig.add_subplot(111)

xgb.plot_importance(XGB_Classifier,ax = axes,height =0.5)

plt.show();

plt.close()
def hyperopt_lgb_score(params):

    clf = LGBMClassifier(**params)

    current_score = cross_val_score(clf, train, target, cv=10).mean()

    print(current_score, params)

    return current_score 

 

space_lgb = {

            'learning_rate': hp.quniform('learning_rate', 0, 0.05, 0.0001),

            'n_estimators': hp.choice('n_estimators', range(100, 1000)),

            'max_depth':  hp.choice('max_depth', np.arange(2, 12, dtype=int)),

            'num_leaves': hp.choice('num_leaves', 2*np.arange(2, 2**11, dtype=int)),

            'min_child_weight': hp.quniform('min_child_weight', 1, 9, 0.025),

            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.005),

            'objective': 'binary',

            'boosting_type': 'gbdt',

            }

 

best = fmin(fn=hyperopt_lgb_score, space=space_lgb, algo=tpe.suggest, max_evals=10)

print('best:')

print(best)
params = space_eval(space_lgb, best)

params
LGB_Classifier = LGBMClassifier(**params)

LGB_Classifier.fit(train, target)

Y_pred = LGB_Classifier.predict(test).astype(int)

LGB_Classifier.score(train, target)

acc_LGB_Classifier = round(LGB_Classifier.score(train, target) * 100, 2)

acc_LGB_Classifier
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_LGB_Classifier.csv', index=False)

LB_LGB_Classifier_all = 0.82296
fig =  plt.figure(figsize = (15,15))

axes = fig.add_subplot(111)

lgb.plot_importance(LGB_Classifier,ax = axes,height = 0.5)

plt.show();

plt.close()
def hyperopt_gb_score(params):

    clf = GradientBoostingClassifier(**params)

    current_score = cross_val_score(clf, train, target, cv=10).mean()

    print(current_score, params)

    return current_score 

 

space_gb = {

            'n_estimators': hp.choice('n_estimators', range(100, 1000)),

            'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int))            

        }

 

best = fmin(fn=hyperopt_gb_score, space=space_gb, algo=tpe.suggest, max_evals=10)

print('best:')

print(best)
params = space_eval(space_gb, best)

params
# Gradient Boosting Classifier



gradient_boosting = GradientBoostingClassifier(**params)

gradient_boosting.fit(train, target)

Y_pred = gradient_boosting.predict(test).astype(int)

gradient_boosting.score(train, target)

acc_gradient_boosting = round(gradient_boosting.score(train, target) * 100, 2)

acc_gradient_boosting
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_gradient_boosting.csv', index=False)

LB_GBC_all = 0.82296
# Ridge Classifier



ridge_classifier = RidgeClassifier()

ridge_classifier.fit(train, target)

Y_pred = ridge_classifier.predict(test).astype(int)

ridge_classifier.score(train, target)

acc_ridge_classifier = round(ridge_classifier.score(train, target) * 100, 2)

acc_ridge_classifier
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_ridge_classifier.csv', index=False)

LB_RidgeClassifier_all = 0.80861
# Bagging Classifier



bagging_classifier = BaggingClassifier()

bagging_classifier.fit(train, target)

Y_pred = bagging_classifier.predict(test).astype(int)

bagging_classifier.score(train, target)

acc_bagging_classifier = round(bagging_classifier.score(train, target) * 100, 2)

acc_bagging_classifier
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_bagging_classifier.csv', index=False)

LB_bagging_classifier_all = 0.80861
def hyperopt_etc_score(params):

    clf = ExtraTreesClassifier(**params)

    current_score = cross_val_score(clf, train, target, cv=10).mean()

    print(current_score, params)

    return current_score 

 

space_etc = {

            'n_estimators': hp.choice('n_estimators', range(100, 1000)),

            'max_features': hp.choice('max_features', np.arange(2, 17, dtype=int)),

            'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 5, dtype=int)),

            'max_depth':  hp.choice('max_depth', np.arange(2, 12, dtype=int)),

        }

 

best = fmin(fn=hyperopt_etc_score, space=space_etc, algo=tpe.suggest, max_evals=10)

print('best:')

print(best)
params = space_eval(space_etc, best)

params
# Extra Trees Classifier



extra_trees_classifier = ExtraTreesClassifier(**params)

extra_trees_classifier.fit(train, target)

Y_pred = extra_trees_classifier.predict(test).astype(int)

extra_trees_classifier.score(train, target)

acc_etc = round(extra_trees_classifier.score(train, target) * 100, 2)

acc_etc
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_etc.csv', index=False)

LB_ETC_all = 0.80861
def build_ann(optimizer='adam'):

    

    # Initializing the ANN

    ann = Sequential()

    

    # Adding the input layer and the first hidden layer of the ANN with dropout

    ann.add(Dense(units=32, kernel_initializer='glorot_uniform', activation='relu', input_shape=(16,)))

    

    # Add other layers, it is not necessary to pass the shape because there is a layer before

    ann.add(Dense(units=64, kernel_initializer='glorot_uniform', activation='relu'))

    ann.add(Dropout(rate=0.5))

    ann.add(Dense(units=64, kernel_initializer='glorot_uniform', activation='relu'))

    ann.add(Dropout(rate=0.5))

    

    # Adding the output layer

    ann.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

    

    # Compiling the ANN

    ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    

    return ann
opt = optimizers.Adam(lr=0.001)

ann = build_ann(opt)

# Training the ANN

history = ann.fit(Xtrain, Ztrain, batch_size=16, epochs=100, validation_data=(Xval, Zval))
# Predicting the Test set results

Y_pred = ann.predict(test)

Y_pred = (Y_pred > 0.5)*1 # convert probabilities to binary output
# Predicting the Train set results

ann_prediction = ann.predict(train)

ann_prediction = (ann_prediction > 0.5)*1 # convert probabilities to binary output



# Compute error between predicted data and true response and display it in confusion matrix

acc_ann1 = round(metrics.accuracy_score(target, ann_prediction) * 100, 2)

acc_ann1
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": np.reshape(Y_pred, len(Y_pred))})

#submission.to_csv('output/submission_ann1.csv', index=False)

LB_ann1_all = 0.59330
# Model

model = Sequential()

model.add(Dense(16, input_dim = train.shape[1], init = 'he_normal', activation = 'relu'))

model.add(Dropout(0.3))

model.add(Dense(64, init = 'he_normal', activation = 'relu'))

model.add(Dropout(0.3))

model.add(Dense(32, init = 'he_normal', activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_accuracy', patience=20, mode='max')

hist = model.fit(train, target, batch_size=64, validation_data=(Xval, Zval), 

               epochs=500, verbose=1, callbacks=[es])
plt.plot(hist.history['accuracy'], label='acc')

plt.plot(hist.history['val_accuracy'], label='val_acc')

# plt.plot(hist.history['acc'], label='acc')

# plt.plot(hist.history['val_acc'], label='val_acc')

plt.ylim((0, 1))

plt.legend()
# Predicting the Test set results

Y_pred = model.predict(test)

Y_pred = (Y_pred > 0.5)*1 # convert probabilities to binary output
# Predicting the Train set results

nn_prediction = model.predict(train)

nn_prediction = (nn_prediction > 0.5)*1 # convert probabilities to binary output



# Compute error between predicted data and true response

acc_ann2 = round(metrics.accuracy_score(target, nn_prediction) * 100, 2)

acc_ann2
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": np.reshape(Y_pred, len(Y_pred))})

#submission.to_csv('output/submission_ann2.csv', index=False)

LB_ann2_all = 0.64114
Voting_Classifier_hard = VotingClassifier(estimators=[('lr', logreg), ('rf', random_forest), ('gbc', gradient_boosting)], voting='hard')

for clf, label in zip([logreg, random_forest, gradient_boosting, Voting_Classifier_hard], 

                      ['Logistic Regression', 'Random Forest', 'Gradient Boosting Classifier', 'Ensemble']):

    scores = cross_val_score(clf, train, target, cv=10, scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
Voting_Classifier_hard.fit(train, target)

Y_pred = Voting_Classifier_hard.predict(test).astype(int)

Voting_Classifier_hard.score(train, target)

acc_VC_hard = round(Voting_Classifier_hard.score(train, target) * 100, 2)

acc_VC_hard
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_VC_hard.csv', index=False)

LB_VC_hard_all = 0.81339
eclf = VotingClassifier(estimators=[('lr', logreg), ('rf', random_forest), ('gbc', gradient_boosting)], voting='soft')

params = {'lr__C': [1.0, 100.0], 'gbc__learning_rate': [0.05, 1]}

Voting_Classifier_soft = GridSearchCV(estimator=eclf, param_grid=params, cv=5)

Voting_Classifier_soft.fit(train, target)

Y_pred = Voting_Classifier_soft.predict(test).astype(int)

Voting_Classifier_soft.score(train, target)

acc_VC_soft = round(Voting_Classifier_soft.score(train, target) * 100, 2)

acc_VC_soft
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_VC_soft.csv', index=False)

LB_VC_soft_all = 0.81339
Y_pred = (((test.WomanOrBoySurvived <= 0.238) & (test.Sex > 0.5) & (test.Alone > 0.5)) | \

          ((test.WomanOrBoySurvived > 0.238) & \

           ~((test.WomanOrBoySurvived > 0.55) & (test.WomanOrBoySurvived <= 0.633))))
simple_rule_model = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,

                       max_features=None, max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, presort=False,

                       random_state=1118, splitter='best') 

simple_rule_model.fit(train, target)

Y_pred = simple_rule_model.predict(test).astype(int)

simple_rule_model.score(train, target)

acc_simple_rule = round(simple_rule_model.score(train, target) * 100, 2)

acc_simple_rule
submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_simple_rule.csv', index=False)

LB_simple_rule_all = 0.83253
# Preparing datasets for only 3 features ('WomanOrBoySurvived', 'Sex', 'Alone')

cols_to_drop3 = ['SibSp', 'Parch', 'Fare', 'LastName', 'Deck',

               'Pclass', 'Age', 'Embarked', 'Title', 'IsWomanOrBoy',

               'WomanOrBoyCount', 'FamilySurvivedCount', 'Family_Size']

train = train.drop(cols_to_drop3, axis=1)

test = test.drop(cols_to_drop3, axis=1)

Xtrain, Xval, Ztrain, Zval = train_test_split(train, target, test_size=0.3, random_state=SEED)

train.info()
# 1. Logistic Regression



logreg = LogisticRegression()

logreg.fit(train, target)

Y_pred = logreg.predict(test).astype(int)

acc3_log = round(logreg.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_logreg3.csv', index=False)

LB_log = 0.77511
# 2. Support Vector Machines



svc = SVC()

svc.fit(train, target)

Y_pred = svc.predict(test).astype(int)

acc3_svc = round(svc.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_svm3.csv', index=False)

LB_svc = 0.82296
# 3. Linear SVC



linear_svc = LinearSVC(dual=False)

linear_svc.fit(train, target)

Y_pred = linear_svc.predict(test).astype(int)

acc3_linear_svc = round(linear_svc.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_linear_svc3.csv', index=False)

LB_linear_svc = 0.77511
# 4. k-Nearest Neighbors algorithm



knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={'n_neighbors': [2, 3]}, cv=10).fit(train, target)

Y_pred = knn.predict(test).astype(int)

acc3_knn = round(knn.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_knn3.csv', index=False)

LB_knn = 0.77990
# 5. Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(train, target)

Y_pred = gaussian.predict(test).astype(int)

acc3_gaussian = round(gaussian.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_GaussianNB3.csv', index=False)

LB_gaussian = 0.66507
# 6. Perceptron



perceptron = Perceptron()

perceptron.fit(train, target)

Y_pred = perceptron.predict(test).astype(int)

acc3_perceptron = round(perceptron.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_perceptron3.csv', index=False)

LB_perceptron = 0.77990
# 7. Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(train, target)

Y_pred = sgd.predict(test).astype(int)

acc3_sgd = round(sgd.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_sgd3.csv', index=False)

LB_sgd = 0.82775
# 8. Decision Tree Classifier



decision_tree = DecisionTreeClassifier()

decision_tree.fit(train, target)

Y_pred = decision_tree.predict(test).astype(int)

acc3_decision_tree = round(decision_tree.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_decision_tree3.csv', index=False)

LB_decision_tree = 0.83253
# 9. Random Forest



random_forest = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [100, 300]}, cv=5).fit(train, target)

random_forest.fit(train, target)

Y_pred = random_forest.predict(test).astype(int)

random_forest.score(train, target)

acc3_random_forest = round(random_forest.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_random_forest3.csv', index=False)

LB_random_forest = 0.83253
# 10. XGB_Classifier



def hyperopt_xgb_score(params):

    clf = XGBClassifier(**params)

    current_score = cross_val_score(clf, train, target, cv=10).mean()

    print(current_score, params)

    return current_score 

 

space_xgb = {

            'learning_rate': hp.quniform('learning_rate', 0, 0.05, 0.0001),

            'n_estimators': hp.choice('n_estimators', range(100, 1000)),

            'eta': hp.quniform('eta', 0.025, 0.5, 0.005),

            'max_depth':  hp.choice('max_depth', np.arange(2, 12, dtype=int)),

            'min_child_weight': hp.quniform('min_child_weight', 1, 9, 0.025),

            'subsample': hp.quniform('subsample', 0.5, 1, 0.005),

            'gamma': hp.quniform('gamma', 0.5, 1, 0.005),

            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.005),

            'eval_metric': 'auc',

            'objective': 'binary:logistic',

            'booster': 'gbtree',

            'tree_method': 'exact',

            'silent': 1,

            'missing': None

        }

 

best = fmin(fn=hyperopt_xgb_score, space=space_xgb, algo=tpe.suggest, max_evals=10)

params = space_eval(space_xgb, best)

XGB_Classifier = XGBClassifier(**params)

XGB_Classifier.fit(train, target)

Y_pred = XGB_Classifier.predict(test).astype(int)

XGB_Classifier.score(train, target)

acc3_XGB_Classifier = round(XGB_Classifier.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_XGB_Classifier3.csv', index=False)

LB_XGB_Classifier = 0.82296

#print(params)
# 11. LGBM_Classifier



def hyperopt_lgb_score(params):

    clf = LGBMClassifier(**params)

    current_score = cross_val_score(clf, train, target, cv=10).mean()

    print(current_score, params)

    return current_score 

 

space_lgb = {

            'learning_rate': hp.quniform('learning_rate', 0, 0.05, 0.0001),

            'n_estimators': hp.choice('n_estimators', range(100, 1000)),

            'max_depth':  hp.choice('max_depth', np.arange(2, 12, dtype=int)),

            'num_leaves': hp.choice('num_leaves', 2*np.arange(2, 2**11, dtype=int)),

            'min_child_weight': hp.quniform('min_child_weight', 1, 9, 0.025),

            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.005),

            'objective': 'binary',

            'boosting_type': 'gbdt',

            }

 

best = fmin(fn=hyperopt_lgb_score, space=space_lgb, algo=tpe.suggest, max_evals=10)

params = space_eval(space_lgb, best)

LGB_Classifier = LGBMClassifier(**params)

LGB_Classifier.fit(train, target)

Y_pred = LGB_Classifier.predict(test).astype(int)

LGB_Classifier.score(train, target)

acc3_LGB_Classifier = round(LGB_Classifier.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_LGB_Classifier3.csv', index=False)

LB_LGB_Classifier = 0.77990

#print(params)
# 12. GradientBoostingClassifier



def hyperopt_gb_score(params):

    clf = GradientBoostingClassifier(**params)

    current_score = cross_val_score(clf, train, target, cv=5).mean()

    print(current_score, params)

    return current_score 

 

space_gb = {

            'n_estimators': hp.choice('n_estimators', range(100, 1000)),

            'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int)),

            'max_features': None

        }

 

best = fmin(fn=hyperopt_gb_score, space=space_gb, algo=tpe.suggest, max_evals=5)

params = space_eval(space_gb, best)

gradient_boosting = GradientBoostingClassifier(**params)

gradient_boosting.fit(train, target)

Y_pred = gradient_boosting.predict(test).astype(int)

gradient_boosting.score(train, target)

acc3_gradient_boosting = round(gradient_boosting.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_gradient_boosting3.csv', index=False)

LB_GBC = 0.83253

#print(params)
# 13. Ridge Classifier



ridge_classifier = RidgeClassifier()

ridge_classifier.fit(train, target)

Y_pred = ridge_classifier.predict(test).astype(int)

ridge_classifier.score(train, target)

acc3_ridge_classifier = round(ridge_classifier.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_ridge_classifier3.csv', index=False)

LB_RidgeClassifier = 0.78468
# 14. Bagging Classifier



bagging_classifier = BaggingClassifier()

bagging_classifier.fit(train, target)

Y_pred = bagging_classifier.predict(test).astype(int)

bagging_classifier.score(train, target)

acc3_bagging_classifier = round(bagging_classifier.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_bagging_classifier3.csv', index=False)

LB_bagging_classifier = 0.83253
# 15. Extra Trees Classifier



def hyperopt_etc_score(params):

    clf = ExtraTreesClassifier(**params)

    current_score = cross_val_score(clf, train, target, cv=5).mean()

    print(current_score, params)

    return current_score 

 

space_etc = {

            'n_estimators': hp.choice('n_estimators', range(100, 1000)),

            'max_features': hp.choice('max_features', np.arange(2, 17, dtype=int)),

            'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 5, dtype=int)),

            'max_depth':  hp.choice('max_depth', np.arange(2, 10, dtype=int)),

            'max_features': None

        }

 

best = fmin(fn=hyperopt_etc_score, space=space_etc, algo=tpe.suggest, max_evals=5)

params = space_eval(space_etc, best)

extra_trees_classifier = ExtraTreesClassifier(**params)

extra_trees_classifier.fit(train, target)

Y_pred = extra_trees_classifier.predict(test).astype(int)

extra_trees_classifier.score(train, target)

acc3_etc = round(extra_trees_classifier.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_etc3.csv', index=False)

LB_ETC = 0.82296

#print(params)
# 16. Neural Network 1 



def build_ann(optimizer='adam'):

    

    # Initializing the ANN

    ann = Sequential()

    

    # Adding the input layer and the first hidden layer of the ANN with dropout

    ann.add(Dense(units=32, kernel_initializer='glorot_uniform', activation='relu', input_shape=(3,)))

    

    # Add other layers, it is not necessary to pass the shape because there is a layer before

    ann.add(Dense(units=64, kernel_initializer='glorot_uniform', activation='relu'))

    ann.add(Dropout(rate=0.5))

    ann.add(Dense(units=64, kernel_initializer='glorot_uniform', activation='relu'))

    ann.add(Dropout(rate=0.5))

    

    # Adding the output layer

    ann.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

    

    # Compiling the ANN

    ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    

    return ann

opt = optimizers.Adam(lr=0.001)

ann = build_ann(opt)

history = ann.fit(Xtrain, Ztrain, batch_size=16, epochs=100, validation_data=(Xval, Zval))

Y_pred = ann.predict(test)

Y_pred = (Y_pred > 0.5)*1 # convert probabilities to binary output

ann_prediction = ann.predict(train)

ann_prediction = (ann_prediction > 0.5)*1 # convert probabilities to binary output

acc3_ann1 = round(metrics.accuracy_score(target, ann_prediction) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": np.reshape(Y_pred, len(Y_pred))})

#submission.to_csv('output/submission_ann1_3.csv', index=False)

LB_ann1 = 0.82296
# 17. Neural Network 2



# Model

model = Sequential()

model.add(Dense(16, input_dim = train.shape[1], init = 'he_normal', activation = 'relu'))

model.add(Dropout(0.3))

model.add(Dense(64, init = 'he_normal', activation = 'relu'))

model.add(Dropout(0.3))

model.add(Dense(32, init = 'he_normal', activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', patience=20, mode='max')

hist = model.fit(train, target, batch_size=64, validation_data=(Xval, Zval), 

               epochs=500, verbose=1, callbacks=[es])

Y_pred = model.predict(test)

Y_pred = (Y_pred > 0.5)*1 # convert probabilities to binary output

nn_prediction = model.predict(train)

nn_prediction = (nn_prediction > 0.5)*1 # convert probabilities to binary output

acc3_ann2 = round(metrics.accuracy_score(target, nn_prediction) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": np.reshape(Y_pred, len(Y_pred))})

#submission.to_csv('output/submission_ann2_3.csv', index=False)

LB_ann2 = 0.77990
# 5.18 VotingClassifier (hard voting)



Voting_Classifier_hard = VotingClassifier(estimators=[('lr', logreg), ('rf', random_forest), ('gbc', gradient_boosting)], voting='hard')

for clf, label in zip([logreg, random_forest, gradient_boosting, Voting_Classifier_hard], 

                      ['Logistic Regression', 'Random Forest', 'Gradient Boosting Classifier', 'Ensemble']):

    scores = cross_val_score(clf, train, target, cv=10, scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

Voting_Classifier_hard.fit(train, target)

Y_pred = Voting_Classifier_hard.predict(test).astype(int)

Voting_Classifier_hard.score(train, target)

acc3_VC_hard = round(Voting_Classifier_hard.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_VC_hard3.csv', index=False)

LB_VC_hard = 0.83253
# 5.19 VotingClassifier (soft voting)



eclf = VotingClassifier(estimators=[('lr', logreg), ('rf', random_forest), ('gbc', gradient_boosting)], voting='soft')

params = {'lr__C': [1.0, 100.0], 'gbc__learning_rate': [0.05, 1]}

Voting_Classifier_soft = GridSearchCV(estimator=eclf, param_grid=params, cv=5)

Voting_Classifier_soft.fit(train, target)

Y_pred = Voting_Classifier_soft.predict(test).astype(int)

Voting_Classifier_soft.score(train, target)

acc3_VC_soft = round(Voting_Classifier_soft.score(train, target) * 100, 2)

submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})

#submission.to_csv('output/submission_VC_soft3.csv', index=False)

LB_VC_soft = 0.83253
# 5.20 The simple rule in one line

Y_pred = (((test.WomanOrBoySurvived <= 0.238) & (test.Sex > 0.5) & (test.Alone > 0.5)) | \

          ((test.WomanOrBoySurvived > 0.238) & \

           ~((test.WomanOrBoySurvived > 0.55) & (test.WomanOrBoySurvived <= 0.633))))

acc3_simple_rule = acc_simple_rule

LB_simple_rule = 0.83253
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 'k-Nearest Neighbors', 'Naive Bayes', 

              'Perceptron', 'Stochastic Gradient Decent', 

              'Decision Tree Classifier', 'Random Forest',  'XGBClassifier', 'LGBMClassifier',

              'GradientBoostingClassifier', 'RidgeClassifier', 'BaggingClassifier', 'ExtraTreesClassifier', 

              'Neural Network 1', 'Neural Network 2', 

              'VotingClassifier-hard voiting', 'VotingClassifier-soft voting',

              'Simple rule'],

    

    'Score16': [acc_log, acc_svc, acc_linear_svc, acc_knn, acc_gaussian, 

              acc_perceptron, acc_sgd, 

              acc_decision_tree, acc_random_forest, acc_XGB_Classifier, acc_LGB_Classifier,

              acc_gradient_boosting, acc_ridge_classifier, acc_bagging_classifier, acc_etc, 

              acc_ann1, acc_ann2, 

              acc_VC_hard, acc_VC_soft,

              acc_simple_rule],



    'Score3': [acc3_log, acc3_svc, acc3_linear_svc, acc3_knn, acc3_gaussian, 

              acc3_perceptron, acc3_sgd, 

              acc3_decision_tree, acc3_random_forest, acc3_XGB_Classifier, acc3_LGB_Classifier,

              acc3_gradient_boosting, acc3_ridge_classifier, acc3_bagging_classifier, acc3_etc, 

              acc3_ann1, acc3_ann2, 

              acc3_VC_hard, acc3_VC_soft,

              acc3_simple_rule],



    'LB_all': [LB_log_all, LB_svc_all, LB_linear_svc_all, LB_knn_all, LB_gaussian_all, 

              LB_perceptron_all, LB_sgd_all, 

              LB_decision_tree_all, LB_random_forest_all, LB_XGB_Classifier_all, LB_LGB_Classifier_all,

              LB_GBC_all, LB_RidgeClassifier_all, LB_bagging_classifier_all, LB_ETC_all, 

              LB_ann1_all, LB_ann2_all, 

              LB_VC_hard_all, LB_VC_soft_all,

              LB_simple_rule_all],

    

    'LB':    [LB_log, LB_svc, LB_linear_svc, LB_knn, LB_gaussian, 

              LB_perceptron, LB_sgd, 

              LB_decision_tree, LB_random_forest, LB_XGB_Classifier, LB_LGB_Classifier,

              LB_GBC, LB_RidgeClassifier, LB_bagging_classifier, LB_ETC, 

              LB_ann1, LB_ann2, 

              LB_VC_hard, LB_VC_soft,

              LB_simple_rule]})
models.sort_values(by=['Score16', 'LB_all', 'LB'], ascending=False)
models.sort_values(by=['Score3', 'LB_all', 'LB'], ascending=False)
models.sort_values(by=['LB_all', 'LB', 'Score3'], ascending=False)
models.sort_values(by=['LB', 'LB_all', 'Score3'], ascending=False)