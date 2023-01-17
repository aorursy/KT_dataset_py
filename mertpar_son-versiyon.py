import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#kaggle çalıştırmak için : shift + enter
import numpy as np
import pandas as pd

#Görselleştirmek için
import matplotlib.pyplot as plt
%matplotlib inline

# Ön işleme
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pandas_profiling as pp

# Modeller
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

# Yapay Sinir Ağları Modelleri Kerase aittir.
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Modelleri ayarlamak için
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

# Hata filtrelemeleri için
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
#datayı kaggle'a yükleyip data sonra onun yolu ile importluyoruz. 
data = pd.read_csv("/kaggle/input/heart.csv")

# 3 adet datayı görüntüledik.
data.head(3)
print(data)
data.info()
pp.ProfileReport(data)

target_name = 'target'
data_target = data[target_name]
data = data.drop([target_name], axis=1)

train, test, target, target_test = train_test_split(data, data_target, test_size=0.2, random_state=0)

train.head(3)

test.head(3)    

train.info()

test.info()


Xtrain, Xval, Ztrain, Zval = train_test_split(train, target, test_size=0.2, random_state=0)

#Modeller ile Skor tahminleri bu alandan sonra başlıyor

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(train, target)
acc_log = round(logreg.score(train, target) * 100, 2)
print("Logistic Regression: " ,acc_log)

acc_test_log = round(logreg.score(test, target_test) * 100, 2)
print("Logistic Regression: ",acc_test_log)

coeff_df = pd.DataFrame(train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

# Support Vector Machines
svc = SVC()
svc.fit(train, target)

acc_svc = round(svc.score(train, target) * 100, 2)
print("Support Vector Machines:",acc_svc)

acc_test_svc = round(svc.score(test, target_test) * 100, 2)

print("Support Vector Machines:" ,acc_test_svc)

# Linear SVC

linear_svc = LinearSVC(dual=False)  # dual=False when n_samples > n_features.
linear_svc.fit(train, target)

acc_linear_svc = round(linear_svc.score(train, target) * 100, 2)
print("Linear SVC : ",acc_linear_svc)

acc_test_linear_svc = round(linear_svc.score(test, target_test) * 100, 2)
print("Linear SVC :",acc_test_linear_svc)

# k-Nearest Neighbors algorithm

knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={'n_neighbors': [2, 3]}, cv=10).fit(train, target)
acc_knn = round(knn.score(train, target) * 100, 2)
print(" k-Nearest Neighbors algorithm : ",acc_knn)

acc_test_knn = round(knn.score(test, target_test) * 100, 2)
print(" k-Nearest Neighbors algorithm: ",acc_test_knn)


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(train, target)

acc_gaussian = round(gaussian.score(train, target) * 100, 2)

print("Gaussian Naive Bayes:", acc_gaussian )
acc_test_gaussian = round(gaussian.score(test, target_test) * 100, 2)

print("Gaussian Naive Bayes:",acc_test_gaussian)


# Perceptron

perceptron = Perceptron()
perceptron.fit(train, target)
acc_perceptron = round(perceptron.score(train, target) * 100, 2)

print("Perceptron : ", acc_perceptron)

acc_test_perceptron = round(perceptron.score(test, target_test) * 100, 2)

print("Perceptron : ", acc_test_perceptron)


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(train, target)
acc_sgd = round(sgd.score(train, target) * 100, 2)
print("Stochastic Gradient Descent: ",acc_sgd)

acc_test_sgd = round(perceptron.score(test, target_test) * 100, 2)
print("Stochastic Gradient Descent: ",acc_test_sgd)

# Decision Tree Classifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(train, target)
acc_decision_tree = round(decision_tree.score(train, target) * 100, 2)
print("Decision Tree Classifier",acc_decision_tree)

acc_test_decision_tree = round(decision_tree.score(test, target_test) * 100, 2)
print("Decision Tree Classifier",acc_test_decision_tree)


# Random Forest

random_forest = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [100, 300]}, cv=5).fit(train, target)
random_forest.fit(train, target)
acc_random_forest = round(random_forest.score(train, target) * 100, 2)
print("Random Forest: ",acc_random_forest)

acc_test_random_forest = round(random_forest.score(test, target_test) * 100, 2)
print("Random Forest: ",acc_test_random_forest)

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
print('En iyi skor:')
print(best)

params = space_eval(space_xgb, best)
print(params)

XGB_Classifier = XGBClassifier(**params)
XGB_Classifier.fit(train, target)
acc_XGB_Classifier = round(XGB_Classifier.score(train, target) * 100, 2)
print("XGB Classifier : ",acc_XGB_Classifier)

acc_test_XGB_Classifier = round(XGB_Classifier.score(test, target_test) * 100, 2)
print("XGB Classifier : ",acc_test_XGB_Classifier)

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
print('En iyi skor:')
print(best)

params = space_eval(space_lgb, best)
print(params)


#LGB Classifier

LGB_Classifier = LGBMClassifier(**params)
LGB_Classifier.fit(train, target)
acc_LGB_Classifier = round(LGB_Classifier.score(train, target) * 100, 2)
print("LGB classifier : ",acc_LGB_Classifier)

acc_test_LGB_Classifier = round(LGB_Classifier.score(test, target_test) * 100, 2)
print("LGB classifier : ", acc_test_LGB_Classifier)

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
print('En iyi skor:')
print(best)

params = space_eval(space_gb, best)
print(params)

# Gradient Boosting Classifier

gradient_boosting = GradientBoostingClassifier(**params)
gradient_boosting.fit(train, target)
acc_gradient_boosting = round(gradient_boosting.score(train, target) * 100, 2)
print("Gradient Boosting Classifier : ",acc_gradient_boosting)

acc_test_gradient_boosting = round(gradient_boosting.score(test, target_test) * 100, 2)
print("Gradient Boosting Classifier : ",acc_test_gradient_boosting)

# Ridge Classifier

ridge_classifier = RidgeClassifier()
ridge_classifier.fit(train, target)
acc_ridge_classifier = round(ridge_classifier.score(train, target) * 100, 2)
print("Ridge Classifier : ",acc_ridge_classifier)

acc_test_ridge_classifier = round(ridge_classifier.score(test, target_test) * 100, 2)
print("Ridge Classifier : ",acc_test_ridge_classifier)

# Bagging Classifier

bagging_classifier = BaggingClassifier()
bagging_classifier.fit(train, target)
Y_pred = bagging_classifier.predict(test).astype(int)
acc_bagging_classifier = round(bagging_classifier.score(train, target) * 100, 2)
print("Bagging Classifier : ",acc_bagging_classifier)

acc_test_bagging_classifier = round(bagging_classifier.score(test, target_test) * 100, 2)
print("Bagging Classifier : ",acc_test_bagging_classifier)

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
            'max_features': None # for small number of features
        }
 
best = fmin(fn=hyperopt_etc_score, space=space_etc, algo=tpe.suggest, max_evals=10)
print('En iyi skor:')
print(best)

params = space_eval(space_etc, best)
print(params)

# Extra Trees Classifier

extra_trees_classifier = ExtraTreesClassifier(**params)
extra_trees_classifier.fit(train, target)
acc_etc = round(extra_trees_classifier.score(train, target) * 100, 2)
print("Extra Trees Classifier :",acc_etc)

acc_test_etc = round(extra_trees_classifier.score(test, target_test) * 100, 2)
print("Extra Trees Classifier :",acc_test_etc)


def build_ann(optimizer='adam'):
    
    # Initializing the ANN
    ann = Sequential()
    
    # Adding the input layer and the first hidden layer of the ANN with dropout
    ann.add(Dense(units=32, kernel_initializer='glorot_uniform', activation='relu', input_shape=(len(train.columns),)))
    
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

# ANN eğitimi
history = ann.fit(Xtrain, Ztrain, batch_size=16, epochs=100, validation_data=(Xval, Zval))

# Predicting Eğitim sonuçları
ann_prediction = ann.predict(train)
ann_prediction = (ann_prediction > 0.5)*1 # olasılıkları ikili çıktıya dönüştürme

# Tahmin edilen veriler ve gerçek yanıt arasındaki hatayı hesaplayın ve karışıklık matrisinde görüntüleme
acc_ann1 = round(metrics.accuracy_score(target, ann_prediction) * 100, 2)
print(acc_ann1)

# Predicting Test sonuçlaır
ann_prediction_test = ann.predict(test)
ann_prediction_test = (ann_prediction_test > 0.5)*1 # olasılıkları ikili çıktıya dönüştürme

# Tahmin edilen veriler ve gerçek yanıt arasındaki hatayı hesaplayın ve karışıklık matrisinde görüntüleme
acc_test_ann1 = round(metrics.accuracy_score(target_test, ann_prediction_test) * 100, 2)
print(acc_test_ann1)

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
hist = model.fit(train, target, batch_size=64, validation_data=(Xval, Zval), epochs=500, verbose=1, callbacks=[es])

plt.plot(hist.history['accuracy'], label='acc')
plt.plot(hist.history['val_accuracy'], label='val_acc')
plt.ylim((0, 1))
plt.legend()

# Predicting Eğitim sonuçları
nn_prediction = model.predict(train)
nn_prediction = (nn_prediction > 0.5)*1 # olasılıkları ikili çıktıya dönüştürme

# Compute error between predicted data and true response
acc_ann2 = round(metrics.accuracy_score(target, nn_prediction) * 100, 2)
print(acc_ann2)

# Predicting Test sonuçları
nn_prediction_test = model.predict(test)
nn_prediction_test = (nn_prediction_test > 0.5)*1 # convert probabilities to binary output

# Compute error between predicted data and true response
acc_test_ann2 = round(metrics.accuracy_score(target_test, nn_prediction_test) * 100, 2)
print(acc_test_ann2)

def hyperopt_ab_score(params):
    clf = AdaBoostClassifier(**params)
    current_score = cross_val_score(clf, train, target, cv=10).mean()
    print(current_score, params)
    return current_score 
 
space_ab = {
            'n_estimators': hp.choice('n_estimators', range(50, 1000)),
            'learning_rate': hp.quniform('learning_rate', 0, 0.05, 0.0001)       
        }
 
best = fmin(fn=hyperopt_ab_score, space=space_ab, algo=tpe.suggest, max_evals=10)
print('En iyi skor:')
print(best)

params = space_eval(space_ab, best)
print(params)

# AdaBoost Classifier

Ada_Boost = AdaBoostClassifier(**params)
Ada_Boost.fit(train, target)
Ada_Boost.score(train, target)
acc_AdaBoost = round(Ada_Boost.score(train, target) * 100, 2)
print("AdaBoost Classifier :",acc_AdaBoost)

acc_test_AdaBoost = round(Ada_Boost.score(test, target_test) * 100, 2)
print("AdaBoost Classifier :",acc_test_AdaBoost)

Voting_Classifier_hard = VotingClassifier(estimators=[('lr', logreg), ('rf', random_forest), ('ab', Ada_Boost)], voting='hard')
for clf, label in zip([logreg, random_forest, Ada_Boost, Voting_Classifier_hard], 
                      ['Logistic Regression', 'Random Forest', 'AdaBoost Classifier', 'Ensemble']):
    scores = cross_val_score(clf, train, target, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    
Voting_Classifier_hard.fit(train, target)
acc_VC_hard = round(Voting_Classifier_hard.score(train, target) * 100, 2)
print("Voting Classifier Hard : ",acc_VC_hard)

acc_test_VC_hard = round(Voting_Classifier_hard.score(test, target_test) * 100, 2)
print("Voting Classifier Hard : ",acc_test_VC_hard)

eclf = VotingClassifier(estimators=[('lr', logreg), ('rf', random_forest), ('ab', Ada_Boost)], voting='soft')
params = {'lr__C': [1.0, 100.0], 'ab__learning_rate': [0.0001, 1]}
Voting_Classifier_soft = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
Voting_Classifier_soft.fit(train, target)
acc_VC_soft = round(Voting_Classifier_soft.score(train, target) * 100, 2)
print("Voting Classifier Soft : ",acc_VC_soft)

acc_test_VC_soft = round(Voting_Classifier_soft.score(test, target_test) * 100, 2)
print("Voting Classifier Soft : ",acc_test_VC_soft)

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 'k-Nearest Neighbors', 'Naive Bayes', 
              'Perceptron', 'Stochastic Gradient Decent', 
              'Decision Tree Classifier', 'Random Forest',  'XGBClassifier', 'LGBMClassifier',
              'GradientBoostingClassifier', 'RidgeClassifier', 'BaggingClassifier', 'ExtraTreesClassifier', 
              'Neural Network 1', 'Neural Network 2', 
              'VotingClassifier-hard voiting', 'VotingClassifier-soft voting',
              'AdaBoostClassifier'],
    
    'Score_train': [acc_log, acc_svc, acc_linear_svc, acc_knn, acc_gaussian, 
              acc_perceptron, acc_sgd, 
              acc_decision_tree, acc_random_forest, acc_XGB_Classifier, acc_LGB_Classifier,
              acc_gradient_boosting, acc_ridge_classifier, acc_bagging_classifier, acc_etc, 
              acc_ann1, acc_ann2, 
              acc_VC_hard, acc_VC_soft,
              acc_AdaBoost],
    'Score_test': [acc_test_log, acc_test_svc, acc_test_linear_svc, acc_test_knn, acc_test_gaussian, 
              acc_test_perceptron, acc_test_sgd, 
              acc_test_decision_tree, acc_test_random_forest, acc_test_XGB_Classifier, acc_test_LGB_Classifier,
              acc_test_gradient_boosting, acc_test_ridge_classifier, acc_test_bagging_classifier, acc_test_etc, 
              acc_test_ann1, acc_test_ann2, 
              acc_test_VC_hard, acc_test_VC_soft,
              acc_test_AdaBoost]
                    })
# Tabloları oluşturuyoruz
models.sort_values(by=['Score_train', 'Score_test'], ascending=False)
print(models.sort_values(by=['Score_train', 'Score_test'], ascending=False))

models['Score_diff'] = abs(models['Score_train'] - models['Score_test'])
models.sort_values(by=['Score_diff'], ascending=True)
print(models.sort_values(by=['Score_diff'], ascending=True))

# Plot
plt.figure(figsize=[25,6])
xx = models['Model']
plt.tick_params(labelsize=14)
plt.plot(xx, models['Score_train'], label = 'Score_train')
plt.plot(xx, models['Score_test'], label = 'Score_test')
plt.legend()
plt.title('Score of 20 popular models for train and test datasets')
plt.xlabel('Models')
plt.ylabel('Score, %')
plt.xticks(xx, rotation='vertical')
plt.savefig('graph.png')
plt.show()
