# Import libraries necessary for this project

import numpy as np

import pandas as pd

from matplotlib import pyplot

import matplotlib.pyplot as plt

from tqdm import tqdm

from IPython.display import display # Allows the use of display() for DataFrames

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics.scorer import make_scorer, accuracy_score, recall_score, roc_auc_score, r2_score

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import OneClassSVM

from sklearn.ensemble import RandomForestClassifier





import xgboost as xgb

from xgboost import plot_tree

from xgboost import plot_importance



# Pretty display for notebooks

%matplotlib inline



random_seed = 42
# Load the trianing/Test dataset

try:

    train = pd.read_csv('../input/aps_failure_training_set_processed_8bit.csv')

    test = pd.read_csv('../input/aps_failure_test_set_processed_8bit.csv')

except:

    print("Dataset could not be loaded. Is the dataset missing?")
# Display a description of the dataset

display(train.shape)

display(test.shape)
train.head()
train['class'] = train['class'].apply(lambda x: 0 if x<=0 else 1)

test['class'] = test['class'].apply(lambda x: 0 if x<=0 else 1)
fig, ax = plt.subplots(1, 1, figsize=(20, 10))

train.boxplot(ax=ax)

plt.show()
train.describe()
X = train.drop('class', axis=1)

y = train['class']



# use the given test set, instead of creating from the training samples

X_test_given = test.drop('class', axis=1)

y_test_given = test['class']
y.value_counts()
1000/60000
y_test_given.value_counts()
fig, ax = plt.subplots(1, 1, figsize=(30, 30))

sns.heatmap(X.corr(), vmax=1, vmin=-1, center=0, annot=True, ax=ax)
scaler = MinMaxScaler().fit(X)

X_scaled = scaler.transform(X)
fig, ax = plt.subplots(1, 1, figsize=(20, 10))

pd.DataFrame(X_scaled).boxplot(ax=ax)

plt.show()
pca = PCA(0.98)

pca.fit(X)

pca.n_components_

X_reduced_data = pca.transform(X)

X_reduced_data.shape
plt.xlabel("Number of components")

plt.ylabel("Cumulated Sum of Ration of variance explained")

plt.xticks(range(0,87))

plt.grid(True)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.show()
plt.xlabel("Number of components")

plt.ylabel("Ration of variance explained")

plt.xticks(range(0,87))

plt.grid(True)

plt.plot(pca.explained_variance_ratio_)

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(30, 30))

sns.heatmap(pd.DataFrame(X_reduced_data).corr(), vmax=1, vmin=-1, center=0, annot=True, ax=ax)
fig, ax = plt.subplots(1, 1, figsize=(20, 10))

pd.DataFrame(X_reduced_data).boxplot(ax=ax)

plt.show()
%time

# fit the model

ocsvm = OneClassSVM(nu=0.5, kernel='rbf', gamma='auto')

ocsvm.fit(X_reduced_data)



y_pred = ocsvm.predict(X_reduced_data)
OUTLIER_DATA = -1

predicted_normal_index = np.where(y_pred != OUTLIER_DATA)

X_normal = X_reduced_data[predicted_normal_index]

fig, ax = plt.subplots(1, 1, figsize=(20, 10))

pd.DataFrame(X_normal).boxplot(ax=ax)

plt.show()
predicted_normal_index = np.where(y_pred != OUTLIER_DATA)

y_normal = y[predicted_normal_index[0]]

y_normal.value_counts()
OUTLIER_DATA = -1

predicted_anormaly_index = np.where(y_pred == OUTLIER_DATA)

X_anomaly = X_reduced_data[predicted_anormaly_index]

fig, ax = plt.subplots(1, 1, figsize=(20, 10))

pd.DataFrame(X_anomaly).boxplot(ax=ax)

plt.show()
predicted_anormaly_index = np.where(y_pred == OUTLIER_DATA)

y_anomaly  = y[predicted_anormaly_index[0]]

y_anomaly.value_counts()
X_reduced_test_data = pca.transform(X_test_given)



MIN = min(np.min(X_reduced_data), np.min(X_reduced_test_data)) - 0.1

MIN
X_reduced_test_data_log = np.log(X_reduced_test_data - MIN)



fig, ax = plt.subplots(1, 1, figsize=(20, 10))

pd.DataFrame(X_reduced_test_data_log).boxplot(ax=ax)

plt.show()
X_train = X

X_test = X_test_given



y_train = y

y_test = y_test_given
rfPredictor = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=42).fit(X,y)

predictRF = rfPredictor.predict(X_test_given)

score = rfPredictor.predict_proba(X)

print('Best ROC-AUC: {:.4f}'.format(roc_auc_score(y, score[:, 1], average='macro')))

print("accuracy score : {}".format(accuracy_score( y_test_given, predictRF)))

print("R-squared, coefficient of determination : {:.3f}".format(r2_score(y_test, predictRF)))

print(classification_report( y_true = y_test_given, y_pred = predictRF))

print(confusion_matrix(y_true = y_test_given, y_pred = predictRF))
%%time

predictor = xgb.XGBClassifier(seed=42)



predictor.fit(X_train, y_train)
score = predictor.predict_proba(X_train)

print('Best ROC-AUC: {:.4f}'.format(roc_auc_score(y_train, score[:, 1], average='macro')))

predict = predictor.predict(X_test)

print("accuracy score : {}".format(accuracy_score( y_test_given, predict)))

print("R-squared, coefficient of determination : {:.3f}".format(r2_score(y_test, predict)))

print(classification_report( y_true = y_test, y_pred = predict ))

confusion_matrix(y_true = y_test, y_pred = predict )
%%time

predictor = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_delta_step=5,

       max_depth=5, min_child_weight=1, missing=None, n_estimators=500,

       n_jobs=1, nthread=4, objective='binary:logistic', random_state=0,

       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42, silent=True,

       subsample=0.5, tree_method='gpu_exact', verbose=10)



predictor.fit(X_train, y_train)
score = predictor.predict_proba(X_train)

print('Best ROC-AUC: {:.4f}'.format(roc_auc_score(y_train, score[:, 1], average='macro')))

predict = predictor.predict(X_test)

print("accuracy score : {}".format(accuracy_score( y_test_given, predict)))

print("R-squared, coefficient of determination : {:.3f}".format(r2_score(y_test, predict)))

print(classification_report( y_true = y_test, y_pred = predict ))

confusion_matrix(y_true = y_test, y_pred = predict )
fig, ax = plt.subplots(1, 1, figsize=(7, 25))

plot_importance(predictor, max_num_features = pca.n_components_, ax=ax)

plt.show()
plot_tree(predictor, rankdir='LR')

fig = plt.gcf()

fig.set_size_inches(150, 100)

plt.show()
X_train = X_reduced_data

X_test = X_reduced_test_data



y_train = y

y_test = y_test_given
%%time

predictor = xgb.XGBClassifier(seed=42)



predictor.fit(X_train, y_train)
score = predictor.predict_proba(X_train)

print('Best ROC-AUC: {:.4f}'.format(roc_auc_score(y_train, score[:, 1], average='macro')))

predict = predictor.predict(X_test)

print("accuracy score : {}".format(accuracy_score( y_test_given, predict)))

print("R-squared, coefficient of determination : {:.3f}".format(r2_score(y_test, predict)))

print(classification_report( y_true = y_test, y_pred = predict ))

confusion_matrix(y_true = y_test, y_pred = predict )
%%time

predictor = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_delta_step=5,

       max_depth=5, min_child_weight=1, missing=None, n_estimators=500,

       n_jobs=1, nthread=4, objective='binary:logistic', random_state=0,

       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42, silent=True,

       subsample=0.5, tree_method='gpu_exact', verbose=10)



predictor.fit(X_train, y_train)
score = predictor.predict_proba(X_train)

print('Best ROC-AUC: {:.4f}'.format(roc_auc_score(y_train, score[:, 1], average='macro')))

predict = predictor.predict(X_test)

print("accuracy score : {}".format(accuracy_score( y_test_given, predict)))

print("R-squared, coefficient of determination : {:.3f}".format(r2_score(y_test, predict)))

print(classification_report( y_true = y_test, y_pred = predict ))

confusion_matrix(y_true = y_test, y_pred = predict )
fig, ax = plt.subplots(1, 1, figsize=(7, 25))

plot_importance(predictor, max_num_features = pca.n_components_, ax=ax)

plt.show()
plot_tree(predictor, rankdir='LR')

fig = plt.gcf()

fig.set_size_inches(150, 100)

plt.show()
MIN = min(np.min(X_reduced_data), np.min(X_reduced_test_data)) - 0.1



X_train = np.log(X_reduced_data - MIN)

X_test = np.log(X_reduced_test_data - MIN)



y_train = y

y_test = y_test_given
%%time

predictor = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_delta_step=5,

       max_depth=5, min_child_weight=1, missing=None, n_estimators=500,

       n_jobs=1, nthread=4, objective='binary:logistic', random_state=0,

       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42, silent=True,

       subsample=0.5, tree_method='gpu_exact', verbose=10)



predictor.fit(X_train, y_train)
score = predictor.predict_proba(X_train)

print('Best ROC-AUC: {:.4f}'.format(roc_auc_score(y_train, score[:, 1], average='macro')))

predict = predictor.predict(X_test)

print("accuracy score : {}".format(accuracy_score( y_test_given, predict)))

print("R-squared, coefficient of determination : {:.3f}".format(r2_score(y_test, predict)))

print(classification_report( y_true = y_test, y_pred = predict ))

confusion_matrix(y_true = y_test, y_pred = predict )
fig, ax = plt.subplots(1, 1, figsize=(7, 25))

plot_importance(predictor, max_num_features = pca.n_components_, ax=ax)

plt.show()
plot_tree(predictor, rankdir='LR')

fig = plt.gcf()

fig.set_size_inches(150, 100)

plt.show()
df_eval = pd.DataFrame([{"model":"RandomForest", "ROC-AUC": 0.9885, "accuracy score" : 0.9858125, "R-squared" : 0.380},

{"model":"XGBoost", "ROC-AUC": 0.9944, "accuracy score" : 0.98975, "R-squared" : 0.552},

{"model":"XGBoost  PCA", "ROC-AUC": 1.0000, "accuracy score" : 0.9889375, "R-squared" : 0.517},

{"model":"XGBoost  Log Transform", "ROC-AUC": 1.0000, "accuracy score" : 0.9884375, "R-squared" : 0.495}]).set_index('model', drop=True)

df_eval.plot(kind="bar")
X_train = X

X_test = X_test_given



y_train = y

y_test = y_test_given
%%time



params = {

'max_depth':[5,6,7],

'learning_rate':[0.1],

'gamma':[0.0],

'min_child_weight':[1],

'max_delta_step':[5],

'colsample_bytree':[0.8],

'n_estimators':[300, 500, 700],

'subsample':[0.5],

'objective':['binary:logistic'],

'nthread':[4],

'scale_pos_weight':[1],

'seed':[random_seed],

'verbose': [10],

'tree_method':['gpu_exact']}





model = xgb.XGBClassifier(tree_method='hist')

#cv = GridSearchCV(model, params, cv=5, n_jobs=4, scoring='roc_auc')

cv = GridSearchCV(model, params, cv=5, n_jobs=4, scoring='recall')



cv.fit(X_train, y_train)

print(cv.best_estimator_)
predictor = cv.best_estimator_

predictor.save_model('./model/xgb.model')
score = predictor.predict_proba(X_train)

print('Best ROC-AUC: {:.4f}'.format(roc_auc_score(y_train, score[:, 1], average='macro')))

predict = predictor.predict(X_test)

print("accuracy score : {}".format(accuracy_score( y_test, predict)))

print("R-squared, coefficient of determination : {:.3f}".format(r2_score(y_test, predict)))

print(classification_report( y_true = y_test, y_pred = predict ))

confusion_matrix(y_true = y_test, y_pred = predict )
print("Best parameters: %s" % cv.best_params_)

print("Best auroc score: %s" % cv.best_score_)
fig, ax = plt.subplots(1, 1, figsize=(7, 25))

plot_importance(predictor, max_num_features = pca.n_components_, ax=ax)

plt.show()
plot_tree(predictor, rankdir='LR')

fig = plt.gcf()

fig.set_size_inches(300, 200)

plt.show()
df_eval = pd.DataFrame([{"model":"Tuned XGBoost", "ROC-AUC": 1.0000, "accuracy score" : 0.99275, "R-squared" : 0.683},

                        {"model":"XGBoost(Default)", "ROC-AUC": 0.9944, "accuracy score" : 0.98975, "R-squared" : 0.552}]).set_index('model', drop=True)

df_eval.plot(kind="bar")
df_eval = pd.DataFrame([{"model":"Tuned XGBoost", "ROC-AUC": 1.0000, "accuracy score" : 0.99275, "R-squared" : 0.683},

                        {"model":"XGBoost(Default)", "ROC-AUC": 0.9944, "accuracy score" : 0.98975, "R-squared" : 0.552},

                        {"model":"RandomForest", "ROC-AUC": 0.9885, "accuracy score" : 0.9858125, "R-squared" : 0.380}]).set_index('model', drop=True)

df_eval.plot(kind="bar")