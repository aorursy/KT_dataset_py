import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("bmh")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
df = pd.read_csv('../input/heart-disease-uci/heart.csv')
df.head()
df.describe()
# there are no missing values on the dataset
df.isnull().values.any()
df.shape
df.dtypes
# unique values on column 'cp'
set(df.cp)
# group by 'cp'
df.groupby('cp').count()
df.groupby('target').count()
corr = df.corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr, annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
sns.heatmap(corr, vmin = -1, vmax = 1, cmap = 'Greens')
# use the dataframe variable to create an array with the columns names 
all_vars = np.array(df.columns)
all_vars
# define features
features = np.array(all_vars[0:13])
features
# define target
target = np.array(all_vars[13])
target
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size = 0.2,
                                                      stratify = df[target], random_state = 0)
# defining variable to store the results
all_models = []
all_scores = []
from sklearn.svm import LinearSVC
def svm_test(X_train, y_train, cv = 10):
  np.random.seed(0)
  svc = LinearSVC(random_state=0)
  cv_scores = cross_val_score(svc, X_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)
  print('Average of', cv, 'tests: ', cv_scores.mean())
  return cv_scores.mean()
res = svm_test(X_train, y_train)
# updating results 
all_models = np.append(all_models, "SVC")
all_scores = np.append(all_scores, round(res, 4))
all_models, all_scores
from sklearn.ensemble import RandomForestClassifier
def rfc_test(X_train, y_train, n_estimators = 100, cv = 10):
  np.random.seed(0)
  rfc = RandomForestClassifier(n_estimators = n_estimators, random_state = 0, n_jobs = -1)
  cv_scores = cross_val_score(rfc, X_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)
  print('Average of', cv, 'tests: ', cv_scores.mean())
  return cv_scores.mean()
res = rfc_test(X_train, y_train)
# updating results 
all_models = np.append(all_models, "RFC")
all_scores = np.append(all_scores, round(res, 4))
from xgboost import XGBClassifier
def xgb_test(X_train, y_train, n_estimators = 100, cv = 10):
  np.random.seed(0)
  xgb = XGBClassifier(n_estimators = n_estimators, random_state = 0, n_jobs = -1)
  cv_scores = cross_val_score(xgb, X_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)
  print('Average of', cv, 'tests: ', cv_scores.mean())
  return cv_scores.mean()
res = xgb_test(X_train, y_train)
# updating results 
all_models = np.append(all_models, "XGB")
all_scores = np.append(all_scores, round(res, 4))
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
def mlp_test(X_train, y_train, cv = 10):
  np.random.seed(0)

  mlp = MLPClassifier(random_state=0)
  scaler = StandardScaler()
  pipe = Pipeline([('scaler', scaler), ('mlp', mlp)])

  cv_scores = cross_val_score(pipe, X_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)
  print('Average of', cv,  'tests: ', cv_scores.mean())
  return cv_scores.mean()
res = mlp_test(X_train, y_train)
# updating results 
all_models = np.append(all_models, "MLP")
all_scores = np.append(all_scores, round(res, 4))
# fitting/training only for all the features case, since it has proven to show better results
model = RandomForestClassifier(random_state = 0, n_jobs = -1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = metrics.accuracy_score(y_test, predictions)
print("Results for test data: Random Forest trained", round(score, 4))
# updating results 
all_models = np.append(all_models, "RFC trained")
all_scores = np.append(all_scores, score)
cm_rfc = confusion_matrix(predictions, y_test)
cm_rfc
model2 = XGBClassifier(random_state = 0, n_jobs = -1)
model2.fit(X_train, y_train)
predictions = model2.predict(X_test)
score = metrics.accuracy_score(y_test, predictions)
print("Results for test data: XGB trained", score)
# updating results 
all_models = np.append(all_models, "XGB trained")
all_scores = np.append(all_scores, round(score, 4))
from sklearn.model_selection import RandomizedSearchCV
# parameters dictionary for RFC
# check documentation for RFC

params_rfc = {
 "n_estimators"             : [50, 100, 150, 200],
 "min_samples_leaf"         : [1, 2, 3, 4, 5],
 "min_weight_fraction_leaf" : [0.00, 0.05, 0.10, 0.15, 0.20],
 "random_state"             : [0],
 "n_jobs"                   : [-1]
}
# parameters dictionary for XGB
# check documentation for XGB

params_xgb = {
 "n_estimators"     : [100, 150, 200, 250],
 "learning_rate"    : [0.50, 0.6, 0.7, 0.8, 0.9],
 "max_depth"        : [3, 5, 8, 10, 12],
 "gamma"            : [0.5, 0.7, 0.8, 0.9],
 "colsample_bytree" : [0.3, 0.5, 0.60, 0.80, 0.90, 0.95],
 "random_state"     : [0],
 "n_jobs"           : [-1]
}
# optimizing rfc (Random Forest Classifier)
random_search_rfc = RandomizedSearchCV(RandomForestClassifier(),
                                       param_distributions = params_rfc,
                                       scoring = 'accuracy',
                                       n_jobs = -1,
                                       random_state = 0,
                                       cv=10)

random_search_rfc.fit(X_train, y_train)
# Random Search score for the training data
random_search_rfc.score(X_train,y_train)
# optimized RFC model
random_search_rfc.best_estimator_
# optimized RFC parameters
random_search_rfc.best_params_
# average score of 3 folds for the best estimator
random_search_rfc.best_score_
# cv score for the optimized RFC model
opt_rfc = random_search_rfc.best_estimator_

score = cross_val_score(opt_rfc, X_train, y_train, cv = 10)
print("Cross Validation score for Optimized Random Forest", score.mean())
# updating results 
all_models = np.append(all_models, "RFC opt")
all_scores = np.append(all_scores, round(score.mean(), 4))
# predict on test data
predictions = opt_rfc.predict(X_test)

# evaluate results
score = metrics.accuracy_score(y_test, predictions)
print("Results for test data: Random Forest Optimized and trained", score)
# updating results 
all_models = np.append(all_models, "RFC opt (val acc)")
all_scores = np.append(all_scores, round(score, 4))
cm_rfc_opt = confusion_matrix(predictions, y_test)
cm_rfc_opt
# optimizing xgb (XGB Classifier)
random_search_xgb = RandomizedSearchCV(XGBClassifier(),
                                       param_distributions = params_xgb,
                                       scoring = 'accuracy',
                                       n_jobs = -1,
                                       random_state = 0,
                                       cv=10)

random_search_xgb.fit(X_train, y_train)
# Random Search score for the training data
random_search_xgb.score(X_train, y_train)
# optimized XGB model
random_search_xgb.best_estimator_
# optimized XGB parameters
random_search_xgb.best_params_
# average score of 3 folds for the best estimator
random_search_xgb.best_score_
# cv score for the optimized RFC model
opt_xgb = random_search_xgb.best_estimator_

score = cross_val_score(opt_xgb, X_train, y_train, cv = 10)
print("Cross Validation score for Optimized XGB", score.mean())
# updating results 
all_models = np.append(all_models, "XGB opt")
all_scores = np.append(all_scores, round(score.mean(), 4))
# predict on test data
predictions = opt_xgb.predict(X_test)

# evaluate results
score = metrics.accuracy_score(y_test, predictions)
print("Results for test data: XGB Optimized and trained", score)
# updating results 
all_models = np.append(all_models, "XGB opt (val acc)")
all_scores = np.append(all_scores, round(score, 4))
cm_xgb_opt = confusion_matrix(predictions, y_test)
cm_xgb_opt
all_models, all_scores
argsort = np.argsort(all_scores)
all_scores_sorted = all_scores[argsort]

all_models_names = all_models
all_models_sorted = all_models_names[argsort]

plt.figure(figsize=(10,6))
fig, ax = plt.subplots()
ax.barh(all_models_sorted, all_scores_sorted)
plt.xlim(0, 1)
plt.title("Heart disease prediction: Model vs Accuracy")
for index, value in enumerate(all_scores_sorted):
    plt.text(value, index, str(round(value, 4)), fontsize = 12)