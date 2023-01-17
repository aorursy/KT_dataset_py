import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.pipeline import Pipeline



# settings

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Calculating Precision, Recall and f1-score

def model_score(actual_value,predicted_values):

  from sklearn.metrics import confusion_matrix 

  from sklearn.metrics import accuracy_score 

  from sklearn.metrics import classification_report 

  from sklearn.metrics import recall_score

  from sklearn.metrics import precision_score

  

  actual = actual_value

  predicted = predicted_values

  results = confusion_matrix(actual, predicted) 

  

  print('Confusion Matrix :')

  print(results) 

  print('Accuracy Score :',accuracy_score(actual, predicted))

  print('Report : ')

  print(classification_report(actual, predicted))

  print('Recall Score : ')

  print(recall_score(actual, predicted))

  print('Precision Score : ')

  print(precision_score(actual, predicted))
df_diabetes = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

df_diabetes.head()
df_diabetes.describe()
# from pandas_profiling import ProfileReport

# report = ProfileReport(df_diabetes, title='Pandas Profiling Report')

# report
# python check if dataset is imbalanced : https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets



target_count = df_diabetes['Outcome'].value_counts()

print('Class 0 (No):', target_count[0])

print('Class 1 (Yes):', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



target_count.plot(kind='bar', title='Non-Diabetic Vs Diabetic')
from sklearn.model_selection import train_test_split



labels = df_diabetes['Outcome'].values

df_diabetes.drop(['Outcome'], axis=1, inplace=True)



X_train, X_test, y_train, y_test = train_test_split(df_diabetes, labels, test_size=0.15, shuffle=True)

# X_train, X_cv, y_train, y_cv = train_test_split(df_diabetes, labels, test_size=0.20, shuffle=True)

# X_cv, X_test, y_cv, y_test = train_test_split(X_cv, y_cv, test_size=0.20, shuffle=True)
print("Shape of train set : ",X_train.shape)

# print("Shape of cv set : ",X_cv.shape)

print("Shape of test set : ",X_test.shape)
from sklearn.feature_selection import SelectKBest, f_classif



# Preserving the list of original features

feature_cols = X_train.columns



# Keep top k features

selector = SelectKBest(f_classif, k=8)

X_train_new = selector.fit_transform(X_train,y_train)
# Get back the features we've kept, zero out all other features

selected_features = pd.DataFrame(selector.inverse_transform(X_train_new), index=X_train.index, columns=feature_cols)

selected_features.head()
# Dropped columns have values of all 0s, so var is 0, drop them

selected_columns = selected_features.columns[selected_features.var() != 0]



# Get the valid dataset with the selected features.

X_train = X_train[selected_columns]

# X_cv = X_cv[selected_columns]



X_test = X_test[selected_columns]
X_train.head()
%%time

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold



# Train multiple models : https://www.kaggle.com/tflare/testing-multiple-models-with-scikit-learn-0-79425

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

# from sklearn.linear_model import LogisticRegressionCV

from xgboost import XGBClassifier



from sklearn.model_selection import TimeSeriesSplit, cross_val_score



models = []



LogisticRegression = LogisticRegression(n_jobs=-1)

LinearSVC = LinearSVC()

KNeighbors = KNeighborsClassifier(n_jobs=-1)

DecisionTree = DecisionTreeClassifier()

RandomForest = RandomForestClassifier()

AdaBoost = AdaBoostClassifier()

Bagging = BaggingClassifier()

ExtraTrees = ExtraTreesClassifier()

GradientBoosting = GradientBoostingClassifier()

# LogisticRegressionCV = LogisticRegressionCV(n_jobs=-1)

XGBClassifier = XGBClassifier(nthread=-1)



models.append(("LogisticRegression",LogisticRegression))

models.append(("LinearSVC", LinearSVC))

models.append(("KNeighbors", KNeighbors))

models.append(("DecisionTree", DecisionTree))

models.append(("RandomForest", RandomForest))

models.append(("AdaBoost", AdaBoost))

models.append(("Bagging", Bagging))

models.append(("ExtraTrees", ExtraTrees))

models.append(("GradientBoosting", GradientBoosting))

# models.append(("LogisticRegressionCV", LogisticRegressionCV))

models.append(("XGBClassifier", XGBClassifier))



metric_names = ['f1', 'average_precision', 'accuracy', 'precision', 'recall']

results = []

names = []

skf = StratifiedKFold()

nested_dict = {}



for name,model in models:

  nested_dict[name] = {}

  for metric in metric_names:

    clf = make_pipeline(StandardScaler(), model)

    score = cross_val_score(clf, X_train, y_train, n_jobs=-1, scoring=metric, cv=skf)

    nested_dict[name][metric] = score.mean()
import json

print(json.dumps(nested_dict, sort_keys=True, indent=4))
from pprint import pprint



# Selected algorithm

algo_1 = LogisticRegression

algo_name_1 = "LogisticRegression"



algo_2 = GradientBoosting

algo_name_2 = "GradientBoosting"



# print("Predicting using : " + algo_name)

# clf = Pipeline([('scalar', StandardScaler()), ('clf', algo)])

# clf.fit(X_train, y_train)

# pprint(clf.get_params())
# Predicting on CV data

# predictions = clf.predict(X_cv)

# model_score(y_cv,predictions)
%%time

from sklearn.model_selection import GridSearchCV

# Parameters of pipelines can be set using ‘__’ separated parameter names:

param_grid = {

    'clf__penalty': ['l1', 'l2', 'elasticnet', 'none'],

    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],

    'clf__fit_intercept' : ['True', 'False'],

    'clf__solver': ['liblinear', 'sag', 'saga'],

    'clf__class_weight' : ['balanced', 'None'],

    'clf__warm_start' : ['True', 'False'],

}



clf = Pipeline([('scalar', StandardScaler()), ('clf', algo_1)])

search_1 = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=1, return_train_score=True)

search_1.fit(X_train, y_train)

print("Best parameter (CV score=%0.3f):" % search_1.best_score_)

print(search_1.best_params_)
%%time

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

# Parameters of pipelines can be set using ‘__’ separated parameter names:

param_grid = {

    "clf__loss":["deviance", "exponential"],

    "clf__learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],

    "clf__min_samples_split": np.linspace(0.1, 0.5, 12),

    "clf__min_samples_leaf": np.linspace(0.1, 0.5, 12),

    "clf__max_depth":[3,5,8,10,15],

    "clf__max_features":["log2","sqrt"],

    "clf__criterion": ["friedman_mse", "mae", "mse"],

    "clf__subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],

    "clf__n_estimators":[10, 50, 100, 200, 300, 400, 500],

    'clf__warm_start' : ['True', 'False'],

}



clf = Pipeline([('scalar', StandardScaler()), ('clf', algo_2)])

# search_1 = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=1, return_train_score=True)

search_2 = RandomizedSearchCV(clf, param_grid, n_jobs=-1, verbose=1, return_train_score=True, n_iter=10000)

search_2.fit(X_train, y_train)

print("Best parameter (CV score=%0.3f):" % search_2.best_score_)

print(search_2.best_params_)
trained_model_1 = search_1.best_estimator_

predictions_trained_model_test = trained_model_1.predict(X_test)
print("Predicting Using : " + algo_name_1)

model_score(y_test,predictions_trained_model_test)
trained_model_2 = search_2.best_estimator_

predictions_trained_model_test = trained_model_2.predict(X_test)
print("Predicting Using : " + algo_name_2)

model_score(y_test,predictions_trained_model_test)
# a = np.array(predictions_trained_model_test)

# b = np.array(y_test)

# accuracy = np.mean( a == b )

# print('Model success rate on test data : ' + str(accuracy))
import pickle

filename = 'trained_model_1.sav'

pickle.dump(trained_model_1, open(filename, 'wb'))



filename = 'trained_model_2.sav'

pickle.dump(trained_model_2, open(filename, 'wb'))



# Load model

# loaded_model = pickle.load(open(filename, 'rb'))

# result = loaded_model.score(X_test, y_test)

# print(result)