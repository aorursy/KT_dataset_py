import warnings



warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np



df = pd.read_excel(

    'https://github.com/bukanyusufandroid/dataset/'

    'blob/master/XL%20Case%20Study/XL%20AA%20Interview'

    '%20-%20Data.xlsx?raw=true',

    sheet_name=0

)



df.head(3)
print(df.describe())

print(df.columns)
feature = ['Treatment', 'VAR_1', 'VAR_2',

       'VAR_3', 'VAR_4', 'VAR_5', 'VAR_6', 'VAR_7', 'VAR_8', 'VAR_9', 'VAR_10',

       'VAR_11', 'VAR_12', 'VAR_13', 'VAR_14', 'VAR_15', 'VAR_16', 'VAR_17',

       'VAR_18', 'VAR_19', 'VAR_20']



categorical = [ 'Response', 'Treatment', 'VAR_7', 'VAR_10', 'VAR_20' ]
import seaborn as sns

import matplotlib.pyplot as plt



pd.set_option('display.max_rows', None, 'display.max_columns', None)

sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))

plt.title('Distribution of Response')

sns.countplot(df['Treatment'])
plt.figure(figsize=(10, 6))

plt.title('Distribution of Response over Treatment')

sns.countplot(df['Treatment'], hue=df['Response'])
_tmp = sorted([i for i in df.columns if 'VAR' in i])
_tmp_ = _tmp[:10]



for i in ['Response', 'Treatment']:

    _tmp_.append(i)



data_ = pd.melt(df[_tmp_],

                id_vars = "Response",

                var_name="features",

                value_name="value"

              )



plt.subplots(figsize=(10,5))

sns.violinplot(x="features", y="value", hue="Response", 

              data=data_, split=True, inner="quart",

                palette='pastel')
_tmp_ = _tmp[10:]



for i in ['Response', 'Treatment']:

    _tmp_.append(i)



data_ = pd.melt(df[_tmp_],

                id_vars = "Response",

                var_name="features",

                value_name="value"

              )



plt.subplots(figsize=(10,5))

sns.violinplot(x="features", y="value", hue="Response", 

              data=data_, split=True, inner="quart",

                palette='pastel')
corr = df.corr()

unstack_corr = corr.unstack()
data_ = pd.melt(

    df[[i for i in df.columns if i not in ['Customer_No', 'Campaign_ID']]],

    id_vars = "Response",

    var_name="features",

    value_name="value"

)



plt.figure(figsize=(20,8))

sns.violinplot(x="features", y="value", hue="Response", 

              data=data_, split=True, inner="quart")

plt.xticks(rotation=90)

plt.show()
feature_ = ['VAR_8', 'VAR_3', 'VAR_14', 'VAR_13', 'VAR_7', 'Response']



data_ = pd.melt(

    df[feature_], id_vars = "Response",

    var_name="features",

    value_name="value"

)



plt.figure(figsize=(20,8))

sns.violinplot(x="features", y="value", hue="Response", 

              data=data_, split=True, inner="quart")

plt.xticks(rotation=90)

plt.show()
sns.countplot(unstack_corr[abs(unstack_corr) >= 0.75])
unstack_corr[abs(unstack_corr) >= 0.75]
from sklearn.model_selection import train_test_split

from random import randint

from imblearn.over_sampling import SMOTENC



random_state = 42



X_train, X_test, y_train, y_test = train_test_split(

    df[feature],

    df['Response'], 

    test_size=0.2,

    random_state=random_state

)



X_train.head(3)
smote_nc = SMOTENC(categorical_features=[0, 7, 10, 20], random_state=42)

X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(

    df[[i for i in feature_ if i != 'Response']],

    df['Response'], 

    test_size=0.2,

    random_state=42

)



X_train_red.head(3)
smote_nc = SMOTENC(categorical_features=[4], random_state=42)

X_resampled_red, y_resampled_red = smote_nc.fit_resample(X_train_red, y_train_red)
from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score



scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}



params_to_test = {

    'n_estimators':[i for i in range(1, 100, 25)],

    'max_depth':[i for i in range(1, 100, 25)],

    'min_samples_split':[i for i in np.arange(0.1, 1, 0.25)],

    'min_samples_leaf':[i for i in np.arange(0.1, 0.5, 0.25)],

    'max_features':[float(i) for i in np.arange(0.1, 1, 0.25)]    

}
from sklearn.ensemble import RandomForestClassifier



rf_model = RandomForestClassifier(n_jobs=-1, random_state=42)
import numpy as np

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



grid_search = GridSearchCV(

    rf_model,

    param_grid=params_to_test,

    cv=5,

    scoring=scoring,

    refit='AUC',

    n_jobs=-1

)



grid_search.fit(X_resampled, y_resampled)



best_params = grid_search.best_params_ 

best_model = RandomForestClassifier(**best_params)
best_model.get_params()
best_model.fit(X_resampled, y_resampled)

mean_accuracy = best_model.score(X_test, y_test)



print(

  "Mean accuracy of RF model auto-parameter with SMOTE and"

  "without feature reduction:",

  mean_accuracy

)
grid_search = GridSearchCV(

    rf_model,

    param_grid=params_to_test,

    cv=5,

    scoring=scoring,

    refit='AUC',

    n_jobs=-1

)



grid_search.fit(X_resampled_red, y_resampled_red)



best_params = grid_search.best_params_ 

best_model = RandomForestClassifier(**best_params)
best_model.get_params()
best_model.fit(X_resampled_red, y_resampled_red)

mean_accuracy = best_model.score(X_test_red, y_test_red)



print(

  "Mean accuracy of RF model auto-parameter with SMOTE and feature reduction:",

  mean_accuracy

)
grid_search = GridSearchCV(

    rf_model,

    param_grid=params_to_test,

    cv=5,

    scoring=scoring,

    refit='AUC',

    n_jobs=-1

)



grid_search.fit(X_train, y_train)



best_params = grid_search.best_params_ 

best_model = RandomForestClassifier(**best_params)
best_model.get_params()
best_model.fit(X_train, y_train)

mean_accuracy = best_model.score(X_test, y_test)



print(

  "Mean accuracy of RF model auto-parameter without SMOTE "

  "and feature reduction:",

  mean_accuracy

)
grid_search = GridSearchCV(

    rf_model,

    param_grid=params_to_test,

    cv=5,

    scoring=scoring,

    refit='AUC',

    n_jobs=-1

)



grid_search.fit(X_train_red, y_train_red)



best_params = grid_search.best_params_ 

best_model = RandomForestClassifier(**best_params)
best_model.get_params()
best_model.fit(X_train_red, y_train_red)

mean_accuracy = best_model.score(X_test_red, y_test_red)



print(

  "Mean accuracy of RF model auto-parameter without SMOTE"

  " but with reduction:",

  mean_accuracy

)
ws_wred = {

    'bootstrap': True,

    'class_weight': None,

    'criterion': 'gini',

    'max_depth': 1,

    'max_features': 0.1,

    'max_leaf_nodes': None,

    'min_impurity_decrease': 0.0,

    'min_impurity_split': None,

    'min_samples_leaf': 0.1,

    'min_samples_split': 0.1,

    'min_weight_fraction_leaf': 0.0,

    'n_estimators': 76,

    'n_jobs': None,

    'oob_score': False,

    'random_state': None,

    'verbose': 0,

    'warm_start': False

}



ws_wtred = {

    'bootstrap': True,

    'class_weight': None,

    'criterion': 'gini',

    'max_depth': 26,

    'max_features': 0.1,

    'max_leaf_nodes': None,

    'min_impurity_decrease': 0.0,

    'min_impurity_split': None,

    'min_samples_leaf': 0.1,

    'min_samples_split': 0.1,

    'min_weight_fraction_leaf': 0.0,

    'n_estimators': 76,

    'n_jobs': None,

    'oob_score': False,

    'random_state': None,

    'verbose': 0,

    'warm_start': False

}
ws_wred == ws_wtred
best_param = {

    'bootstrap': True,

    'class_weight': None,

    'criterion': 'gini',

    'max_depth': 1,

    'max_features': 0.1,

    'max_leaf_nodes': None,

    'min_impurity_decrease': 0.0,

    'min_impurity_split': None,

    'min_samples_leaf': 0.1,

    'min_samples_split': 0.1,

    'min_weight_fraction_leaf': 0.0,

    'n_estimators': 76,

    'n_jobs': None,

    'oob_score': False,

    'random_state': None,

    'verbose': 0,

    'warm_start': False

}
from sklearn.metrics import roc_curve, auc



max_depths = list(range(1, 20, 1))



train_results = []

test_results = []



for max_depth in max_depths:

  

   rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1, random_state=42)

   rf.fit(X_resampled_red, y_resampled_red)

   train_pred = rf.predict(X_resampled_red)

    

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_resampled_red, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   train_results.append(roc_auc)

   y_pred = rf.predict(X_test_red)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_red, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D



line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')

line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')



plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Tree depth')

plt.show()
max_features = list(range(1,df[feature_].shape[1]))



train_results = []

test_results = []



for max_feature in max_features:

  

   rf = RandomForestClassifier(max_features=max_feature)

   rf.fit(X_resampled_red, y_resampled_red)

   train_pred = rf.predict(X_resampled_red)

   

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_resampled_red, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   train_results.append(roc_auc)

   y_pred = rf.predict(X_test_red)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_red, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   test_results.append(roc_auc)
line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')

line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')



plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('max features')

plt.show()
!pip install forestci duecredit
import forestci as fci



responding = np.where(y_test_red == 1)[0]

not_responding = np.where(y_test_red == 0)[0]
rf_model = RandomForestClassifier(**best_param)
rf_model.fit(X_train_red, y_train_red)
prediction = rf_model.predict_proba(X_test_red)

probability = prediction[:,1]
fig, ax = plt.subplots(1)

ax.hist(prediction[responding, 1], histtype='step', label='Responding')

ax.hist(prediction[not_responding, 0], histtype='step', label='Not Responding')

ax.set_xlabel('Prediction (responding probability)')

ax.set_ylabel('Number of observations')

plt.legend()
rf_model = RandomForestClassifier(**best_param)

rf_model.get_params()
rf_model.fit(X_train_red, y_test_red)
test_data = pd.read_excel(

    'https://github.com/bukanyusufandroid/dataset/'

    'blob/master/XL%20Case%20Study/XL%20AA%20Interview'

    '%20-%20Data.xlsx?raw=true',

    sheet_name=1

)



test_data = test_data[[i for i in feature_ if i != 'Response']]
test_data.head(3)
pred = rf_model.predict(test_data)

pred_proba = rf_model.predict_proba(test_data)
test_data['Response'] = pred

test_data['Probability 0'] = pred_proba[:,0]

test_data['Probability 1'] = pred_proba[:,1]
test_data.head(3)