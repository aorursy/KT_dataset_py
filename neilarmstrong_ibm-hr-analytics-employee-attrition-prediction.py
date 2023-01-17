# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install feature-engine
!pip install pydotplus
!pip install eli5
!pip install shap
import os

import joblib

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import pydotplus



%matplotlib inline

np.random.seed(0)



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



# Revise according to your data directory

PATH = '/kaggle/input/ibm-hr-data/'

FILE = 'IBM_HR_Data.csv'
df = pd.read_csv(filepath_or_buffer=os.path.join(PATH, FILE))



# We are dropping the following columns since they are not features, just ID nos and so on. Plus, 'StandardHours' contains no variation

df.drop(labels=['EmployeeCount', 'EmployeeNumber', 'ApplicationID', 'Over18', 'StandardHours'], axis='columns', inplace=True)



# Overview of dtypes and Missing Values

def observe_data(df):

    '''

    Presents exploratory data summary in a crisp manner; 

    with dtype, null values, total values and feature summary columns.

    '''

    df = df.copy()

    properties = pd.Series()

    for i in df.columns.tolist():

        if pd.api.types.is_object_dtype(df[i]):

            properties[i] = df[i].unique().tolist()

        elif pd.api.types.is_numeric_dtype(df[i]):

            properties[i] = round(df[i].describe(),2).tolist()

        elif pd.api.types.is_datetime64_any_dtype(df[i]):

            properties[i] = [df[i].min().strftime(format='%Y-%m-%d'), df[i].max().strftime(format='%Y-%m-%d')]

        elif pd.api.types.is_categorical_dtype(df[i]):

            properties[i] = list(df[i].unique())

    observe = pd.concat([df.dtypes, df.isnull().sum(), df.notnull().sum(), properties], axis=1)

    observe.columns = ['dtypes', 'Missing_Vals', 'Total_Vals', 'Properties']

    return observe



observe_data(df)
# We will binarize 'NumCompaniesWorked'

def binarize(column, bins, num_categories=[1,2,3]):

    x = pd.cut(x=column.tolist(), bins=bins, include_lowest=True)

    x.categories = num_categories

    tmp = pd.concat([column, pd.Series(x)], axis=1)

    

    column = x

    return column
# Categories are binarized into: 0-2 years: single; 3-5 years: few, 6-9 years: many

bins = pd.IntervalIndex.from_tuples([(-1, 2), (2, 5), (5, 9)])



# transforming 'NumCompaniesWorked' and a few more variables

df['NumCompaniesWorked'] = binarize(column=df['NumCompaniesWorked'], bins=bins).astype('O')

df['TrainingTimesLastYear'] = df['TrainingTimesLastYear'].astype('O')

df['WorkLifeBalance'] = df['WorkLifeBalance'].map({1:'Low',2:'Medium',3:'High',4:'Very High',5:'Very High'})

df['BusinessTravel'] = df['BusinessTravel'].map({'Travel_Rarely':2, 'Travel_Frequently':3, 'Non-Travel':1})
# List of all variables that are of 'O' type

categorical_features = df.select_dtypes(include=['object','category']).columns.tolist()

categorical_features.remove('Attrition')



# A view of categorical features

print('\033[1m' +'categorical features: ', '\033[0m',categorical_features)

print('='*100)

# List of all variables that are of 'float' or 'int' type

numerical_features = df.select_dtypes(include=np.number).columns.tolist()

print('\033[1m' +'numerical features: ', '\033[0m',numerical_features)

print('='*100)



# This lists shall be expanded after categorical encoding with categorical_features 

ordinal_features = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel','BusinessTravel', 'NumCompaniesWorked']

nominal_features = [i for i in categorical_features if i not in ordinal_features]

common = list(set(ordinal_features).intersection(set(numerical_features)))

actual_numerical = [i for i in numerical_features if i not in common]

print('\033[1m' +'ordinal features: ', '\033[0m',ordinal_features) #

print('='*100)

print('\033[1m' +'nominal features: ', '\033[0m',nominal_features) #

print('='*100)

print('\033[1m' +'actual numerical: ', '\033[0m',actual_numerical) #
# If we recall, all the variables with missing values were of numerical types,

# Let's view them once again to ensure the actual_numerical dtypes captures all of them

observe_data(df[actual_numerical])
# The missing values are found in 'Age', 'DailyRate', 'HourlyRate', 'MonthlyIncome' and 'MonthlyRate'; all are numerical type

df.loc[df.isna().any(axis=1),numerical_features]
# Scikit-learn libraries

from sklearn.preprocessing import StandardScaler, QuantileTransformer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier

from sklearn.metrics import recall_score, classification_report, confusion_matrix, roc_curve

from sklearn import tree

from sklearn.linear_model import LogisticRegression



# For oversampling through imbalance-learn

from imblearn.pipeline import make_pipeline, Pipeline

from imblearn.combine import SMOTETomek



# For data processing through feature-engine

from feature_engine.variable_transformers import YeoJohnsonTransformer

from feature_engine.missing_data_imputers import MeanMedianImputer

from feature_engine.categorical_encoders import WoERatioCategoricalEncoder



# For visualizing trees

from graphviz import Source

from IPython.display import SVG, Image



# Model Interpretation

import eli5

import shap
shap.initjs()
# Data preprocessing pipeline

## Train-Test split

X = df.drop(labels='Attrition', axis=1)

y = df['Attrition'].map({'Voluntary Resignation':1, 'Current employee':0})



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)



## Imputing numerical missing values from their respective 'mean'/ 'median'

impute = MeanMedianImputer(imputation_method='median', variables=actual_numerical)



## Transforming nominal categorical features based on their probability ratio (ordinal features are already transformed)

transform_nominal = WoERatioCategoricalEncoder(encoding_method='ratio', variables=nominal_features)



## Pre-processing pipeline

preprocessor = make_pipeline(impute, transform_nominal)



## Transforming the variable

X_train = preprocessor.fit_transform(X_train, y_train)

X_test = preprocessor.transform(X_test)



## Oversampling using SMOTE and creating hard boundaries using Tomac lines

smotetomec = SMOTETomek(random_state=0)

X_sample, y_sample = smotetomec.fit_resample(X_train, y_train)
## Instantiating Random Forest classifier

classifier = RandomForestClassifier(max_depth=5, min_samples_leaf=100, class_weight={1:1.5}, random_state=0)



## Parameters grid

parameter_grid = {

    'criterion': ['entropy', 'gini'], 

    'min_samples_leaf': [10, 50, 80], 

    'max_depth':[2,3,4,5]} #



## Instantiating and fitting GridsearchCV to the train set

gscvrf = GridSearchCV(estimator=classifier, param_grid=parameter_grid, cv=10, iid=False, scoring='f1_weighted', verbose=False)

gscvrf.fit(X_sample, y_sample)



## Predicting test outcomes

y_pred = gscvrf.predict(X_test)



## confusion matrix

cf = pd.DataFrame(confusion_matrix(y_test, y_pred), index=['current_employee', 'resigned'], columns=['current_employee', 'resigned'])[::-1].T[::-1]
# Model Evaluation

print(pd.Series(gscvrf.best_params_))

print('='*40)

print('recall score: %.3f' % recall_score(y_test, y_pred))

print('='*40)

print(classification_report(y_test, y_pred))

print('='*40)

print(cf)
# ROC Curve

y_pred_train_prob = gscvrf.predict_proba(X_train)[:,1]

y_pred_test__prob = gscvrf.predict_proba(X_test)[:,1]



fp_rate_train, tp_rate_train, thresh1 = roc_curve(y_train, y_pred_train_prob)

fp_rate_test, tp_rate_test, thresh2 = roc_curve(y_test, y_pred_test__prob)



plt.figure(figsize=(8,8))

plt.plot(fp_rate_train, tp_rate_train, label='train')

plt.plot(fp_rate_test, tp_rate_test, label='test')

plt.plot([0, 1], ls="--")

plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.title('ROC Curve', fontweight='semibold')

plt.legend(loc='center left', bbox_to_anchor=(1.,.5), frameon=False)

plt.grid()

plt.show()
# Let's interpret mean weightage of each feature in the model, globally

eli5.show_weights(gscvrf.best_estimator_, feature_names=X_train.columns.tolist())
# We can also view mean weightage of a sample observation, i.e., local interpret

eli5.show_prediction(estimator=gscvrf.best_estimator_, doc=X_test.sample(), feature_names=X_train.columns.tolist(), show_feature_values=True)
observations = shap.sample(X_test)

explainer_rf = shap.TreeExplainer(gscvrf.best_estimator_)



shap_vals_rf = explainer_rf.shap_values(observations)
shap.summary_plot(shap_values=shap_vals_rf, features=X_test)
shap.force_plot(base_value=explainer_rf.expected_value[1], shap_values=shap_vals_rf[1], features=observations, feature_names=X_test.columns.tolist())
# joblib.dump(value=gscvrf, filename=os.path.join(PATH, 'randomforest.pkl'))
## Instantiating Decision Tree as base classifier

base_classifier = tree.DecisionTreeClassifier(max_depth=5, min_impurity_decrease=0.001, class_weight={1:1.5})



## Instantiating Adaptive Boosting as meta classifier

meta_classifier = AdaBoostClassifier(learning_rate=0.1, random_state=0, base_estimator=base_classifier)



## Parameters grid

parameter_grid = {

    'n_estimators': [i for i in range(20,50,10)], 

    'learning_rate': [i for i in np.linspace(start=0.1, stop=0.25, num=5)]}



## Instantiating and fitting GridsearchCV to the train set

gscvab = GridSearchCV(estimator=meta_classifier, param_grid=parameter_grid, cv=10, iid=False, n_jobs=-1, scoring='f1_weighted', verbose=False)

gscvab.fit(X_sample, y_sample)



## Predicting test outcomes

y_pred = gscvab.predict(X_test)



## confusion matrix

cf = pd.DataFrame(confusion_matrix(y_test, y_pred), index=['current_employee', 'resigned'], columns=['current_employee', 'resigned'])[::-1].T[::-1]
# Model Evaluation

print(pd.Series(gscvab.best_params_))

print('='*40)

print('recall score: %.3f' % recall_score(y_test, y_pred))

print('='*40)

print(classification_report(y_test, y_pred))

print('='*40)

print(cf)
# ROC Curve

y_pred_train_prob = gscvab.predict_proba(X_train)[:,1]

y_pred_test__prob = gscvab.predict_proba(X_test)[:,1]



fp_rate_train, tp_rate_train, thresh1 = roc_curve(y_train, y_pred_train_prob)

fp_rate_test, tp_rate_test, thresh2 = roc_curve(y_test, y_pred_test__prob)



plt.figure(figsize=(8,8))

plt.plot(fp_rate_train, tp_rate_train, label='train')

plt.plot(fp_rate_test, tp_rate_test, label='test')

plt.plot([0, 1], ls="--")

plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.title('ROC Curve', fontweight='semibold')

plt.legend(loc='center left', bbox_to_anchor=(1.,.5), frameon=False)

plt.grid()

plt.show()
# Model Interpretation

pd.Series(gscvab.best_estimator_.feature_importances_, index=X_train.columns.tolist()).sort_values(ascending=False).plot(kind='bar', figsize=(10,6), title='Feature Importance');
# joblib.dump(value=gscvab, filename=os.path.join(PATH, 'adaboost.pkl'))
## Instantiating Random Forest classifier

classifier = GradientBoostingClassifier(n_estimators=30, min_samples_leaf=50, min_impurity_decrease=0.02, random_state=0, max_features='auto', n_iter_no_change=3)



## Parameters grid

parameter_grid = {

    'n_estimators': [20, 30, 40],

    'max_depth': [3,4,5], 

    'learning_rate': [i for i in np.linspace(start=0.1, stop=0.5, num=5)], 

    'loss': ['deviance', 'exponential']}



## Instantiating and fitting GridsearchCV to the train set

gscvgb = GridSearchCV(estimator=classifier, param_grid=parameter_grid, cv=10, scoring='f1_weighted', verbose=False)

gscvgb.fit(X_sample, y_sample)



## Predicting test outcomes

y_pred = gscvgb.predict(X_test)



## confusion matrix

cf = pd.DataFrame(confusion_matrix(y_test, y_pred), index=['current_employee', 'resigned'], columns=['current_employee', 'resigned'])[::-1].T[::-1]
# Model Evaluation

print(pd.Series(gscvgb.best_params_))

print('='*40)

print('recall score: %.3f' % recall_score(y_test, y_pred))

print('='*40)

print(classification_report(y_test, y_pred))

print('='*40)

print(cf)
# ROC Curve

y_pred_train_prob = gscvgb.predict_proba(X_train)[:,1]

y_pred_test__prob = gscvgb.predict_proba(X_test)[:,1]



fp_rate_train, tp_rate_train, thresh1 = roc_curve(y_train, y_pred_train_prob)

fp_rate_test, tp_rate_test, thresh2 = roc_curve(y_test, y_pred_test__prob)



plt.figure(figsize=(8,8))

plt.plot(fp_rate_train, tp_rate_train, label='train')

plt.plot(fp_rate_test, tp_rate_test, label='test')

plt.plot([0, 1], ls="--")

plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.title('ROC Curve', fontweight='semibold')

plt.legend(loc='center left', bbox_to_anchor=(1.,.5), frameon=False)

plt.grid()

plt.show()
# Let's interpret mean weightage of each feature in the model, globally

eli5.show_weights(gscvgb.best_estimator_, feature_names=X_train.columns.tolist())
# We can also view mean weightage of a sample observation, i.e., local interpret

eli5.show_prediction(estimator=gscvgb.best_estimator_, doc=X_test.sample(), feature_names=X_train.columns.tolist(), show_feature_values=True)
observations = shap.sample(X_test)

explainer_gb  = shap.TreeExplainer(gscvgb.best_estimator_)



shap_vals_gb = explainer_gb.shap_values(observations)
shap.summary_plot(shap_values=shap_vals_rf, features=observations)
shap.force_plot(base_value=explainer_gb.expected_value, shap_values=shap_vals_gb, features=observations, feature_names=X_test.columns.tolist())
# joblib.dump(value=gscvgb, filename=os.path.join(PATH, 'gradientboost.pkl'))
## Stacking Classifier Steps

### Extracting the best estimator from each model into a list

random_forest = gscvrf.best_estimator_

adaptive_boost= gscvab.best_estimator_

gradient_boost= gscvgb.best_estimator_



classifier_list = [('random_forest',random_forest), 

                   ('adaptive_boost',adaptive_boost), 

                   ('gradient_boost',gradient_boost)]



# ### Declaring meta classifier

m_classifier = LogisticRegression()



# ### Instantiating Stacking Classifier

stack = StackingClassifier(estimators=classifier_list, final_estimator=m_classifier)

stack.fit(X_sample, y_sample)



# Predicting test outcomes

y_pred = stack.predict(X_test)



# ## confusion matrix

cf = pd.DataFrame(confusion_matrix(y_test, y_pred), index=['current_employee', 'resigned'], columns=['current_employee', 'resigned'])[::-1].T[::-1]
# Model Evaluation

print(classification_report(y_test, y_pred))

print('='*80)

print(cf)
# ROC Curve

y_pred_train_prob = stack.predict_proba(X_train)[:,1]

y_pred_test__prob = stack.predict_proba(X_test)[:,1]



fp_rate_train, tp_rate_train, thresh1 = roc_curve(y_train, y_pred_train_prob)

fp_rate_test, tp_rate_test, thresh2 = roc_curve(y_test, y_pred_test__prob)



plt.figure(figsize=(8,8))

plt.plot(fp_rate_train, tp_rate_train, label='train')

plt.plot(fp_rate_test, tp_rate_test, label='test')

plt.plot([0, 1], ls="--")

plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.title('ROC Curve', fontweight='semibold')

plt.legend(loc='center left', bbox_to_anchor=(1.,.5), frameon=False)

plt.grid()

plt.show()
# Global Interpretation of the Meta Model of Stacking Classifier

eli5.show_weights(stack.final_estimator_, feature_names=['Random_Forest', 'Adaptive_Boost', 'Gradient_Boost'])
# How this perticular observation is scored?

eli5.show_prediction(estimator=stack.final_estimator_, doc=stack.transform(X_test)[0], feature_names=['Random_Forest', 'Adaptive_Boost', 'Gradient_Boost'], show_feature_values=True)
explainer = shap.LinearExplainer(model=stack.final_estimator_, data=stack.transform(X_test), nsamples=100)



observations = stack.transform(X_test.sample(1000, random_state=0))

shap_values = explainer.shap_values(observations)
shap.force_plot(base_value=explainer.expected_value, shap_values=shap_values, features=observations, feature_names=['Random_Forest', 'Adaptive_Boost', 'Gradient_Boost'])
shap.summary_plot(shap_values=shap_values, features=observations, feature_names=['Random_Forest', 'Adaptive_Boost', 'Gradient_Boost'])
# Classifier list: this time we'll also add stacking

classifier_list = [('random_forest',random_forest), 

                   ('adaptive_boost',adaptive_boost), 

                   ('gradient_boost',gradient_boost), 

                   ('stacking', stack)]



# Instantiating Stacking Classifier

vote = VotingClassifier(estimators=classifier_list, voting='soft')

vote.fit(X_sample, y_sample)



## Predicting test outcomes

y_pred = vote.predict(X_test)



# ## confusion matrix

cf = pd.DataFrame(confusion_matrix(y_test, y_pred), index=['current_employee', 'resigned'], columns=['current_employee', 'resigned'])[::-1].T[::-1]
# Model Evaluation

print(classification_report(y_test, y_pred))

print('='*40)

print(cf)
# ROC Curve

y_pred_train_prob = vote.predict_proba(X_train)[:,1]

y_pred_test__prob = vote.predict_proba(X_test)[:,1]



fp_rate_train, tp_rate_train, thresh1 = roc_curve(y_train, y_pred_train_prob)

fp_rate_test, tp_rate_test, thresh2 = roc_curve(y_test, y_pred_test__prob)



plt.figure(figsize=(8,8))

plt.plot(fp_rate_train, tp_rate_train, label='train')

plt.plot(fp_rate_test, tp_rate_test, label='test')

plt.plot([0, 1], ls="--")

plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.title('ROC Curve', fontweight='semibold')

plt.legend(loc='center left', bbox_to_anchor=(1.,.5), frameon=False)

plt.grid()

plt.show()
compensation = pd.concat([X_train['DailyRate'], y_train], axis=1)



sns.boxplot(data=compensation, y='DailyRate', x='Attrition');