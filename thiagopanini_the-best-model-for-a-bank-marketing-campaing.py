# Standard libraries
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
from math import ceil
from datetime import datetime

# Viz libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Utils (homemade)
from viz_utils import *
from prep_utils import *
from ml_utils import *

# Ml libraries
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
import lightgbm as lgb

# Deep Learning frameworks
import tensorflow as tf
# Reading the data
path = '../input/bank-marketing-dataset/bank.csv'
df_ori = pd.read_csv(path, sep=',')
df_ori.columns = [col.lower().strip().replace('.', '_') for col in df_ori.columns]

print(f'Data shape: {df_ori.shape}')
df_ori.head()
# Creating the class object
prep = DataPrep()

# Transforming the dataset target
df = df_ori.copy()
df['target'] = (df['deposit'] == 'yes') * 1
df.drop('deposit', axis=1, inplace=True)

# Returing an overview from the data
target = 'target'
df_overview = prep.data_overview(df, label_name=target)
df_overview
# Ploting a donut chart with the target variable
label_names = ['No', 'Yes']
color_list = ['salmon', 'darkslateblue']

fig, ax = plt.subplots(figsize=(8, 8))
title = 'Donut Chart for Target Variable'
donut_plot(df, target, label_names, ax=ax, text=f'Total: {len(df)}', colors=color_list, title=title)
plt.show()
# Visualizing the categorical features
cat_features = [col for col, dtype in df.dtypes.items() if dtype == 'object']
catplot_analysis(df, cat_features, fig_cols=3, hue='target', palette=['salmon', 'darkslateblue'], figsize=(16, 16))
# Parameters
num_features = ['age', 'balance', 'duration', 'campaign']
color_list = ['salmon', 'darkslateblue']
# Analyzing numerical features
distplot(df, num_features, fig_cols=3, hue='target', color=color_list, figsize=(16, 12))
# Stripplot
stripplot(df, num_features, fig_cols=3, hue='target', palette=color_list)
# Stripplot
boxenplot(df, num_features, fig_cols=3, hue='target', palette=color_list)
# Analisando top variáveis com maior correlação POSITIVA
top_pos_corr_cols = target_correlation_matrix(df, label_name='target', corr='positive')
# Columns to be dropped
to_drop = ['duration']
df_drop = df.drop(to_drop, axis=1)

# Verifyng
print(f'Shape before the drop: {df.shape}')
print(f'Shape after the drop: {df_drop.shape}')
df_drop.head()
# Splitting the data
X = df_drop.drop('target', axis=1)
y = df_drop['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
# Returing features by dtype
num_features = [col for col, dtype in X_train.dtypes.items() if dtype != 'object']
cat_features = [col for col, dtype in X_train.dtypes.items() if dtype == 'object']
print(f'Total of numerical features: {len(num_features)}')
print(f'Total of categorical features: {len(cat_features)}')

# Splitting data by dtype
X_train_num = X_train[num_features]
X_train_cat = X_train[cat_features]
print(f'\nShape of numerical training data: {X_train_num.shape}')
print(f'Shape of categorical training data: {X_train_cat.shape}')
# Class for splitting the data by dtype
class SplitDataDtype(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Returing features by dtype
        self.num_features = [col for col, dtype in X.dtypes.items() if dtype != 'object']
        self.cat_features = [col for col, dtype in X.dtypes.items() if dtype == 'object']
        
        # Indexing data
        X_num = X[self.num_features]
        X_cat = X[self.cat_features]
        
        return X_num, X_cat
# Creating object and calling the fit_transform method
dtype_splitter = SplitDataDtype()
X_train_num, X_train_cat = dtype_splitter.fit_transform(X_train)

print(f'Shape of numerical training data: {X_train_num.shape}')
print(f'Shape of categorical training data: {X_train_cat.shape}')
# Class por encoding the data
class DummiesEncoding(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Collecting variables
        self.cat_features_ori = [col for col, dtype in X.dtypes.items() if dtype == 'object']
        
        # Applying encoding with get_dummies()
        X_cat_dum = pd.get_dummies(X)
        
        # Merging the datasets and eliminating old columns
        X_dum = X.join(X_cat_dum)
        X_dum = X_dum.drop(self.cat_features_ori, axis=1)
        self.features_after_encoding = list(X_dum.columns)
        
        return X_dum
# Applying encoding on categorical data
encoder = DummiesEncoding()
X_train_encoded = encoder.fit_transform(X_train_cat)

print(f'Shape of X_train_encoded: {X_train_encoded.shape}')
X_train_encoded.head()
# Scaling with StandardScaler() class
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_num)

# Looking at the first line
X_train_scaled[0]
# Initial block code for splitting the data
dtype_spliter = SplitDataDtype()
X_num, X_cat = dtype_spliter.fit_transform(X_train)
num_features = dtype_spliter.num_features
cat_features = dtype_spliter.cat_features

# Numerical pipeline
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('encoder', DummiesEncoding())
])

# Full pipeline
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Applying the complete pipeline on the training set
X_train_prep = full_pipeline.fit_transform(X_train)

# Returing features
cat_features_encoded = full_pipeline.named_transformers_['cat']['encoder'].features_after_encoding
model_features = num_features + cat_features_encoded
# Result
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_train_prep: {X_train_prep.shape}')
print(f'Total features: {len(model_features)}')
print(f'\nFirst line of X_train_prep: \n\n{X_train_prep[0]}')
# Applying the same pipeline for the test set
X_test_prep = full_pipeline.fit_transform(X_test)

print(f'Shape of X_test_prep: {X_test_prep.shape}')
# Saving everything on a prepared set to feed some homemade classes
set_prep = {
    'X_train_prep': X_train_prep,
    'X_test_prep': X_test_prep,
    'y_train': y_train,
    'y_test': y_test
}
# Creating the model and a class object
logreg_clf = LogisticRegression()
logreg_tool = BinaryBaselineClassifier(logreg_clf, set_prep, model_features)
# Defining hyperparmeters
logreg_param_grid = {
    'C': np.linspace(0.1, 10, 20),
    'penalty': ['l1', 'l2'],
    'class_weight': ['balanced', None],
    'random_state': [42],
    'solver': ['liblinear']
}

# Training the model and optimizing AUC score
logreg_tool.fit(rnd_search=True, param_grid=logreg_param_grid, scoring='roc_auc')
# Model performance
logreg_train_performance = logreg_tool.evaluate_performance()
logreg_train_performance
# Plotting confusion matrix
title = 'Logistic Regression\nConfusion Matrix'
classes = ['No', 'Yes']

# Creating figure and calling the method
plt.figure(figsize=(5, 5))
logreg_tool.plot_confusion_matrix(classes, title=title)
plt.show()
plt.figure(figsize=(12, 7))
logreg_tool.plot_roc_curve()
plt.annotate('Área under the curve (ROC) with 50%\n(Random model)', xy=(0.5, 0.5), xytext=(0.6, 0.4),
             arrowprops=dict(facecolor='#6E726D', shrink=0.05))
plt.show()
# Plotting the learning curve
logreg_tool.plot_learning_curve()
# Full performance with Logistic Regression
logreg_test_performance = logreg_tool.evaluate_performance(test=True)
logreg_performance = logreg_train_performance.append(logreg_test_performance)
logreg_performance
# Creating objects
tree_model = DecisionTreeClassifier()
tree_tool = BinaryBaselineClassifier(tree_model, set_prep, model_features)
# Defining hyperparameters
tree_param_grid = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [3, 5, 10, 20],
    'max_features': np.arange(1, X_train_prep.shape[1]),
    'class_weight': ['balanced', None],
    'random_state': [42]
}

tree_tool.fit(rnd_search=True, scoring='roc_auc', param_grid=tree_param_grid)
# Performance
tree_train_performance = tree_tool.evaluate_performance()
tree_train_performance
# Variables to plotting
logreg_title = 'LogisticRegression\nConfusion Matrix'
tree_title = 'DecisionTree Classifier\nConfusion Matrix'
classes = ['No', 'Yes']

# Creating figure and calling the method
plt.figure(figsize=(10, 5))

# Logistic Regression
plt.subplot(1, 2, 1)
logreg_tool.plot_confusion_matrix(classes, title=logreg_title)

# Decision Trees
plt.subplot(1, 2, 2)
tree_tool.plot_confusion_matrix(classes, title=tree_title, cmap=plt.cm.Greens)
plt.tight_layout()
plt.show()
# Creating a figure and calling the method for each estimator
plt.figure(figsize=(12, 7))

logreg_tool.plot_roc_curve()
tree_tool.plot_roc_curve()
plt.annotate('Área under the curve (ROC) with 50%\n(Random model)', xy=(0.5, 0.5), xytext=(0.6, 0.4),
             arrowprops=dict(facecolor='#6E726D', shrink=0.05))
plt.show()
# Plotting the learning curve
tree_tool.plot_learning_curve()
# Evaluating feature importance
feat_imp = tree_tool.feature_importance_analysis()
feat_imp.head(30)
# Complete performance
tree_test_performance = tree_tool.evaluate_performance(test=True)
tree_performance = tree_train_performance.reset_index().append(tree_test_performance.reset_index())

all_performances = logreg_performance.reset_index().append(tree_performance).reset_index(drop=True)
cm = sns.light_palette('cornflowerblue', as_cmap=True)
all_performances.style.background_gradient(cmap=cm)
# Creating objects
forest_model = RandomForestClassifier()
forest_tool = BinaryBaselineClassifier(forest_model, set_prep, model_features)
# Defining hyperparameters
forest_param_grid = {
    'bootstrap': [True, False],
    'max_depth': [3, 5, 10, 20, 50],
    'n_estimators': [50, 100, 200, 500],
    'random_state': [42],
    'max_features': ['auto', 'sqrt'],
    'class_weight': ['balanced', None]
}

forest_tool.fit(rnd_search=True, scoring='roc_auc', param_grid=forest_param_grid)
# Model performance
forest_train_performance = forest_tool.evaluate_performance()
forest_train_performance
# Variables for plotting
logreg_title = 'LogisticRegression\nConfusion Matrix'
tree_title = 'DecisionTree Classifier\nConfusion Matrix'
forest_title = 'RandomForest Classifier\nConfusion Matrix'
classes = ['No', 'Yes']

# Creating figure and calling the method
plt.figure(figsize=(15, 5))

# Logistic Regression
plt.subplot(1, 3, 1)
logreg_tool.plot_confusion_matrix(classes, title=logreg_title)

# Decision Trees
plt.subplot(1, 3, 2)
tree_tool.plot_confusion_matrix(classes, title=tree_title, cmap=plt.cm.Greens)

# Random Forest
plt.subplot(1, 3, 3)
tree_tool.plot_confusion_matrix(classes, title=forest_title, cmap=plt.cm.Reds)

plt.tight_layout()
plt.show()
# Creating figure and calling the method for each estimator
plt.figure(figsize=(12, 7))

# ROC Curve for the models
logreg_tool.plot_roc_curve()
tree_tool.plot_roc_curve()
forest_tool.plot_roc_curve()

# Annotation
plt.annotate('Área under the curve (ROC) with 50%\n(Random model)', xy=(0.5, 0.5), xytext=(0.6, 0.4),
             arrowprops=dict(facecolor='#6E726D', shrink=0.05))
plt.show()
# Plotting the learning curve
forest_tool.plot_learning_curve()
# Complete performance
forest_test_performance = forest_tool.evaluate_performance(test=True)
forest_performance = forest_train_performance.reset_index().append(forest_test_performance.reset_index())

all_performances = all_performances.append(forest_performance).reset_index(drop=True)
cm = sns.light_palette('cornflowerblue', as_cmap=True)
all_performances.style.background_gradient(cmap=cm)
# Creating objects
voting_model = VotingClassifier(
    estimators=[('logreg', logreg_tool.trained_model), ('forest', forest_tool.trained_model)],
    voting='soft'
)

# Training the model
voting_tool = BinaryBaselineClassifier(voting_model, set_prep, model_features)
voting_tool.fit(rnd_search=False)
# Model performance
voting_train_performance = voting_tool.evaluate_performance()
voting_train_performance
# Plotting the matrices
logreg_title = 'LogisticRegression\nConfusion Matrix'
tree_title = 'DecisionTree Classifier\nConfusion Matrix'
forest_title = 'RandomForest Classifier\nConfusion Matrix'
voting_title = 'Voting Classifier\nConfusion Matrix'
classes = ['No', 'Yes']

# Creating figure e calling the model
plt.figure(figsize=(15, 10))

# Logistic Regression
plt.subplot(2, 3, 1)
logreg_tool.plot_confusion_matrix(classes, title=logreg_title)

# Decision Trees
plt.subplot(2, 3, 2)
tree_tool.plot_confusion_matrix(classes, title=tree_title, cmap=plt.cm.Greens)

# Random Forest
plt.subplot(2, 3, 3)
tree_tool.plot_confusion_matrix(classes, title=forest_title, cmap=plt.cm.Reds)

# Voting Classifier
plt.subplot(2, 3, 4)
voting_tool.plot_confusion_matrix(classes, title=voting_title, cmap=plt.cm.Greys)

plt.tight_layout()
plt.show()
# Creating figure and calling the method for each estimator
plt.figure(figsize=(12, 7))

# ROC Curve for the models
logreg_tool.plot_roc_curve()
tree_tool.plot_roc_curve()
forest_tool.plot_roc_curve()
voting_tool.plot_roc_curve()

# Annotation
plt.annotate('Área under the curve (ROC) with 50%\n(Random model)', xy=(0.5, 0.5), xytext=(0.6, 0.4),
             arrowprops=dict(facecolor='#6E726D', shrink=0.05))
plt.show()
# Plotting the learning curve
voting_tool.plot_learning_curve()
# Complete performance
voting_test_performance = voting_tool.evaluate_performance(test=True)
voting_performance = voting_train_performance.reset_index().append(voting_test_performance.reset_index())

all_performances = all_performances.append(voting_performance).reset_index(drop=True)
cm = sns.light_palette('cornflowerblue', as_cmap=True)
all_performances.style.background_gradient(cmap=cm)
# Creating the bagging model based on the Random Forest Classifier
bagging_model = BaggingClassifier(
    RandomForestClassifier(bootstrap=False, criterion='gini', max_depth=10, max_features='sqrt', 
                           n_estimators=50, random_state=42), 
    n_estimators=20,
    max_samples=100,
    bootstrap=True,
    n_jobs=-1,
    oob_score=True
)

# Training model
bagging_tool = BinaryBaselineClassifier(bagging_model, set_prep, model_features)
bagging_tool.fit(rnd_search=False)
# Verificando performance
bagging_train_performance = bagging_tool.evaluate_performance()
bagging_train_performance
# Plotting Confusion Matrix
logreg_title = 'LogisticRegression\nConfusion Matrix'
tree_title = 'DecisionTree Classifier\nConfusion Matrix'
forest_title = 'RandomForest Classifier\nConfusion Matrix'
voting_title = 'Voting Classifier\nConfusion Matrix'
bagging_title = 'Bootstrap Aggregating\nConfusion Matrix'
classes = ['No', 'Yes']

# Creating figure and calling the method
plt.figure(figsize=(15, 10))

# Regressão Logística
plt.subplot(2, 3, 1)
logreg_tool.plot_confusion_matrix(classes, title=logreg_title)

# Decision Trees
plt.subplot(2, 3, 2)
tree_tool.plot_confusion_matrix(classes, title=tree_title, cmap=plt.cm.Greens)

# Random Forest
plt.subplot(2, 3, 3)
tree_tool.plot_confusion_matrix(classes, title=forest_title, cmap=plt.cm.Reds)

# Voting Classifier
plt.subplot(2, 3, 4)
voting_tool.plot_confusion_matrix(classes, title=voting_title, cmap=plt.cm.Greys)

# Bagging Classifier
plt.subplot(2, 3, 5)
bagging_tool.plot_confusion_matrix(classes, title=bagging_title, cmap=plt.cm.Oranges)

plt.tight_layout()
plt.show()
# Creating figure and calling the method
plt.figure(figsize=(12, 7))

# Plotting the ROC Curve
logreg_tool.plot_roc_curve()
tree_tool.plot_roc_curve()
forest_tool.plot_roc_curve()
voting_tool.plot_roc_curve()
bagging_tool.plot_roc_curve()

# Anotação
plt.annotate('Área under the curve (ROC) with 50%\n(Random model)', xy=(0.5, 0.5), xytext=(0.6, 0.4),
             arrowprops=dict(facecolor='#6E726D', shrink=0.05))
plt.show()
# Plotting the learning curve
bagging_tool.plot_learning_curve()
# Complete Performance
bagging_test_performance = bagging_tool.evaluate_performance(test=True)
bagging_performance = bagging_train_performance.reset_index().append(bagging_test_performance.reset_index())

all_performances = all_performances.append(bagging_performance).reset_index(drop=True)
cm = sns.light_palette('cornflowerblue', as_cmap=True)
all_performances.style.background_gradient(cmap=cm)
# Training the model
adaboost_model = AdaBoostClassifier(
    RandomForestClassifier(bootstrap=False, criterion='gini', max_depth=10, max_features='sqrt', 
                           n_estimators=50, random_state=42), 
    n_estimators=20,
    learning_rate=0.5,
    random_state=42
)

adaboost_tool = BinaryBaselineClassifier(adaboost_model, set_prep, model_features)
adaboost_tool.fit(rnd_search=False)
# Model performance
adaboost_train_performance = adaboost_tool.evaluate_performance()
adaboost_train_performance
# PLotting the matrices
logreg_title = 'LogisticRegression\nConfusion Matrix'
tree_title = 'DecisionTree Classifier\nConfusion Matrix'
forest_title = 'RandomForest Classifier\nConfusion Matrix'
voting_title = 'Voting Classifier\nConfusion Matrix'
bagging_title = 'Bootstrap Aggregating\nConfusion Matrix'
adaboost_title = 'Adaptative Boosting\nConfusion Matrix'
classes = ['No', 'Yes']

# Creating figure and calling the methods
plt.figure(figsize=(15, 10))

# Logistic Regression
plt.subplot(2, 3, 1)
logreg_tool.plot_confusion_matrix(classes, title=logreg_title)

# Decision Trees
plt.subplot(2, 3, 2)
tree_tool.plot_confusion_matrix(classes, title=tree_title, cmap=plt.cm.Greens)

# Random Forest
plt.subplot(2, 3, 3)
tree_tool.plot_confusion_matrix(classes, title=forest_title, cmap=plt.cm.Reds)

# Voting Classifier
plt.subplot(2, 3, 4)
voting_tool.plot_confusion_matrix(classes, title=voting_title, cmap=plt.cm.Greys)

# Bagging Classifier
plt.subplot(2, 3, 5)
bagging_tool.plot_confusion_matrix(classes, title=bagging_title, cmap=plt.cm.Oranges)

# Adaboost Classifier
plt.subplot(2, 3, 6)
adaboost_tool.plot_confusion_matrix(classes, title=adaboost_title, cmap=plt.cm.Purples)

plt.tight_layout()
plt.show()
# Creating figure
plt.figure(figsize=(12, 7))

# Plotting the ROC Curve for each estimator
logreg_tool.plot_roc_curve()
tree_tool.plot_roc_curve()
forest_tool.plot_roc_curve()
voting_tool.plot_roc_curve()
bagging_tool.plot_roc_curve()
adaboost_tool.plot_roc_curve()

# Annotation
plt.annotate('Área under the curve (ROC) with 50%\n(Random model)', xy=(0.5, 0.5), xytext=(0.6, 0.4),
             arrowprops=dict(facecolor='#6E726D', shrink=0.05))
plt.show()
# Learning Curve
adaboost_tool.plot_learning_curve()
# Complete Performance
adaboost_test_performance = adaboost_tool.evaluate_performance(test=True)
adaboost_performance = adaboost_train_performance.reset_index().append(adaboost_test_performance.reset_index())

all_performances = all_performances.append(adaboost_performance).reset_index(drop=True)
cm = sns.light_palette('cornflowerblue', as_cmap=True)
all_performances.style.background_gradient(cmap=cm)
# Creating the object
gboost_model = GradientBoostingClassifier(
    n_estimators=20,
    learning_rate=1.0,
    max_depth=5, 
    max_features=15,
    random_state=42
)

# Training the model
gboost_tool = BinaryBaselineClassifier(gboost_model, set_prep, model_features)
gboost_tool.fit(rnd_search=False)
# Model Performance
gboost_train_performance = gboost_tool.evaluate_performance()
gboost_train_performance
# Plotting matrices
logreg_title = 'LogisticRegression\nConfusion Matrix'
tree_title = 'DecisionTree Classifier\nConfusion Matrix'
forest_title = 'RandomForest Classifier\nConfusion Matrix'
voting_title = 'Voting Classifier\nConfusion Matrix'
bagging_title = 'Bootstrap Aggregating\nConfusion Matrix'
adaboost_title = 'Adaptative Boosting\nConfusion Matrix'
gboost_title = 'Gradient Boosting\nConfusion Matrix'
classes = ['No', 'Yes']

# Creating figure and calling the methods
plt.figure(figsize=(15, 15))

# Logistic Regression
plt.subplot(3, 3, 1)
logreg_tool.plot_confusion_matrix(classes, title=logreg_title)

# Decision Trees
plt.subplot(3, 3, 2)
tree_tool.plot_confusion_matrix(classes, title=tree_title, cmap=plt.cm.Greens)

# Random Forest
plt.subplot(3, 3, 3)
tree_tool.plot_confusion_matrix(classes, title=forest_title, cmap=plt.cm.Reds)

# Voting Classifier
plt.subplot(3, 3, 4)
voting_tool.plot_confusion_matrix(classes, title=voting_title, cmap=plt.cm.Greys)

# Bagging Classifier
plt.subplot(3, 3, 5)
bagging_tool.plot_confusion_matrix(classes, title=bagging_title, cmap=plt.cm.Oranges)

# Adaboost Classifier
plt.subplot(3, 3, 6)
adaboost_tool.plot_confusion_matrix(classes, title=adaboost_title, cmap=plt.cm.Purples)

# Gradient Boosting
plt.subplot(3, 3, 7)
gboost_tool.plot_confusion_matrix(classes, title=gboost_title, cmap=plt.cm.cool)

plt.tight_layout()
plt.show()
# Creating figure
plt.figure(figsize=(12, 7))

# Plotando curva para os modelos
logreg_tool.plot_roc_curve()
tree_tool.plot_roc_curve()
forest_tool.plot_roc_curve()
voting_tool.plot_roc_curve()
bagging_tool.plot_roc_curve()
adaboost_tool.plot_roc_curve()
gboost_tool.plot_roc_curve()

# Annotation
plt.annotate('Área under the curve (ROC) with 50%\n(Random model)', xy=(0.5, 0.5), xytext=(0.6, 0.4),
             arrowprops=dict(facecolor='#6E726D', shrink=0.05))
plt.show()
# Learning Curve
gboost_tool.plot_learning_curve()
# Complete performance
gboost_test_performance = gboost_tool.evaluate_performance(test=True)
gboost_performance = gboost_train_performance.reset_index().append(gboost_test_performance.reset_index())

all_performances = all_performances.append(gboost_performance).reset_index(drop=True)
cm = sns.light_palette('cornflowerblue', as_cmap=True)
all_performances.style.background_gradient(cmap=cm)
# Setting up a LightGBM model
train_data = lgb.Dataset(X_train_prep, label=y_train)
test_data = lgb.Dataset(X_test_prep, label=y_test)

# Parameters
lgbm_params = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}
lgbm_model = lgb.LGBMClassifier(**lgbm_params)
# Training the model
lgbm_tool = BinaryBaselineClassifier(lgbm_model, set_prep, model_features)
lgbm_tool.fit(rnd_search=False)
# Model Performance
lgbm_train_performance = lgbm_tool.evaluate_performance()
lgbm_train_performance
# Plotting matrices
logreg_title = 'LogisticRegression\nConfusion Matrix'
tree_title = 'DecisionTree Classifier\nConfusion Matrix'
forest_title = 'RandomForest Classifier\nConfusion Matrix'
voting_title = 'Voting Classifier\nConfusion Matrix'
bagging_title = 'Bootstrap Aggregating\nConfusion Matrix'
adaboost_title = 'Adaptative Boosting\nConfusion Matrix'
gboost_title = 'Gradient Boosting\nConfusion Matrix'
lgbm_title = 'LightGBM\nConfusion Matrix'
classes = ['No', 'Yes']

# Creating figure
plt.figure(figsize=(15, 15))

# Logistic Regression
plt.subplot(3, 3, 1)
logreg_tool.plot_confusion_matrix(classes, title=logreg_title)

# Decision Trees
plt.subplot(3, 3, 2)
tree_tool.plot_confusion_matrix(classes, title=tree_title, cmap=plt.cm.Greens)

# Random Forest
plt.subplot(3, 3, 3)
tree_tool.plot_confusion_matrix(classes, title=forest_title, cmap=plt.cm.Reds)

# Voting Classifier
plt.subplot(3, 3, 4)
voting_tool.plot_confusion_matrix(classes, title=voting_title, cmap=plt.cm.Greys)

# Bagging Classifier
plt.subplot(3, 3, 5)
bagging_tool.plot_confusion_matrix(classes, title=bagging_title, cmap=plt.cm.Oranges)

# Adaboost Classifier
plt.subplot(3, 3, 6)
adaboost_tool.plot_confusion_matrix(classes, title=adaboost_title, cmap=plt.cm.Purples)

# Gradient Boosting
plt.subplot(3, 3, 7)
gboost_tool.plot_confusion_matrix(classes, title=gboost_title, cmap=plt.cm.cool)

# LightGBM
plt.subplot(3, 3, 8)
lgbm_tool.plot_confusion_matrix(classes, title=lgbm_title, cmap=plt.cm.winter)

plt.tight_layout()
plt.show()
# Creating figure and calling the method
plt.figure(figsize=(12, 7))

# Plotting curves
logreg_tool.plot_roc_curve()
tree_tool.plot_roc_curve()
forest_tool.plot_roc_curve()
voting_tool.plot_roc_curve()
bagging_tool.plot_roc_curve()
adaboost_tool.plot_roc_curve()
gboost_tool.plot_roc_curve()
lgbm_tool.plot_roc_curve()

# Annotation
plt.annotate('Área under the curve (ROC) with 50%\n(Random model)', xy=(0.5, 0.5), xytext=(0.6, 0.4),
             arrowprops=dict(facecolor='#6E726D', shrink=0.05))
plt.show()
# Learning curve
lgbm_tool.plot_learning_curve()
# Complete performance
lgbm_test_performance = lgbm_tool.evaluate_performance(test=True)
lgbm_performance = lgbm_train_performance.reset_index().append(lgbm_test_performance.reset_index())

all_performances = all_performances.append(lgbm_performance).reset_index(drop=True)
cm = sns.light_palette('cornflowerblue', as_cmap=True)
all_performances.style.background_gradient(cmap=cm)
