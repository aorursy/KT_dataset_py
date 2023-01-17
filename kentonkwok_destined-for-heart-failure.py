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
df= pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df
df.describe()
import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('darkgrid')



plt.figure(figsize=(10,10))

sns.heatmap(df.corr(), annot=True, square=True)
df.corr()['DEATH_EVENT'].apply(np.abs).sort_values(ascending=False)
sns.pairplot(df, vars=['serum_creatinine', 'ejection_fraction', 'age', 'serum_sodium', 'platelets'], hue = 'DEATH_EVENT')
df['DEATH_EVENT'].mean()
#smoke_sex = df.groupby(['smoking','sex', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack()

pd.crosstab([df.smoking, df.sex ], df['DEATH_EVENT']) 
# normalise index because we want to see the proportion of each group

pd.crosstab([df.smoking, df.sex], df['DEATH_EVENT'], normalize='index') 
from sklearn.model_selection import train_test_split



# Select subset of predictors

cols_to_use = df.columns[: - 2]

X = df[cols_to_use]



# Select target

y = df.iloc[:,-1]



# Separate data into training and validation sets

# as splits have randomness, we apply a random_state seed for reproducibility

# stratified as imbalanced classes

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=20, stratify = y)

    
# define useful helper function for displaying cv scores.



def display_scores(scores, return_scores = False):

    if return_scores: 

        print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())
# import models

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC, NuSVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



# import method for selection criteria

from sklearn.model_selection import cross_val_score





# create list of Classification models to choose from

models = [XGBClassifier(),

          SVC(C= 0.1),

          NuSVC(),

          LogisticRegression(),

          RandomForestClassifier(),

          DecisionTreeClassifier()]



# loop through models and evaluate their performance based on 10-fold cross validation 

def loopthruModels(models, X, y, cv=10): 

    for model in models: 

        print(str(model))

        scores = cross_val_score(model, X,y, scoring="f1_macro", cv= cv)

        display_scores(scores)

        print('---')

        

loopthruModels(models, X_train, y_train)
from sklearn.model_selection import GridSearchCV



xg_param_grid = [

    {'n_estimators': [50, 70, 100, 500, 1000], 

     'learning_rate':[0.001, 0.01, 0.05, 0.1,  0.5],

     'n_jobs': [4]}

]



xgmodel = XGBClassifier()



grid_search = GridSearchCV(xgmodel, xg_param_grid, cv=10, scoring="f1_macro", return_train_score= True)

grid_search.fit(X_train,y_train)

print(grid_search.best_params_)



cvres = grid_search.cv_results_



for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):

    print(mean_score, params)
forest_param_grid = [

    {'n_estimators': [200, 700, 1000], 

     'max_features':[0.2, 0.5, None],

    'n_jobs': [4]}

]



forest = RandomForestClassifier()



grid_search = GridSearchCV(forest, forest_param_grid, cv=10, scoring="f1_macro", return_train_score=True)

grid_search.fit(X_train,y_train)

print(grid_search.best_params_)



cvres = grid_search.cv_results_



for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):

    print(mean_score, params)
forest_model = RandomForestClassifier()



forest_model.fit(X_train, y_train)



# evaluate each feature's importance

importances = forest_model.feature_importances_



# sort by descending order of importances

indices = np.argsort(importances)[::-1]



#create sorted dictionary

forest_importances = {}



print("Feature ranking:")

for f in range(X.shape[1]):

    forest_importances[X.columns[indices[f]]] = importances[indices[f]]

    print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel



# Set the regularization parameter C=1

logistic = LogisticRegression(C = .1, penalty="l1", solver='liblinear', random_state=7).fit(X_train, y_train)

model = SelectFromModel(logistic, prefit=True)



X_new = model.transform(X)

X_new



# Get back the kept features as a DataFrame with dropped columns as all 0s

selected_features = pd.DataFrame(model.inverse_transform(X_new), 

                                 index=X.index,

                                 columns=X.columns)



# Dropped columns have values of all 0s, keep other columns 

reg_selected_columns = selected_features.columns[selected_features.var() != 0]

reg_selected_columns
# define function for evaluating different feature selections



def eval_features(model, param_grid, feature_list): 

    for feature in feature_list: 

        grid_search = GridSearchCV(model, param_grid, cv=10, scoring="f1_macro", return_train_score= True)

        grid_search.fit(X_train[feature],y_train)

        print(grid_search.best_params_)



        cvres = grid_search.cv_results_

        max_score = 0 



        for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):

            if mean_score > max_score:

                max_score = mean_score

                best_params = params    

        print(max_score, best_params)
xgmodel = XGBClassifier()



# parameter grid for XGBoost

xg_param_grid = [

    {'n_estimators': [25, 50,  500, 1000], 

     'learning_rate':[0.001, 0.01, 0.05, 0.1,  0.5],

     'n_jobs': [4]}

]



# we want to test this series of features

feature_list = [['serum_creatinine', 'ejection_fraction'],

                ['serum_creatinine', 'ejection_fraction', 'age'], 

                ['serum_creatinine', 'ejection_fraction', 'age', 'creatinine_phosphokinase'],

                ['serum_creatinine', 'ejection_fraction', 'age', 'creatinine_phosphokinase', 'platelets'],

                reg_selected_columns]



eval_features(xgmodel, xg_param_grid, feature_list)    
forest = RandomForestClassifier()



forest_param_grid = [

    {'n_estimators': [200, 700, 1000], 

     'max_features':[0.2, 0.5, None],

    'n_jobs': [4]}

]



# use same feature list

eval_features(forest, forest_param_grid, feature_list)   
# defining useful functions

from sklearn.metrics import confusion_matrix

def getScore(model, X_train= X_train, y_train= y_train): 

    my_pipeline = Pipeline(steps=[

                                  ('model', model)

                                 ])

    # Preprocessing of training data, fit model 

    my_pipeline.fit(X_train, y_train)



    # Preprocessing of validation data, get predictions

    preds = my_pipeline.predict(X_valid[sel_cols])

    score = my_pipeline.score(X_valid[sel_cols], y_valid)

    return score



def getconfusion(model, y_valid, y_pred, f1= True): 

    cf_matrix = confusion_matrix(y_valid, y_pred)



    f1 = model.score(X_valid[sel_cols], y_valid)

    print('The F1 score is:', f1)

    sns.heatmap(cf_matrix, annot=True)

    plt.xlabel('Predicted values')

    plt.ylabel('True values')

from sklearn.pipeline import Pipeline



# relevant columns for a good prediction

sel_cols = ['serum_creatinine', 'ejection_fraction', 'age']



# model with optimised hyperparameters

model = XGBClassifier(n_estimators=1000, learning_rate= 0.5)



steps=[

       ('model', model)

      ]



# Bundle preprocessing and modeling code in a pipeline

pipe = Pipeline(steps)



# Preprocessing of training data, fit model 

pipe.fit(X_train[sel_cols], y_train)



# Preprocessing of validation data, get predictions

preds = pipe.predict(X_valid[sel_cols])



# Evaluate the model using the confusion matrix

getconfusion(pipe, y_valid, preds)
