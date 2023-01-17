# To import & read dataset

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# For graphical representations of data & information

import seaborn as sns

import matplotlib.pyplot as plt



# To deprecate unwanted warnings (optional)

import warnings



# Data Preprocessing

from sklearn.preprocessing import LabelEncoder   # To convert string/char data (here target labels) to integer type



# To prepare Train & Test dataset

from sklearn.model_selection import train_test_split



# To Load required training libraries

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

import xgboost as xgb



# Include libraries to check performance of model

from sklearn.metrics import classification_report, confusion_matrix



# To Tune the training model

from sklearn.model_selection import GridSearchCV

# print(os.listdir("../input/"))



pd.set_option('display.max_rows', None, 'display.max_columns', None)           # To display all the row & columns of the dataframe (optional)



df_cancer = pd.read_csv("../input/data.csv")



df_cancer.keys()
# Deprecating all warning we get below

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

warnings.filterwarnings(action='ignore', category=FutureWarning)

warnings.filterwarnings(action='ignore', category=Warning)

# Count the target samples 

sns.countplot(df_cancer['diagnosis'])    # Count the samples
# Assign the Target Classes (M & B) integers values (0 & 1)

diagnosis_feature = df_cancer['diagnosis']

catConvertor = LabelEncoder()

df_cancer['diagnosis']= catConvertor.fit_transform(df_cancer['diagnosis'].astype('str'))

# Remove redundant fields

df_cancer = df_cancer.drop(['id','Unnamed: 32'], axis=1)
# Visualize the relation between various fields

sns.pairplot(df_cancer, hue='diagnosis', vars=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 

                                               'smoothness_mean','area_se','area_worst','symmetry_mean'])
sns.scatterplot(x='concavity_mean', y='concave points_mean',hue='diagnosis', data = df_cancer)
# Normalize the dataset  -- This will maintain correlation but improve prediction

minValues = df_cancer.min()

range_df = (df_cancer - minValues).max()

df_cancer_scaled = (df_cancer - minValues)/range_df
# Check correlation between features

plt.figure(figsize=(30,30))                   

sns.set(font_scale=1.1)

sns.heatmap(df_cancer_scaled.corr(),annot=True)
# Prepare the dataset

X = df_cancer_scaled.drop(['diagnosis'],axis=1)

y = df_cancer['diagnosis']



# X      # Print X

# y      # Print y
# Split the dataset into training and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)
# Try various models



def models(x_train, y_train, x_test, y_test):

    MLA = [

        # Discriminant Analysis

        discriminant_analysis.LinearDiscriminantAnalysis(),

        discriminant_analysis.QuadraticDiscriminantAnalysis(),

        

        # Ensemble Methods

        ensemble.AdaBoostClassifier(),

        ensemble.BaggingClassifier(),

        ensemble.ExtraTreesClassifier(),

        ensemble.GradientBoostingClassifier(),

        ensemble.RandomForestClassifier(),



        # Gaussian Processes

        gaussian_process.GaussianProcessClassifier(),



        # GLM

        linear_model.LogisticRegressionCV(),

        linear_model.PassiveAggressiveClassifier(),

        linear_model.RidgeClassifierCV(),

        linear_model.SGDClassifier(),

        linear_model.Perceptron(),        



        # Navies Bayes

        naive_bayes.BernoulliNB(),

        naive_bayes.GaussianNB(),



        # Nearest Neighbor

        neighbors.KNeighborsClassifier(),



        # SVM

        svm.SVC(probability=True),

        svm.NuSVC(probability=True),

        svm.LinearSVC(),



        # Trees

        tree.DecisionTreeClassifier(),

        tree.ExtraTreeClassifier(),

        

        # XGBoost

        xgb.XGBClassifier()        

        

    ]

    

    MLA_compare = pd.DataFrame()

    for row_index, alg in enumerate(MLA):

        alg.fit(x_train, y_train)

        MLA_compare.loc[row_index, 'MLA Name'] = alg.__class__.__name__

        MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(x_train, y_train), 4)

        MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(x_test, y_test), 4)



    MLA_compare.sort_values(by=['MLA Test Accuracy'], ascending=False, inplace=True)

    print(MLA_compare)
# If you run this multiple time, you will notice the ranks keep changing. However, there are few algorithms who are consistent and hence they are recommended to be choosen.

models(X_train, y_train, X_test, y_test)        
# Let's now pickup few models and tune them

''' Here we tune below model:-

 1) SVC

 2) LinearSVC

 3) GaussianNB

 4) QuadraticDiscriminantAnalysis

 5) XGBClassifier

 6) DecisionTreeClassifier

 7) GradientBoostingClassifier

 8) BaggingClassifier

 

'''
def plot_confusion_matrix(y_test, y_predict):

    cm = confusion_matrix(y_test, y_pred = y_predict)

    sns.heatmap(cm, annot=True)

    

    print(classification_report(y_test,y_predict))
# Model 1:  SVC



# Base Model

model = svm.SVC()

model.fit(X_train, y_train)

print('Before tuning, parameters: ', model.get_params())

print("Before tuning, test set score: {:.5f} \n\n". format(model.score(X_test, y_test)))



plot_confusion_matrix(y_test, y_predict= model.predict(X_test))
# Accuracy of 99% is pretty good & detecting a benign person to have cancer is safe, so it is not reallly required to tune this model.

# However, to just list down the parameters in SVC, we have tried to tune this model, which gives us similar results



# Tune Model 

param_grid = {

    'C': [1,0.7], 

    #'cache_size': 200, 

    #'class_weight': None, 

    #'coef0': 0.0, 

    #'decision_function_shape': 'ovr', 

    'degree': [3,5], 

    #'gamma': 'auto_deprecated'

    'kernel': ['rbf','poly','linear'],

    # max_iter': -1, 

    #'probability': False, 

    #'random_state': None, 

    #'shrinking': True, 

    #'tol': 0.001 ,

    #'verbose': False

}



tune_model = GridSearchCV(svm.SVC(), param_grid = param_grid, scoring='precision', cv=5, refit = True, verbose=3)

tune_model.fit(X_train, y_train)



print('\nAfter tuning, parameters: ', tune_model.best_params_)

print("After tuning, test set score: {:.5f} \n\n".format(tune_model.score(X_test, y_test)))



plot_confusion_matrix(y_test, y_predict= tune_model.predict(X_test))



print('--'*10)
# Model 2:  Linear SVC



# Base Model

model = svm.LinearSVC()

model.fit(X_train, y_train)

print('Before tuning, parameters: ', model.get_params())

print("Before tuning, test set score: {:.5f} \n\n". format(model.score(X_test, y_test)))



plot_confusion_matrix(y_test, y_predict= model.predict(X_test))
# Tune Model 

param_grid = {

    'C':[1.5,1.4,1.2,1],

    #'class_weight': None,

    #'dual': True, 

    #'fit_intercept': True, 

    #'intercept_scaling': 1, 

    #'loss': 'squared_hinge'

    #'max_iter': 1000, 

    #'multi_class': 'ovr', 

    #'penalty': 'l2', 

    #'random_state': None,

    'tol': [0.00001, 1e-05, 1e-04],

    #'verbose': 0

}



tune_model = GridSearchCV(svm.LinearSVC(), param_grid = param_grid, scoring='recall', cv=5, refit = True, verbose=5)

tune_model.fit(X_train, y_train)



print('\nAfter tuning, parameters: ', tune_model.best_params_)

print("After tuning, test set score: {:.5f} \n\n".format(tune_model.score(X_test, y_test)))



plot_confusion_matrix(y_test, y_predict= tune_model.predict(X_test))



print('--'*10)
# Model 3:  GaussianProcessClassifier



# Base Model

model = gaussian_process.GaussianProcessClassifier()

model.fit(X_train, y_train)



print('Before tuning, parameters: ', model.get_params())

print("Before tuning, test set score: {:.5f} \n\n". format(model.score(X_test, y_test)))



plot_confusion_matrix(y_test, y_predict= model.predict(X_test))
# Tune Model 

param_grid = {

    #'copy_X_train': True, 

    #'kernel': None, 

    'max_iter_predict': [300], 

    'multi_class': ['one_vs_rest', 'one_vs_one'],

    #'n_jobs': None, 

    #'n_restarts_optimizer': 0, 

    #'optimizer': 'fmin_l_bfgs_b', 

    #'random_state': None, 

    'warm_start': [True]

}



#tune_model = gaussian_process.GaussianProcessClassifier(n_estimators=23, max_depth=5, learning_rate=0.1, max_features=25)

tune_model = GridSearchCV(gaussian_process.GaussianProcessClassifier(), param_grid=param_grid, scoring='roc_auc', cv=5)

tune_model.fit(X_train, y_train)



print('After tuning, parameters: ', tune_model.best_params_, sep="\n\n")

print("After tuning, test     set score: {:.5f}".format(tune_model.score(X_test, y_test)))



plot_confusion_matrix(y_test, y_predict= tune_model.predict(X_test))



print('--'*10)
# Model 4:  ExtraTreesClassifier



# Base Model

model = ensemble.ExtraTreesClassifier()

model.fit(X_train, y_train)



print('Before tuning, parameters: ', model.get_params())

print("Before tuning, test set score: {:.5f} \n\n". format(model.score(X_test, y_test)))



plot_confusion_matrix(y_test, y_predict= model.predict(X_test))
# Tune Model

param_grid = {

    # 'bootstrap': False, 

    # 'class_weight': None, 

    # 'criterion': 'gini', 

    # 'max_depth': None,

    # 'max_features': 'auto', 

    # 'max_leaf_nodes': None,

    # 'min_impurity_decrease': 0.0, 

    # 'min_impurity_split': None, 

    # 'min_samples_leaf': 1, 

    # 'min_samples_split': 2, 

    # 'min_weight_fraction_leaf': 0.0, 

    'n_estimators': [20, 30, 40, 50, 70], 

    # 'n_jobs': 1, 

    # 'oob_score': False, 

    # 'random_state': None, 

    #'verbose': [1], 

    'warm_start': [True]

}



tune_model = GridSearchCV(ensemble.ExtraTreesClassifier(), param_grid=param_grid, scoring='roc_auc', cv=4)

tune_model.fit(X_train, y_train)



print('After tuning, parameters: ', tune_model.best_params_, sep="\n\n")

print("After tuning, test set score: {:.5f}".format(tune_model.score(X_test, y_test)))



plot_confusion_matrix(y_test, y_predict= tune_model.predict(X_test))



print('-'*10)
# Model 5:  XGBClassifier



# Base Model

model = xgb.XGBClassifier()

model.fit(X_train, y_train)



print('Before tuning, parameters: ', model.get_params())

print("Before tuning, test set score: {:.5f} \n\n". format(model.score(X_test, y_test)))



plot_confusion_matrix(y_test, y_predict= model.predict(X_test))
# Tune Model



tune_model = xgb.XGBClassifier( max_depth=2, objective='binary:logistic', silent=False, seed = 333, eval_metric="mlogloss",

                      learning_rate=0.1, colsample_bytree = 0.5, subsample = 0.1, n_estimators=100, eta=0.1, reg_alpha =0.3, gamma=0, nround=100)



tune_model.fit(X_train, y_train)

y_predict = tune_model.predict(X_test)



print("After tuning, test set score: {:.5f}".format(tune_model.score(X_test, y_test)))



plot_confusion_matrix(y_test, y_predict= tune_model.predict(X_test))



print('--'*10)
# Model 6:  GaussianNB



# Base Model

model = naive_bayes.GaussianNB()

model.fit(X_train, y_train)



print('Before tuning, parameters: ', model.get_params())

print("Before tuning, test set score: {:.5f} \n\n". format(model.score(X_test, y_test)))



plot_confusion_matrix(y_test, y_predict= model.predict(X_test))
# Tune Model

""" NB makes very strong independent assumptions and unlinke other algorithms this algo doesn't have any hyperparameters except priors. So tuning this algorithm is really not recommended. 

Hereby, I have tried tunning the algo using GridSearchCV, but get the same confusion matrix.

"""

param_grid = {

    #'priors': None, 

    #'var_smoothing': 1e-09

}    



tune_model = GridSearchCV(naive_bayes.GaussianNB(), param_grid = param_grid, refit = True, scoring='roc_auc', cv=2, verbose=2)

#tune_model = naive_bayes.GaussianNB(priors=[0.4,0.6])

tune_model.fit(X_train, y_train)

y_predict = tune_model.predict(X_test)



                          

print('After tuning, parameters: ', tune_model.best_params_)

print("After tuning, test set score: {:.5f} \n".format(tune_model.score(X_test, y_test)))





plot_confusion_matrix(y_test, y_predict= tune_model.predict(X_test))



print('--'*10)
# Model 7:  BaggingClassifier



# Base Model

model = ensemble.BaggingClassifier()

model.fit(X_train, y_train)



print('Before tuning, parameters: ', model.get_params())

print("Before tuning, test set score: {:.5f} \n\n". format(model.score(X_test, y_test)))

print('Before tuning, parameters: ', model.bootstrap_features)



plot_confusion_matrix(y_test, y_predict= model.predict(X_test))
# Tune Model

param_grid = {

    #'base_estimator': None, 

    'bootstrap':[True] ,

    'bootstrap_features': [True],

    'max_features': [30, 25,28,27], 

    'max_samples': [90,50,70], 

    'n_estimators': [95,90,100],

    #'n_jobs': None, 

    #'oob_score': False,

    #'random_state': [45,30],

    #'verbose': 0

    'warm_start': [True, False]        

}



tune_model = GridSearchCV(ensemble.BaggingClassifier(), param_grid = param_grid, refit = True, scoring='roc_auc', cv=5)

#tune_model = ensemble.BaggingClassifier(max_features=30, max_samples=100, n_estimators=100)

tune_model.fit(X_train, y_train)

y_predict = tune_model.predict(X_test)



print('After tuning, parameters: ', tune_model.best_params_)

print("After tuning, test set score: {:.5f} \n".format(tune_model.score(X_test, y_test)))





plot_confusion_matrix(y_test, y_predict= tune_model.predict(X_test))



print('--'*10)
# Model 8:  QuadraticDiscriminantAnalysis



# Base Model

model = discriminant_analysis.QuadraticDiscriminantAnalysis()

model.fit(X_train, y_train)



print('Before tuning, parameters: ', model.get_params())

print("Before tuning, test set score: {:.5f} \n\n". format(model.score(X_test, y_test)))



plot_confusion_matrix(y_test, y_predict= model.predict(X_test))

        
# Tune Model

param_grid = {

    'priors': 2,

    'reg_param': 1, 

    'store_covariance': True, 

    #'store_covariances': None, 

    #'tol': 0.0001

}



tune_model = discriminant_analysis.QuadraticDiscriminantAnalysis(priors=2, reg_param =1, store_covariance =False)

tune_model.fit(X_train, y_train)

print('/nAfter tuning, parameters: ', tune_model.get_params())

print("After tuning, test set score: {:.5f} \n".format(tune_model.score(X_test, y_test)))



plot_confusion_matrix(y_test, y_predict= tune_model.predict(X_test))



print('--'*10)