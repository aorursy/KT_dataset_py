# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



import matplotlib.pyplot as plt

import seaborn as sns #for better and easier plots

%matplotlib inline



import os

print(os.listdir("../input"))





import warnings

warnings.filterwarnings(action="ignore")
data = pd.read_csv("../input/heart.csv")
print("Shape of the data: ", data.shape) #printing out the shape of data, 303x14
data.head()#let's use .head() and see what the data has for us.
#only shows null values. 

## shows the percentage of null values

def missing_values_calculate(trainset): 

    nulldata = (trainset.isnull().sum() / len(trainset)) * 100

    nulldata = nulldata.drop(nulldata[nulldata == 0].index).sort_values(ascending=False)

    ratio_missing_data = pd.DataFrame({'Ratio' : nulldata})

    return ratio_missing_data.head(30)
missing_values_calculate(data) #calling the function to check the data
data.columns
data.dtypes #let's take a look at the types of each column
sns.set(rc={'figure.figsize':(11.7,8.27)}) #setting the size of the figure to make it easier to read.

sns.countplot(data["age"]) #age seems to have a positive correlation to the chance of heart disease.
g = sns.FacetGrid(data, col="sex")

g.map(plt.hist, "age")
data['sex'].value_counts() #in this given data, we have a significantly assymetry in gender, way more mens.
g = sns.FacetGrid(data, col="target", hue="sex")

g.map(plt.scatter, "age", "chol", alpha=.7)

g.add_legend()
sns.set(rc={'figure.figsize':(11.7,8.27)})

g = sns.FacetGrid(data, col="target", height=4, aspect=2)

g.map(sns.barplot, "age", "cp")# I would like to see how chest pain is distributed in relationship to the target and age
corr = data.corr() #let's take a look at pearson's correlation

corr['target'].sort_values(ascending=False)
#we can use the following sklearn method to create a training and testing sample that is stratified.

from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) #20% for the testing sample

for train_index, test_index in split.split(data, data["sex"]):

    strat_train_set = data.loc[train_index]

    strat_test_set = data.loc[test_index]
print(strat_train_set.shape, strat_test_set.shape, data.shape) # let's check the shape of the datasets created
#now, I am gonna create Xtrain, Ytrain, Xtest, Ytest

Xtrain = strat_train_set.drop('target', axis=1).copy()

Ytrain = strat_train_set['target'].copy()

Xtest = strat_test_set.drop('target', axis=1).copy()

Ytest = strat_test_set['target'].copy()
print(Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape) #And again, let's check the final shape of our datasets.
#Gonna create a simple pipeline, for imputing future values, if NaN and standard scalling

#from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler



#this is a class to make selection of numerical atributes elegant and simple.   

num_pipeline = Pipeline([

        ("imputer", SimpleImputer(strategy="median")),

        ('scaler', StandardScaler())

    ])
#and finally, I am gonna create this function and use it to apply different model and see the results, accuracy, recal and F1 score

from sklearn.base import clone

from sklearn.metrics import mean_squared_error

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score



#Generic function for making a classification model and accessing performance:

def classification_model(model, X_train, y_train):

    #Fit the model:

    model.fit(X_train,y_train)

    n_cache = []

    

    train_predictions = model.predict(X_train)

    precision = precision_score(y_train, train_predictions)

    recall = recall_score(y_train, train_predictions)

    f1 = f1_score(y_train, train_predictions)

    

    print("Precision ", precision)

    print("Recall ", recall)

    print("F1 score ", f1)



    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        

    print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(cv_score)))
#Xtrain = num_pipeline.fit_transform(Xtrain) you might want to turn on the pipeline to see whether it fits better to our model

#Xtest_prepared = num_pipeline.fit_transform(Xtest)
#ok, let's take a look at our fist model.

#stochastic gradient descent SGD



##Note: some hyperparameters will have a different defaut value in future versions of Scikit-Learn, such as max_iter and tol. 

##To be future-proof, we explicitly set these hyperparameters to their future default values.



from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

classification_model(sgd_clf,Xtrain,Ytrain)
#Now, logistic regression...

from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression(solver="liblinear")

classification_model(log_reg, Xtrain, Ytrain)
#gonna use the decision tree without specifying any parameter, it's very likely to overfit the data, however, I am gonna use it

##to take a look at feature importances and plot a graph using it.

from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor()

classification_model(tree_reg, Xtrain, Ytrain)
# Plot feature importance

feature_importance = tree_reg.feature_importances_

# make importances relative to max importance

plt.figure(figsize=(40, 40))

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, strat_train_set.columns[sorted_idx], fontsize=30)

plt.xlabel('Relative Importance', fontsize=30)

plt.title('Variable Importance', fontsize=40)
#Gradient Boosting for classification.



from sklearn.ensemble import GradientBoostingClassifier

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.005, 'loss': 'deviance'}



clf = GradientBoostingClassifier(**params)

classification_model(clf, Xtrain, Ytrain)
from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(n_estimators=200, random_state=42)

classification_model(forest_clf, Xtrain, Ytrain)
#let's train a rather simple neural network

from sklearn.neural_network import MLPClassifier



mlp_clf = MLPClassifier(random_state=42)

classification_model(mlp_clf, Xtrain, Ytrain)
import xgboost #importing the package



extra_parameters = {'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 5,

                        "min_child_weight": 3, 'gamma': 0.2, 'subsample': 0.6, 'colsample_bytree': 1.0,

                        'objective': 'binary:logistic', 'scale_pos_weight': 1, 'seed': 27

                   }



xgb_clf = xgboost.XGBClassifier(random_state=42)

classification_model(xgb_clf, Xtrain, Ytrain)
#Gonna implement an ensemble model using the voting classifier

from sklearn.ensemble import VotingClassifier



#gonna create a list to help us put together the models and give it a name.



named_estimators = [

    #("sgd_clf", sgd_clf),

    #("random_forest_clf", forest_clf),

    #("gdb_clf", clf),

    #("mlp_clf", mlp_clf),

    ("logistic", log_reg),

    ("xboost", xgb_clf),

]
voting_clf = VotingClassifier(named_estimators)

#voting_clf.fit(Xtrain, Ytrain)

classification_model(voting_clf, Xtrain, Ytrain)
#importing relevant modules for this part

import eli5 #for purmutation importance

from eli5.sklearn import PermutationImportance

import shap #for SHAP values

#from pdpbox import pdp, info_plots #for partial plots
# I am gonna try first with de Gradient Boosting model, 

##And then with the Random forest model, just to see the difference, if any

perm = PermutationImportance(log_reg, random_state=1).fit(Xtest, Ytest) #gonna use logistic regression here, as the voting classifier seems to no work properly

eli5.show_weights(perm, feature_names = Xtest.columns.tolist())
explainer = shap.TreeExplainer(forest_clf) #using the random forest as shap seems to no support the voting classifier

shap_values = explainer.shap_values(Xtest)



shap.summary_plot(shap_values[1], Xtest, plot_type="bar")
def heart_disease_risk_factors(model, patient):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(patient)

    shap.initjs()

    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient)
#let's get a patience and take a look at the shap force plot.

p = 7 #helping variable, easily specify different patients.

data_for_prediction = Xtest.iloc[p,:].astype(float) #as I am using pipeline, I am using different datasets in order

heart_disease_risk_factors(forest_clf, data_for_prediction)#again, I am using the random forest model. 

#unfortunately, not all the models used can be used for the shap force plot