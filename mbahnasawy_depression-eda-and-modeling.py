# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import missingno as missing
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score ,auc, plot_roc_curve
from sklearn import svm
import sklearn.metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("../input/b_depressed.csv")

# preview the data
df.head()
df.info()
df.describe()
# lets check the no. of unique items present in the categorical column

df.select_dtypes('object').nunique()
plt.figure(figsize=(25,6))
plt.subplot(1, 4, 1)
sns.distplot(df['Age'])

plt.subplot(1, 4, 2)
sns.distplot(df['Number_children'])

plt.subplot(1, 4, 3)
sns.distplot(df['education_level'])

plt.subplot(1, 4, 4)
sns.distplot(df['no_lasting_investmen'])

plt.suptitle('Checking for Skewness', fontsize = 15)
plt.show()


plt.figure(figsize=(25,6))
plt.subplot(1, 4, 1)
sns.distplot(df['farm_expenses'])

plt.subplot(1, 4, 2)
sns.distplot(df['incoming_agricultural'])

plt.subplot(1, 4, 3)
sns.distplot(df['gained_asset'])

plt.subplot(1, 4, 4)
sns.distplot(df['durable_asset'])

plt.suptitle('Checking for Skewness', fontsize = 15)
plt.show()

plt.figure(figsize=(20,15))

plt.subplot(3,3,1)
sns.countplot(x='sex', hue='depressed', data=df)
plt.subplot(3,3,2)
sns.countplot(x='Married', hue='depressed', data=df)
plt.subplot(3,3,3)
sns.countplot(x='Number_children', hue='depressed', data=df)

plt.subplot(3,3,4)
sns.countplot(x='education_level', hue='depressed', data=df)
plt.subplot(3,3,5)
sns.countplot(x='total_members', hue='depressed', data=df)
plt.subplot(3,3,6)
sns.countplot(x='incoming_no_business', hue='depressed', data=df)

plt.subplot(3,3,7)
sns.countplot(x='incoming_salary', hue='depressed', data=df)
plt.subplot(3,3,8)
sns.countplot(x='incoming_own_farm', hue='depressed', data=df)
plt.subplot(3,3,9)
sns.countplot(x='incoming_business', hue='depressed', data=df)

plt.tight_layout()
plt.show()
dfPairplot = df.drop(['Survey_id' , 'Married' , 'Ville_id' , 'sex'  , 'gained_asset' , 'durable_asset' , 'save_asset' , 'living_expenses' , 'other_expenses' , 'incoming_salary' , 'incoming_own_farm' , 'incoming_business' , 'incoming_no_business' , 'incoming_agricultural' , 'farm_expenses' , 'labor_primary' , 'lasting_investment' , 'no_lasting_investmen'], axis=1)
dfPairplot.head()
plt.figure(figsize=(25,6))
sns.pairplot(data=dfPairplot,hue='depressed',plot_kws={'alpha':0.2})
plt.show()
dfPairplot = df.drop(['save_asset','Survey_id' , 'Ville_id' , 'sex' , 'Age' , 'Married' , 'Number_children' , 'education_level' , 'total_members' , 'living_expenses' , 'other_expenses' , 'incoming_salary' , 'incoming_own_farm' , 'incoming_business' , 'incoming_no_business' , 'incoming_agricultural' , 'farm_expenses' , 'labor_primary' , 'lasting_investment' , 'no_lasting_investmen'], axis=1)
dfPairplot.head()
plt.figure(figsize=(25,6))
sns.pairplot(data=dfPairplot,hue='depressed',plot_kws={'alpha':0.2})
plt.show()
dfPairplot = df.drop(['Survey_id' , 'Ville_id' , 'sex' , 'Age' , 'Married' , 'Number_children' , 'education_level' , 'total_members' , 'gained_asset' , 'durable_asset' , 'save_asset' , 'living_expenses' , 'other_expenses' , 'incoming_salary' , 'incoming_own_farm' , 'incoming_business' , 'incoming_no_business'  , 'labor_primary'     , 'no_lasting_investmen'], axis=1)
dfPairplot.head()
plt.figure(figsize=(25,6))
sns.pairplot(data=dfPairplot,hue='depressed',plot_kws={'alpha':0.2})
plt.show()
facet = sns.FacetGrid(df,hue="depressed", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0,df['Age'].max()))
facet.add_legend()

plt.show()
facet = sns.FacetGrid(df,hue="depressed", aspect=4)
facet.map(sns.kdeplot, 'gained_asset', shade=True)
facet.set(xlim=(0,df['gained_asset'].max()))
facet.add_legend()

plt.show()
facet = sns.FacetGrid(df,hue="depressed", aspect=4)
facet.map(sns.kdeplot, 'durable_asset', shade=True)
facet.set(xlim=(0,df['durable_asset'].max()))
facet.add_legend()

plt.show()
facet = sns.FacetGrid(df,hue="depressed", aspect=4)
facet.map(sns.kdeplot, 'incoming_agricultural', shade=True)
facet.set(xlim=(0,df['incoming_agricultural'].max()))
facet.add_legend()

plt.show()
facet = sns.FacetGrid(df,hue="depressed", aspect=4)
facet.map(sns.kdeplot, 'farm_expenses', shade=True)
facet.set(xlim=(0,df['farm_expenses'].max()))
facet.add_legend()

plt.show()
facet = sns.FacetGrid(df,hue="depressed", aspect=4)
facet.map(sns.kdeplot, 'lasting_investment', shade=True)
facet.set(xlim=(0,df['lasting_investment'].max()))
facet.add_legend()

plt.show()
dfCorr = df.drop(['no_lasting_investmen'], axis=1)
plt.subplots(figsize=(20,10)) 
sns.heatmap(dfCorr.corr(), annot = True, fmt = ".2f")
plt.show()
dfDrop = df.drop(['no_lasting_investmen', 'Survey_id', 'Ville_id', 'gained_asset', 'durable_asset', 'save_asset', 'farm_expenses', 'labor_primary', 'Number_children','lasting_investment','incoming_agricultural'], axis=1)
def plotLearningCurves(X_train, y_train, classifier, title):
    train_sizes, train_scores, test_scores = learning_curve(
            classifier, X_train, y_train, cv=5, scoring="accuracy")
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b" ,label="Training Error")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r" ,label="Cross Validation Error")
    
    plt.legend()
    plt.grid()
    plt.title(title, fontsize = 18, y = 1.03)
    plt.xlabel('Data Size', fontsize = 14)
    plt.ylabel('Error', fontsize = 14)
    plt.tight_layout()
def plotValidationCurves(X_train, y_train, classifier, param_name, param_range, title):
    train_scores, test_scores = validation_curve(
        classifier, X_train, y_train, param_name = param_name, param_range = param_range,
        cv=5, scoring="accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_scores_mean, 'o-', color="b" ,label="Training Error")
    plt.plot(param_range, test_scores_mean, 'o-', color="r" ,label="Cross Validation Error")

    plt.legend()
    plt.grid()
    plt.title(title, fontsize = 18, y = 1.03)
    plt.xlabel('Complexity', fontsize = 14)
    plt.ylabel('Error', fontsize = 14)
    plt.tight_layout()
def printConfusionMatrix(y_train, pred):
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("Classification Report:",)
    print (classification_report(y_test, pred))
    print("Accuracy:", accuracy_score(y_test, pred))
X = dfDrop.iloc[:, :-1].values
y = dfDrop.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
rf = RandomForestClassifier(n_estimators = 9,
                                    max_depth=3,
                                    min_samples_split=9,
                                    min_samples_leaf=5
                                   )
rf.fit(X_train, y_train)
rf_pred1 = rf.predict(X_test)
plt.figure(figsize = (16,5))
title = 'Random Forest Learning Curve 1'
plotLearningCurves(X_train, y_train, rf, title)
title = 'Random Forest Validation Curve 1'
param_name = 'n_estimators'
param_range = [4, 6, 9]
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, rf, param_name, param_range, title)
printConfusionMatrix(y_test, rf_pred1)
plot_roc_curve(rf, X_test, y_test)
plt.show()

rf = RandomForestClassifier(n_estimators = 9,
                                    max_depth=3,
                                    criterion='entropy',
                                    min_samples_split=9,
                                    min_samples_leaf=5
                                   )
rf.fit(X_train, y_train)
rf_pred2 = rf.predict(X_test)
plt.figure(figsize = (16,5))
title = 'Random Forest Learning Curve 2'
plotLearningCurves(X_train, y_train, rf, title)
plt.figure(figsize = (16,5))
title = 'Random Forest Validation Curve 2'
param_name = 'n_estimators'
param_range = [4, 6, 9]
plotValidationCurves(X_train, y_train, rf, param_name, param_range, title)
printConfusionMatrix(y_test, rf_pred2)
plot_roc_curve(rf, X_test, y_test)
plt.show()
rf = RandomForestClassifier(n_estimators = 9,
                                    max_depth=3,
                                    criterion='entropy',
                                    min_samples_split=10,
                                    min_samples_leaf=5
                                   )
rf.fit(X_train, y_train)
rf_pred3 = rf.predict(X_test)
plt.figure(figsize = (16,5))
title = 'Random Forest Learning Curve 3'
plotLearningCurves(X_train, y_train, rf, title)
title = 'Random Forest Validation Curve 3'
param_name = 'n_estimators'
param_range = [4, 6, 9]
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, rf, param_name, param_range, title)
printConfusionMatrix(y_test, rf_pred3)
plot_roc_curve(rf, X_test, y_test)
plt.show()
rf = RandomForestClassifier(n_estimators = 9,
                                    max_depth=5,
                                    criterion='entropy',
                                    min_samples_split=9,
                                    min_samples_leaf=10
                                   )
rf.fit(X_train, y_train)
rf_pred4 = rf.predict(X_test)
plt.figure(figsize = (16,5))
title = 'Random Forest Learning Curve 4'
plotLearningCurves(X_train, y_train, rf, title)
title = 'Random Forest Validation Curve 4'
param_name = 'n_estimators'
param_range = [4, 6, 9]
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, rf, param_name, param_range, title)
printConfusionMatrix(y_test, rf_pred4)
plot_roc_curve(rf, X_test, y_test)
plt.show()
rf = RandomForestClassifier(n_estimators = 9,
                                    max_depth=5,
                                    criterion='entropy',
                                    max_features='sqrt',
                                    min_samples_split=9,
                                    min_samples_leaf=5
                                   )
rf.fit(X_train, y_train)
rf_pred5 = rf.predict(X_test)
plt.figure(figsize = (16,5))
title = 'Random Forest Learning Curve 5'
plotLearningCurves(X_train, y_train, rf, title)

title = 'Random Forest Validation Curve 5'
param_name = 'n_estimators'
param_range = [4, 6, 9]
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, rf, param_name, param_range, title)


printConfusionMatrix(y_test, rf_pred5)

plot_roc_curve(rf, X_test, y_test)
plt.show()
Classifier = RandomForestClassifier()
grid_obj = GridSearchCV(Classifier,
                        {'n_estimators': [4, 6, 9],
                         'max_features': ['log2', 'sqrt','auto'],
                         'criterion': ['entropy', 'gini'],
                         'max_depth': [2, 3, 5, 8],
                         'min_samples_split': [2, 5, 8, 10],
                         'min_samples_leaf': [1, 3, 5]
                        },
                        scoring=make_scorer(accuracy_score))
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
Classifier = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
Classifier.fit(X_train, y_train)

predictions = Classifier.predict(X_test)

print("Best Params: " , grid_obj.best_estimator_)
print("Best Score: " , grid_obj.best_score_)
X = dfDrop.iloc[:, :-1].values
y = dfDrop.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Create KNN classifier
knn=KNeighborsClassifier(n_neighbors=3)
# Fit the classifier to the data
knn.fit(X_train,y_train)
#show first 5 model predictions on the test data
knn_pred1=knn.predict(X_test)
plt.figure(figsize=(16,5))
title='KNN Learning Curve 1'
plotLearningCurves(X_train,y_train,knn,title)
title = 'KNN Validation Curve 1' 
param_name = 'n_neighbors'
param_range = np.arange(1,9,2)
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, knn, param_name, param_range, title)
printConfusionMatrix(y_test, knn_pred1)
plot_roc_curve(knn, X_test, y_test)
plt.show()
# Create KNN classifier
knn=KNeighborsClassifier(n_neighbors=7)
# Fit the classifier to the data
knn.fit(X_train,y_train)
#show first 5 model predictions on the test data
knn_pred2=knn.predict(X_test)
plt.figure(figsize=(16,5))
title='KNN Learning Curve 2'
plotLearningCurves(X_train,y_train,knn,title)
title = 'KNN Validation Curve 2' 
param_name = 'n_neighbors'
param_range = np.arange(1,9,2)
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, knn, param_name, param_range, title)
printConfusionMatrix(y_test, knn_pred2)
plot_roc_curve(knn, X_test, y_test)
plt.show()
# Create KNN classifier
knn=KNeighborsClassifier(n_neighbors=10)
# Fit the classifier to the data
knn.fit(X_train,y_train)
#show first 5 model predictions on the test data
knn_pred3=knn.predict(X_test)
plt.figure(figsize=(16,5))
title='KNN Learning Curve 3'
plotLearningCurves(X_train,y_train,knn,title)
title = 'KNN Validation Curve 3' 
param_name = 'n_neighbors'
param_range = np.arange(1,9,2)
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, knn, param_name, param_range, title)
printConfusionMatrix(y_test, knn_pred3)
plot_roc_curve(knn, X_test, y_test)
plt.show()
# Create KNN classifier
knn=KNeighborsClassifier(n_neighbors=20)
# Fit the classifier to the data
knn.fit(X_train,y_train)
#show first 5 model predictions on the test data
knn_pred4=knn.predict(X_test)
plt.figure(figsize=(16,5))
title='KNN Learning Curve 4'
plotLearningCurves(X_train,y_train,knn,title)
title = 'KNN Validation Curve 4' 
param_name = 'n_neighbors'
param_range = np.arange(1,9,2)
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, knn, param_name, param_range, title)
printConfusionMatrix(y_test, knn_pred4)
plot_roc_curve(knn, X_test, y_test)
plt.show()
# Create KNN classifier
knn=KNeighborsClassifier(n_neighbors=17)
# Fit the classifier to the data
knn.fit(X_train,y_train)
#show first 5 model predictions on the test data
knn_pred5=knn.predict(X_test)
plt.figure(figsize=(16,5))
title='KNN Learning Curve 5'
plotLearningCurves(X_train,y_train,knn,title)
title = 'KNN Validation Curve 5' 
param_name = 'n_neighbors'
param_range = np.arange(1,9,2)
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, knn, param_name, param_range, title)
printConfusionMatrix(y_test, knn_pred5)
plot_roc_curve(knn, X_test, y_test)
plt.show()
#create new a knn model
knn2=KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid= {'n_neighbors': np.arange(1, 20)}
#use gridsearch to test all values for n_neighbors
knn_gscv=GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X, y)

print("Best Params: " , knn_gscv.best_estimator_)
print("Best Score: " , knn_gscv.best_score_)

# Instantiate the classfiers and make a list
classifiers = [RandomForestClassifier(),
               KNeighborsClassifier()]

result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])


# print('auc =', auc)
lr_fpr1, lr_tpr1, _ = roc_curve(y_test, rf_pred5)
lr_fpr2, lr_tpr2, _ = roc_curve(y_test,  knn_pred2)

# fpr , tpr, _= roc_curve(X_test, predict6_test)
auc1 = roc_auc_score(y_test, rf_pred5)
auc2 = roc_auc_score(y_test,  knn_pred2)
 
    
result_table = result_table.append({'classifiers':RandomForestClassifier.__class__.__name__,
                                     'fpr':lr_fpr1, 
                                     'tpr':lr_tpr1, 
                                     'auc':auc1}, ignore_index=True)

result_table = result_table.append({'classifiers':KNeighborsClassifier.__class__.__name__,
                                     'fpr':lr_fpr2, 
                                     'tpr':lr_tpr2, 
                                     'auc':auc2}, ignore_index=True)


fig = plt.figure(figsize=(8,6))

plt.plot(result_table.loc[0]['fpr'], 
         result_table.loc[0]['tpr'], 
         label="RandomForestClassifier, AUC={:.3f}".format( result_table.loc[0]['auc']))

plt.plot(result_table.loc[1]['fpr'], 
         result_table.loc[1]['tpr'], 
         label="KNeighborsClassifier, AUC={:.3f}".format( result_table.loc[1]['auc']))


plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()
    
!apt-get remove swig 
!apt-get install swig3.0 build-essential -y
!ln -s /usr/bin/swig3.0 /usr/bin/swig
!apt-get install build-essential
!pip install --upgrade setuptools
!pip install auto-sklearn
X = dfDrop.iloc[:, :-1].values
y = dfDrop.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import os  
import autosklearn.regression


automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_cv_example_tmp3',
    output_folder='/tmp/autosklearn_cv_example_out3',
    delete_tmp_folder_after_terminate=False,
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 5},
)

# fit() changes the data in place, but refit needs the original data. We
# therefore copy the data. In practice, one should reload the data
automl.fit(X_train.copy(), y_train.copy(), dataset_name='Students')
# During fit(), models are fit on individual cross-validation folds. To use
# all available data, we call refit() which trains all models in the
# final ensemble on the whole dataset.
automl.refit(X_train.copy(), y_train.copy())

print(automl.show_models())

predictions = automl.predict(X_test)
print("Accuracy as per AutoML: ", sklearn.metrics.accuracy_score(y_test, predictions))