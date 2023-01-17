import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

from sklearn import metrics



%matplotlib inline
pima = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

pima.head()
pima.info()
pima.shape
pima.describe()
pima.columns
pima.BloodPressure.unique()
pima.Age.unique()
pima.BMI.unique()
pima.SkinThickness.unique()
sns.boxplot(pima.Pregnancies)
plt.figure(figsize = (10,4))

plt.subplot(1,2,1)

plt.title('Pregnancies Count')

sns.countplot(pima['Pregnancies'])

plt.subplot(1,2,2)

plt.title('Pregnancies Count across diabetic and Non diabetics')

sns.countplot(pima['Pregnancies'], hue = pima['Outcome'])
plt.figure(figsize = (25,10))

plt.subplot(2,1,1)

plt.title('Age Count')

sns.countplot(pima.sort_values(by= 'Age').Age)

plt.subplot(2,1,2)

plt.title('Age Count across diabetic and Non diabetics')

sns.countplot(pima.sort_values(by= 'Age').Age, hue = pima['Outcome'])
# School Holiday

plt.rcParams['figure.figsize'] = (10,7)

for i in [0,1] :

    data = pima[pima.Outcome == i]

    if(len(data) == 0):

        continue

    plt.scatter(x = data.Age, y = data.BloodPressure, alpha = 0.6, label = i)

plt.title('Age vs BloodPressure')

plt.xlabel('Age')

plt.ylabel('BloodPressure')

plt.legend()
# School Holiday

plt.rcParams['figure.figsize'] = (10,7)

for i in [0,1] :

    data = pima[pima.Outcome == i]

    if(len(data) == 0):

        continue

    plt.scatter(x = data.Age, y = data.BMI, alpha = 0.6, label = i)

plt.title('Age vs BMI')

plt.xlabel('Age')

plt.ylabel('BMI')

plt.legend()
# School Holiday

plt.rcParams['figure.figsize'] = (10,7)

for i in [0,1] :

    data = pima[pima.Outcome == i]

    if(len(data) == 0):

        continue

    plt.scatter(x = data.Insulin, y = data.DiabetesPedigreeFunction, alpha = 0.6, label = i)

plt.title('PedigreeFunction vs Insulin')

plt.xlabel('PedigreeFunction')

plt.ylabel('Insulin')

plt.legend()
# School Holiday

plt.rcParams['figure.figsize'] = (10,7)

for i in [0,1] :

    data = pima[pima.Outcome == i]

    if(len(data) == 0):

        continue

    plt.scatter(x = data.BMI, y = data.SkinThickness, alpha = 0.6, label = i)

plt.title('BMI vs SkinThickness')

plt.xlabel('BMI')

plt.ylabel('SkinThickness')

plt.legend()
corrmat = pima.corr()

top_corr_feat = corrmat.index

sns.heatmap(pima[top_corr_feat].corr(), annot = True, cmap = 'BrBG')
print('No of entries for Glucose being 0: ',len(pima[pima.Glucose == 0]))

print('No of entries for BloodPressure being 0: ',len(pima[pima.BloodPressure == 0]))

print('No of entries for SkinThickness being 0: ',len(pima[pima.SkinThickness == 0]))

print('No of entries for Insulin being 0: ',len(pima[pima.Insulin == 0]))

print('No of entries for BMI being 0: ',len(pima[pima.BMI == 0]))
# Train Test Split

y = pima.pop('Outcome')

X = pima



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 100)
from sklearn.impute import SimpleImputer

fillvalues = SimpleImputer(missing_values = 0, strategy = 'mean')



X_train = fillvalues.fit_transform(X_train)

X_test = fillvalues.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(random_state = 100)

random_forest_model.fit(X_train, y_train)
y_train_predict = random_forest_model.predict(X_train)

y_train_pred = pd.DataFrame({'Diabetes':y_train.values, 'Diabetes_Pred':y_train_predict})



# Let's see the head

y_train_pred.head()
# Let's take a look at the confusion matrix again 

confusion = sklearn.metrics.confusion_matrix(y_train_pred.Diabetes, y_train_pred.Diabetes_Pred )

confusion
# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_depth': range(1, 40)}



# instantiate the model

rforest = RandomForestClassifier(criterion = "gini", 

                               random_state = 100)



# fit tree on training data

forest = GridSearchCV(rforest, parameters, 

                    cv=n_folds, 

                   scoring="accuracy", return_train_score=True)

forest.fit(X_train, y_train)
# scores of GridSearch CV

scores = forest.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with max_depth

plt.figure()

plt.plot(scores["param_max_depth"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_max_depth"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_depth")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# From above graph, the best max_depth value looks to 7. Lets also make use og the `Model.best_params_` to confirm our finding

forest.best_estimator_
forest1 = RandomForestClassifier(max_depth = 7, criterion = "gini", 

                               random_state = 100)

forest1.fit(X_train, y_train)
y_train_predict = forest1.predict(X_train)

y_train_pred = pd.DataFrame({'Diabetes':y_train.values, 'Diabetes_Pred':y_train_predict})



# Let's see the head

y_train_pred.head()
# Let's take a look at the confusion matrix again 

confusion = sklearn.metrics.confusion_matrix(y_train_pred.Diabetes, y_train_pred.Diabetes_Pred )

confusion
# classification metrics

from sklearn.metrics import classification_report,confusion_matrix

y_pred = forest1.predict(X_test)

print(classification_report(y_test, y_pred))



# confusion matrix on test data

print(confusion_matrix(y_test,y_pred))
# Lets also look at the roc auc. Note that for roc auc, we would need the predicted prob and not the labels.



y_pred_auc = forest1.predict_proba(X_test)

roc = metrics.roc_auc_score(y_test, y_pred_auc[:, 1])

print("AUC: %.2f%%" % (roc * 100.0))
from xgboost import XGBClassifier



# fit model on training data with default hyperparameters

model = XGBClassifier()

model.fit(X_train, y_train)



# make predictions for test data

# use predict_proba since we need probabilities to compute auc

y_pred = model.predict_proba(X_test)



# evaluate predictions

roc = metrics.roc_auc_score(y_test, y_pred[:, 1])

print("AUC: %.2f%%" % (roc * 100.0))
# hyperparameter tuning with XGBoost



# creating a KFold object 

folds = 3



# specify range of hyperparameters

param_grid = {'learning_rate': [0.2, 0.4, 0.6], 

             'subsample': [0.2, 0.4, 0.6, 0.8]}          





# specify model

xgb_model = XGBClassifier(max_depth=2, n_estimators=200)



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = xgb_model, 

                        param_grid = param_grid, 

                        scoring= 'roc_auc', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)      

# fit the model

model_cv.fit(X_train, y_train)
# cv results

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# convert parameters to int for plotting on x-axis

cv_results['param_learning_rate'] = cv_results['param_learning_rate'].astype('float')

# cv_results['param_max_depth'] = cv_results['param_max_depth'].astype('float')

cv_results.head()
# # plotting

plt.figure(figsize=(16,6))



param_grid = {'learning_rate': [0.2, 0.4, 0.6], 

             'subsample': [0.2, 0.4, 0.6, 0.8]} 





for n, subsample in enumerate(param_grid['subsample']):

    



    # subplot 1/n

    plt.subplot(1,len(param_grid['subsample']), n+1)

    df = cv_results[cv_results['param_subsample']==subsample]



    plt.plot(df["param_learning_rate"], df["mean_test_score"])

    plt.plot(df["param_learning_rate"], df["mean_train_score"])

    plt.xlabel('learning_rate')

    plt.ylabel('AUC')

    plt.title("subsample={0}".format(subsample))

    plt.ylim([0.60, 1])

    plt.legend(['test score', 'train score'], loc='upper left')

    plt.xscale('log')
# chosen hyperparameters

# 'objective':'binary:logistic' outputs probability rather than label, which we need for auc



# Lets choose the best params,

params = {'learning_rate': 0.2,

          'max_depth': 2, 

          'n_estimators':200,

          'subsample':0.4,

         'objective':'binary:logistic'}



# fit model on training data

model = XGBClassifier(params = params)

model.fit(X_train, y_train)
# classification metrics

from sklearn.metrics import classification_report,confusion_matrix

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))



# confusion matrix on test data

print(confusion_matrix(y_test,y_pred))
# predict the prob. rather than label, since we want to compute roc auc.

y_pred = model.predict_proba(X_test)



# roc_auc

auc = sklearn.metrics.roc_auc_score(y_test, y_pred[:, 1])

auc