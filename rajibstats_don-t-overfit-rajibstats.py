# Import general useful packages

import numpy as np

import pandas as pd



# Import matplotlib for visualisations

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns

import scikitplot as skplt



# Import all machine learning algorithms

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xgb



# Import other useful subpackage

from sklearn import model_selection

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.model_selection import GridSearchCV , train_test_split , cross_val_score



#ignore warning messages 

import warnings

warnings.filterwarnings('ignore')
#Loadning trainning and testing data

training_data = pd.read_csv('/kaggle/input/dont-overfit-ii/train.csv')

testing_data = pd.read_csv('/kaggle/input/dont-overfit-ii/test.csv')



#/kaggle/input/dont-overfit-ii/train.csv

#/kaggle/input/dont-overfit-ii/test.csv

print(training_data.info())

print(testing_data.info())
# For training data

print("Training Data: {}".format(training_data.shape))

print("Null values present in training data: {}".format(training_data.isnull().values.any()))

  

# For testing data

print("Testing Data: {}".format(testing_data.shape))

print("Null values present in testing data: {}".format(testing_data.isnull().values.any()))



##Its showing NULL value presents in testing data only due to Target column. So we can ignore.
#training_data.head(5)

training_data.head(5)
#Distribution of Labels on train data

ax = sns.countplot(y=training_data['target'])

xx=training_data['target'].value_counts()

print(xx)

ax.set_title('Distribution of Labels on train data')

plt.show()



##the target in data set is imbalance. target is binary and has some disbalance: 36% of samples belong to 0 class;
# Univariate Analysis

training_data.describe()
#Distribution of means of all columns

training_data[training_data.columns[2:]].mean().plot('hist');

plt.title('Distribution of means of all columns');
#Correlation Check (Multicolinearity)

corrs = training_data.corr().abs().unstack().sort_values(kind="quicksort").reset_index()

corrs = corrs[corrs['level_0'] != corrs['level_1']]

corrs.tail(10)
#We can see that correlations between features are lower that 0.3 and the most correlated feature with target 

#has correlation of 0.33. So we have no highly correlated features which we could drop, 

#on the other hand we could drop some columns with have little correlation with the target.
#To split the Features and Target values of both Training and Test dataset

# Get X and y for training data

y_train = training_data['target']

X_train = training_data.drop(columns = ['target', 'id'])



# Get X and y for testing data

#y_test = testing_data['target'] # it is blank, but have to remove from X_test

X_test = testing_data.drop(columns = ['id'])
#K-fold stratified cross validation on Dataset1 to check over fitting...

n_fold = 10

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

repeated_folds = RepeatedStratifiedKFold(n_splits=20, n_repeats=20, random_state=42)
#Standardize features by removing the mean and scaling to unit variance

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.fit_transform(X_test)
#Cross Validation to reduce overfitting of the data

#Appling Logistic regression for binary classification

#for small samples set, So better to limited yourself to linear models.



random_state = 42

lr_clf = LogisticRegression(random_state = random_state, solver='liblinear', max_iter=1000)

param_grid = {'class_weight' : ['balanced', None], 

                'penalty' : ['l2','l1'],  

                'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



#Find best hyperparameters using GridSearchCV (roc_auc)

grid_lr = GridSearchCV(estimator = lr_clf, param_grid = param_grid , cv=folds, scoring = 'roc_auc', verbose = 1, n_jobs = -1)

grid_lr.fit(X_train,y_train)



print("Best Score:" + str(grid_lr.best_score_))

print("Best Parameters: " + str(grid_lr.best_params_))



#model with best parameters

model_lr=LogisticRegression(class_weight='balanced', solver='liblinear', penalty ='l1', C= 0.1, max_iter=10000)

model_lr.fit(X_train,y_train)

# To Cross validate and remodel it with less features

# cv - number of runs to find cross validated model



#Score of each cross validation score

scores_lr = model_selection.cross_val_score(model_lr,X_train,y_train,scoring="roc_auc",cv=10)

print(scores_lr) # moreover consistance score for flods

scores_lr = np.mean(scores_lr)

print('Mean cv Score',scores_lr)

print("Score on training data: " + str(model_lr.score(X_train,y_train)*100) + "%")
# tring to apply various classification algorithmns to check wheather gerring consistance CV scores

from sklearn.svm import SVC

svc = SVC(probability=True, gamma='scale')



parameter_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0],

                  'kernel': ['linear', 'poly', 'rbf'],

                 }



grid_search_svc = GridSearchCV(svc, param_grid=parameter_grid, cv=folds, scoring='roc_auc', n_jobs=-1)

grid_search_svc.fit(X_train, y_train)

print('Best score: {}'.format(grid_search_svc.best_score_))

print('Best parameters: {}'.format(grid_search_svc.best_params_))



model_svc = SVC(probability=True, gamma='scale', **grid_search_svc.best_params_)

model_svc.fit(X_train,y_train)



#Cross Validation to reduce overfitting of the data

scores_svc = model_selection.cross_val_score(model_svc,X_train,y_train,scoring="roc_auc",cv=10)

print(scores_svc)   #less consistance than LR

scores_svc = np.mean(scores_svc)

print('Mean cv Score',scores_svc)

print("Score on training data: " + str(model_svc.score(X_train,y_train)*100) + "%")
#lets check another one

from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier()



parameter_grid = {'n_neighbors': [2, 3, 5, 10, 20],

                  'weights': ['uniform', 'distance'],

                  'leaf_size': [5, 10, 30]

                 }



grid_search_knc = GridSearchCV(knc, param_grid=parameter_grid, cv=folds, scoring='roc_auc', n_jobs=-1)

grid_search_knc.fit(X_train, y_train)

print('Best score: {}'.format(grid_search_knc.best_score_))

print('Best parameters: {}'.format(grid_search_knc.best_params_))



model_knn = KNeighborsClassifier(**grid_search_knc.best_params_)

model_knn.fit(X_train,y_train)



#Cross Validation to reduce overfitting of the data

scores_knn = model_selection.cross_val_score(model_knn,X_train,y_train,scoring="roc_auc",cv=10)

print(scores_knn) #scores are less compare to above two algorithms

scores_knn = np.mean(scores_knn)

print('Mean cv Score',scores_knn)

print("Score on training data: " + str(model_knn.score(X_train,y_train)*100) + "%")
## for small samples set, using non linear models that RF, GBM or XGB, thats not gonna work. 

## But still appling ensemble learning algorithm (Bagging and Boosting)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()



parameter_grid = {'n_estimators': [10, 50, 100, 1000],

                  'max_depth': [None, 3, 5, 15]

                 }



grid_rf = GridSearchCV(rfc, param_grid=parameter_grid, cv=folds, scoring='roc_auc', n_jobs=-1)

grid_rf.fit(X_train, y_train)

print('Best score: {}'.format(grid_rf.best_score_))

print('Best parameters: {}'.format(grid_rf.best_params_))



model_rfc = RandomForestClassifier(**grid_rf.best_params_)

model_rfc.fit(X_train,y_train)



#Cross Validation to reduce overfitting of the data

scores_rf = model_selection.cross_val_score(model_rfc,X_train,y_train,scoring="roc_auc",cv=10)

print(scores_rf)

scores_rf = np.mean(scores_rf)

print('Mean cv Score',scores_rf)

print("Score on training data: " + str(model_rfc.score(X_train,y_train)*100) + "%")
from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier()



parameter_grid = {'n_estimators': [5, 10, 20, 50, 100],

                  'learning_rate': [0.001, 0.01, 0.1, 1.0, 10.0]

                 }



grid_search_abc = GridSearchCV(abc, param_grid=parameter_grid, cv=folds, scoring='roc_auc')

grid_search_abc.fit(X_train, y_train)

print('Best score: {}'.format(grid_search_abc.best_score_))

print('Best parameters: {}'.format(grid_search_abc.best_params_))



model_abc = AdaBoostClassifier(**grid_search_abc.best_params_)

model_abc.fit(X_train,y_train)



#Cross Validation to reduce overfitting of the data

scores_abc = model_selection.cross_val_score(model_abc,X_train,y_train,scoring="roc_auc",cv=10)   

print(scores_abc)

scores_abc = np.mean(scores_abc)

print('Mean cv Score',scores_abc)

print("Score on training data: " + str(model_abc.score(X_train,y_train)*100) + "%")
## CV_Score of various algorithms (Bar plot)

df = pd.DataFrame({'Algorithm':['LR','SVC','KNC','RFC', 'ABC'],

                   'Score':[scores_lr,scores_svc,scores_knn,scores_rf,scores_abc]})

print(df)

CV_Scores  = df['Score']

colors = cm.rainbow(np.linspace(0, 2, 9))

#labels = ['LogisticRegression','SVC','KNeighborsClassifier','RandomForestClassifier', 'AdaBoostClassifier']

labels = df['Algorithm']



plt.bar(labels,

        CV_Scores,

        color = colors)

plt.xlabel('Classifiers')

plt.ylabel('CV_Scores')

plt.title('CV_Score of various algorithms')
## CV_Score of various algorithms (Box-Plot)

plt.figure(figsize=(10, 5));

scores_df1 = pd.DataFrame({'LogisticRegression': [scores_lr]})

scores_df2 = pd.DataFrame({'AdaBoostClassifier': [scores_abc]})

scores_df3 = pd.DataFrame({'SVC': [scores_svc]})

scores_df4 = pd.DataFrame({'KNeighborsClassifier': [scores_knn]})

scores_df5 = pd.DataFrame({'RandomForestClassifier': [scores_rf]})

df = scores_df1.append([scores_df2, scores_df3, scores_df4, scores_df5],  ignore_index=True)

sns.boxplot(data=df);

plt.xticks(rotation=25);
#We can see that logistic regression is superior to most other models. 

#It seems that other models either overfit or can't work on this small dataset.
##Furure scope::

#  1. can check with other algorithms like Naive Bayes, extratree classifier, SGDClassifier etc  and compare CV scores

#  2. feature selection (variable reduction) 
## Now I am going to split trainning data set (250 observations) by 80/20 set 

#and training model on 80 % randon sample and predicting on 20% sample to check model accuracy
# Spliting data (80/20)

y = training_data['target']

X = training_data.drop(columns = ['target', 'id'])

from sklearn.model_selection import train_test_split

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, y, test_size = 0.2, random_state = 3)
# Applying various Classification algorithms without doing variable reductions

accuracy_scores = np.zeros(6)



# Support Vector Classifier

clf_svc = SVC().fit(X_train_new, y_train_new)

prediction1 = clf_svc.predict(X_test_new)

accuracy_scores[0] = accuracy_score(y_test_new, prediction1)*100

print('Support Vector Classifier accuracy: {}%'.format(accuracy_scores[0]))



# Logistic Regression

clf_lr = LogisticRegression(class_weight = 'balanced').fit(X_train_new, y_train_new)

prediction2 = clf_lr.predict(X_test_new)

accuracy_scores[1] = accuracy_score(y_test_new, prediction2)*100

print('Logistic Regression accuracy: {}%'.format(accuracy_scores[1]))



# K Nearest Neighbors

clf_knn = KNeighborsClassifier( ).fit(X_train_new, y_train_new)

prediction3 = clf_knn.predict(X_test_new)

accuracy_scores[2] = accuracy_score(y_test_new, prediction3)*100

print('K Nearest Neighbors Classifier accuracy: {}%'.format(accuracy_scores[2]))



# Random Forest

clf_rf = RandomForestClassifier(class_weight = 'balanced').fit(X_train_new, y_train_new)

prediction4 = clf_rf.predict(X_test_new)

accuracy_scores[3] = accuracy_score(y_test_new, prediction4)*100

print('Random Forest Classifier accuracy: {}%'.format(accuracy_scores[3]))



# Gradient Boosting

clf_gb = GradientBoostingClassifier().fit(X_train_new, y_train_new)

prediction5 = clf_gb.predict(X_test_new)

accuracy_scores[4] = accuracy_score(y_test_new, prediction5)*100

print('Gradient Boosting Classifier accuracy: {}%'.format(accuracy_scores[4]))



#XGBoosting

xgb_model = xgb.XGBClassifier() # 160/90 = 1.88

xgb_model.fit(X_train_new, y_train_new)

prediction6 = xgb_model.predict(X_test_new)

accuracy_scores[5] = accuracy_score(y_test_new, prediction6)*100

print('XGBoost Classifier accuracy: {}%'.format(accuracy_scores[5]))
#Here accuracy is higher for Gradient Boosting and XGB classifier 

#but based on bellow confusion matrix, Misclassification error is less for Logistic Regression

# So, Finally selecting Logistic Regression as a final model and going to use LR for prediction
# Confusion Matrix for above all models

from sklearn.metrics import confusion_matrix

conf1 = confusion_matrix(y_test_new, prediction1)

print(conf1)

conf2 = confusion_matrix(y_test_new, prediction2)

print(conf2)

conf3 = confusion_matrix(y_test_new, prediction3)

print(conf3)

conf4 = confusion_matrix(y_test_new, prediction4)

print(conf4)

conf5 = confusion_matrix(y_test_new, prediction5)

print(conf5)

conf6 = confusion_matrix(y_test_new, prediction6)

print(conf6)
# check validation statistics (Classification Summary)

print(classification_report(y_test_new, prediction2)) # from confusion matrix Logistic perform well without variable reduction

# Plot confusion Matrix

skplt.metrics.plot_confusion_matrix(y_test_new, prediction2, figsize=(10, 8))

plt.show()
# ROC Curves

y_probas = clf_lr.predict_proba(X_test_new)

skplt.metrics.plot_roc(y_test_new, y_probas, figsize=(10, 8))   # Plot ROC Curve

plt.show()



## test sample is too small so ROC curve is little bit as expected.
#Prediction on test data

prediction = model_lr.predict(X_test)

print(prediction)

prediction_prob = model_lr.predict_proba(X_test)

#print(prediction_prob)



predictions_df = pd.DataFrame(prediction)

predictions_df.rename(columns={0:'target'}, inplace=True)

result = pd.concat([testing_data['id'], predictions_df], axis=1)

print(result)



predictions_df_prob = pd.DataFrame(prediction_prob)

predictions_df_prob.rename(columns={0:'Probability for 0', 1: 'Probability for 1'}, inplace=True)

print(predictions_df_prob)

result3 = pd.concat([testing_data['id'], predictions_df_prob['Probability for 1']], axis=1)

result3.rename(columns={'Probability for 1':'target'}, inplace=True)

print(result3)

result.to_csv('sample submission.csv', index=False)
## End of code