#loading libraries



import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, make_scorer, confusion_matrix, classification_report, precision_recall_curve, plot_precision_recall_curve, average_precision_score, auc

from sklearn.model_selection import train_test_split

import seaborn as sns

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

import xgboost as xgb

import shap

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#loading the data into a dataframe

credit_df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
#preview of the first 10 rows of data

credit_df.head(10)
#displaying descriptive statistics

credit_df.describe()
#exploring datatypes and count of non-NULL rows for each feature

credit_df.info()
#checking for duplicated observations

credit_df.duplicated().value_counts()
#dropping duplicated observations

credit_df = credit_df.drop_duplicates()
#defining independent (X) and dependent (Y) variables from dataframe

X = credit_df.drop(columns = 'Class')

Y = credit_df['Class'].values
#splitting a testing set from the data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, stratify = Y, random_state = 42)

#splitting a validation set from the training set to tune parameters

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.20, stratify = Y_train, random_state = 42)
#creating a scorer from the f1-score metric

f1_scorer = make_scorer(f1_score)
# defining the space for hyperparameter tuning

space = {'eta': hp.uniform("eta", 0.1, 1),

        'max_depth': hp.quniform("max_depth", 3, 18, 1),

        'gamma': hp.uniform ('gamma', 1,9),

        'reg_alpha' : hp.quniform('reg_alpha', 50, 200, 1),

        'reg_lambda' : hp.uniform('reg_lambda', 0, 1),

        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),

        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),

        'n_estimators': hp.quniform('n_estimators', 100, 200, 10)

        }
#defining function to optimize

def hyperparameter_tuning(space):

    clf = xgb.XGBClassifier(n_estimators = int(space['n_estimators']),       #number of trees to use

                            eta = space['eta'],                              #learning rate

                            max_depth = int(space['max_depth']),             #depth of trees

                            gamma = space['gamma'],                          #loss reduction required to further partition tree

                            reg_alpha = int(space['reg_alpha']),             #L1 regularization for weights

                            reg_lambda = space['reg_lambda'],                #L2 regularization for weights

                            min_child_weight = space['min_child_weight'],    #minimum sum of instance weight needed in child

                            colsample_bytree = space['colsample_bytree'],    #ratio of column sampling for each tree

                            nthread = -1)                                    #number of parallel threads used

    

    evaluation = [(X_train, Y_train), (X_val, Y_val)]

    

    clf.fit(X_train, Y_train,

            eval_set = evaluation,

            early_stopping_rounds = 10,

            verbose = False)



    pred = clf.predict(X_val)

    pred = [1 if i>= 0.5 else 0 for i in pred]

    f1 = f1_score(Y_val, pred)

    print ("SCORE:", f1)

    return {'loss': -f1, 'status': STATUS_OK }
# run the hyper paramter tuning

trials = Trials()

best = fmin(fn = hyperparameter_tuning,

            space = space,

            algo = tpe.suggest,

            max_evals = 100,

            trials = trials)



print (best)
#plotting feature space and f1-scores for the different trials

parameters = space.keys()

cols = len(parameters)



f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20,5))

cmap = plt.cm.jet

for i, val in enumerate(parameters):

    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()

    ys = [-t['result']['loss'] for t in trials.trials]

    xs, ys = zip(*sorted(zip(xs, ys)))

    axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.25, c=cmap(float(i)/len(parameters)))

    axes[i].set_title(val)

    axes[i].grid()
#printing best model parameters

print(best)
#initializing XGBoost Classifier with best model parameters

best_clf = xgb.XGBClassifier(n_estimators = int(best['n_estimators']), 

                            eta = best['eta'], 

                            max_depth = int(best['max_depth']), 

                            gamma = best['gamma'], 

                            reg_alpha = int(best['reg_alpha']), 

                            min_child_weight = best['min_child_weight'], 

                            colsample_bytree = best['colsample_bytree'], 

                            nthread = -1)
#fitting XGBoost Classifier with best model parameters to training data

best_clf.fit(X_train, Y_train)
#using the model to predict on the test set

Y_pred = best_clf.predict(X_test)
#printing f1 score of test set predictions

print('The f1-score on the test data is: {0:.2f}'.format(f1_score(Y_test, Y_pred)))
#creating a confusion matrix and labels

cm = confusion_matrix(Y_test, Y_pred)

labels = ['Normal', 'Fraud']
#plotting the confusion matrix

sns.heatmap(cm, annot = True, xticklabels = labels, yticklabels = labels, fmt = 'd')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.title('Confusion Matrix for Credit Card Fraud Detection')
#printing classification report

print(classification_report(Y_test, Y_pred))
Y_score = best_clf.predict_proba(X_test)[:, 1]

average_precision = average_precision_score(Y_test, Y_score)

fig = plot_precision_recall_curve(best_clf, X_test, Y_test)

fig.ax_.set_title('Precision-Recall Curve: AP={0:.2f}'.format(average_precision))
#extracting the booster from model

booster = best_clf.get_booster()



# scoring features based on information gain

importance = booster.get_score(importance_type = "gain")



#rounding importances to 2 decimal places

for key in importance.keys():

    importance[key] = round(importance[key],2)



# plotting feature importances

ax = xgb.plot_importance(importance, importance_type='gain', show_values=True)

plt.title('Feature Importances (Gain)')

plt.show()
#obtaining SHAP values for XGBoost Model

explainer = shap.TreeExplainer(best_clf)

shap_values = explainer.shap_values(X_train)
#plotting SHAP Values of Feature Importances

shap.summary_plot(shap_values, X_train)