# here we will import the libraries used for machine learning

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import randint

import pandas as pd # data processing, CSV file I/O, data manipulation 

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph. 

from pandas import set_option

plt.style.use('ggplot') # nice plots



from sklearn.model_selection import train_test_split # to split the data into two parts

from sklearn.linear_model import LogisticRegression # to apply the Logistic regression

from sklearn.feature_selection import RFE

from sklearn.model_selection import KFold # for cross validation

from sklearn.model_selection import GridSearchCV # for tuning parameter

from sklearn.model_selection import RandomizedSearchCV  # Randomized search on hyper parameters.

from sklearn.preprocessing import StandardScaler # for normalization

from sklearn.pipeline import Pipeline 

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn import metrics # for the check the error and accuracy of the model

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)



import os

#print(os.listdir("../input"))
data = pd.read_csv('../input/UCI_Credit_Card.csv')

data.sample(5)
data.rename(columns={"default.payment.next.month": "Default"}, inplace=True)

data.drop('ID', axis = 1, inplace =True) # drop column "ID"

data.info()
# Separating features and target

y = data.Default     # target default=1 or non-default=0

features = data.drop('Default', axis = 1, inplace = False)
data['EDUCATION'].unique()
data['EDUCATION']=np.where(data['EDUCATION'] == 5, 4, data['EDUCATION'])

data['EDUCATION']=np.where(data['EDUCATION'] == 6, 4, data['EDUCATION'])

data['EDUCATION']=np.where(data['EDUCATION'] == 0, 4, data['EDUCATION'])
data['EDUCATION'].unique()
data['MARRIAGE'].unique()
data['MARRIAGE']=np.where(data['MARRIAGE'] == 0, 3, data['MARRIAGE'])

data['MARRIAGE'].unique()
# The frequency of defaults

yes = data.Default.sum()

no = len(data)-yes



# Percentage

yes_perc = round(yes/len(data)*100, 1)

no_perc = round(no/len(data)*100, 1)



import sys 

plt.figure(figsize=(7,4))

sns.set_context('notebook', font_scale=1.2)

sns.countplot('Default',data=data, palette="Blues")

plt.annotate('Non-default: {}'.format(no), xy=(-0.3, 15000), xytext=(-0.3, 3000), size=12)

plt.annotate('Default: {}'.format(yes), xy=(0.7, 15000), xytext=(0.7, 3000), size=12)

plt.annotate(str(no_perc)+" %", xy=(-0.3, 15000), xytext=(-0.1, 8000), size=12)

plt.annotate(str(yes_perc)+" %", xy=(0.7, 15000), xytext=(0.9, 8000), size=12)

plt.title('COUNT OF CREDIT CARDS', size=14)

#Removing the frame

plt.box(False);
set_option('display.width', 100)

set_option('precision', 2)



print("SUMMARY STATISTICS OF NUMERIC COLUMNS")

print()

print(data.describe().T)
# Creating a new dataframe with categorical variables

subset = data[['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 

               'PAY_5', 'PAY_6', 'Default']]



f, axes = plt.subplots(3, 3, figsize=(20, 15), facecolor='white')

f.suptitle('FREQUENCY OF CATEGORICAL VARIABLES (BY TARGET)')

ax1 = sns.countplot(x="SEX", hue="Default", data=subset, palette="Blues", ax=axes[0,0])

ax2 = sns.countplot(x="EDUCATION", hue="Default", data=subset, palette="Blues",ax=axes[0,1])

ax3 = sns.countplot(x="MARRIAGE", hue="Default", data=subset, palette="Blues",ax=axes[0,2])

ax4 = sns.countplot(x="PAY_0", hue="Default", data=subset, palette="Blues", ax=axes[1,0])

ax5 = sns.countplot(x="PAY_2", hue="Default", data=subset, palette="Blues", ax=axes[1,1])

ax6 = sns.countplot(x="PAY_3", hue="Default", data=subset, palette="Blues", ax=axes[1,2])

ax7 = sns.countplot(x="PAY_4", hue="Default", data=subset, palette="Blues", ax=axes[2,0])

ax8 = sns.countplot(x="PAY_5", hue="Default", data=subset, palette="Blues", ax=axes[2,1])

ax9 = sns.countplot(x="PAY_6", hue="Default", data=subset, palette="Blues", ax=axes[2,2]);
x1 = list(data[data['Default'] == 1]['LIMIT_BAL'])

x2 = list(data[data['Default'] == 0]['LIMIT_BAL'])



plt.figure(figsize=(12,4))

sns.set_context('notebook', font_scale=1.2)

#sns.set_color_codes("pastel")

plt.hist([x1, x2], bins = 40, normed=False, color=['steelblue', 'lightblue'])

plt.xlim([0,600000])

plt.legend(['Yes', 'No'], title = 'Default', loc='upper right', facecolor='white')

plt.xlabel('Limit Balance (NT dollar)')

plt.ylabel('Frequency')

plt.title('LIMIT BALANCE HISTOGRAM BY TYPE OF CREDIT CARD', SIZE=15)

plt.box(False)

plt.savefig('ImageName', format='png', dpi=200, transparent=True);
Repayment = data[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]



Repayment = pd.concat([y,Repayment],axis=1)

Repayment = pd.melt(Repayment,id_vars="Default",

                    var_name="Repayment_Status",

                    value_name='value')



plt.figure(figsize=(10,5))

sns.set_context('notebook', font_scale=1.2)

sns.boxplot(y="value", x="Repayment_Status", hue="Default", data=Repayment, palette='Blues')

plt.legend(loc='best', title= 'Default', facecolor='white')

plt.xlim([-1.5,5.5])

plt.title('REPAYMENT STATUS - BOXPLOT', size=14)

plt.box(False)

plt.savefig('ImageName', format='png', dpi=200);
## data are distributed in a wide range (below), need to be normalizded.

plt.figure(figsize=(15,3))

ax= data.drop('Default', axis=1).boxplot(data.columns.name, rot=90)

outliers = dict(markerfacecolor='b', marker='p')

ax= features.boxplot(features.columns.name, rot=90, flierprops=outliers)

plt.xticks(size=12)

ax.set_ylim([-5000,100000])

plt.box(False);
stdX = (features - features.mean()) / (features.std())              # standardization

data_st = pd.concat([y,stdX.iloc[:,:]],axis=1)

data_st = pd.melt(data_st,id_vars="Default",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(20,10))

sns.set_context('notebook', font_scale=1)

sns.violinplot(y="value", x="features", hue="Default", data=data_st,split=True, 

               inner="quart", palette='Blues')

plt.legend(loc=4, title= 'Default', facecolor='white')

plt.ylim([-3,3])

plt.title('STANDARDIZED FEATURES - VIOLIN PLOT', size=14)

plt.box(False)

plt.savefig('ImageName', format='png', dpi=200, transparent=False);
#  looking at correlations matrix, defined via Pearson function  

corr = data.corr() # .corr is used to find corelation

f,ax = plt.subplots(figsize=(8, 7))

sns.heatmap(corr, cbar = True,  square = True, annot = False, fmt= '.1f', 

            xticklabels= True, yticklabels= True

            ,cmap="coolwarm", linewidths=.5, ax=ax)

plt.title('CORRELATION MATRIX - HEATMAP', size=18);
sns.lmplot(x='LIMIT_BAL', y= 'BILL_AMT2', data = data, hue ='Default', 

           palette='coolwarm')

plt.title('Linear Regression: distinguishing between Default and Non-default', size=16)





sns.lmplot(x='BILL_AMT1', y= 'BILL_AMT2', data = data, hue ='Default', 

           palette='coolwarm')

plt.title('Linear Regression: Cannot distinguish between Default and Non-default', size=16);



print('Uncorrelated data are poentially more useful: discrimentory!')
# Original dataset

X = data.drop('Default', axis=1)  

y = data['Default']



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)
# Dataset with standardized features

Xstd_train, Xstd_test, ystd_train, ystd_test = train_test_split(stdX,y, test_size=0.2, stratify=y,

                                                                random_state=42)
NUM_FEATURES = 3

model = LogisticRegression()

rfe_stand = RFE(model, NUM_FEATURES)

fit_stand = rfe_stand.fit(stdX, y)

#print("St Model Num Features:", fit_stand.n_features_)

#print("St Model Selected Features:", fit_stand.support_)

print("Std Model Feature Ranking:", fit_stand.ranking_)

# calculate the score for the selected features

score_stand = rfe_stand.score(stdX,y)

print("Standardized Model Score with selected features is: %f (%f)" % (score_stand.mean(), score_stand.std()))
feature_names = np.array(features.columns)

print('Most important features (RFE): %s'% feature_names[rfe_stand.support_])
# Dataset with three most important features

Ximp = stdX[['PAY_0', 'BILL_AMT1', 'PAY_AMT2']]

X_tr, X_t, y_tr, y_t = train_test_split(Ximp,y, test_size=0.2, stratify=y, random_state=42)
# Setup the hyperparameter grid, (not scaled data)

param_grid = {'C': np.logspace(-5, 8, 15)}



# Instantiate a logistic regression classifier

logreg = LogisticRegression()



# Instantiate the RandomizedSearchCV object

logreg_cv = RandomizedSearchCV(logreg,param_grid , cv=5, random_state=0)



# Fit it to the data

logreg_cv.fit(X_train, y_train)



# Print the tuned parameters and score

print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
LR = LogisticRegression(C=0.00005, random_state=0)

LR.fit(X_train, y_train)

y_pred = LR.predict(X_test)

print('Accuracy:', metrics.accuracy_score(y_pred,y_test))



## 5-fold cross-validation 

cv_scores =cross_val_score(LR, X, y, cv=5)



# Print the 5-fold cross-validation scores

print()

print(classification_report(y_test, y_pred))

print()

print("Average 5-Fold CV Score: {}".format(round(np.mean(cv_scores),4)),

      ", Standard deviation: {}".format(round(np.std(cv_scores),4)))



plt.figure(figsize=(4,3))

ConfMatrix = confusion_matrix(y_test,LR.predict(X_test))

sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 

            xticklabels = ['Non-default', 'Default'], 

            yticklabels = ['Non-default', 'Default'])

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title("Confusion Matrix - Logistic Regression");
# Instantiate the RandomizedSearchCV object:

logreg_cv_std = RandomizedSearchCV(logreg,param_grid , cv=5, random_state=0)



# Fit it to the standardized data

logreg_cv_std.fit(Xstd_train, ystd_train)



# Print the tuned parameters 

print("Tuned Logistic Regression Parameters with standardized features: {}".format(logreg_cv_std.best_params_)) 
LRS = LogisticRegression(C=3.73, random_state=0)

LRS.fit(Xstd_train, ystd_train)

y_pred = LRS.predict(Xstd_test)

print('Accuracy:', metrics.accuracy_score(y_pred,ystd_test))



## 5-fold cross-validation 

cv_scores =cross_val_score(LRS, stdX, y, cv=5)



# Print the 5-fold cross-validation scores

print()

print(classification_report(ystd_test, y_pred))

print()

print("Average 5-Fold CV Score: {}".format(round(np.mean(cv_scores),4)),

      ", Standard deviation: {}".format(round(np.std(cv_scores),4)))



plt.figure(figsize=(4,3))

ConfMatrix = confusion_matrix(ystd_test,LRS.predict(Xstd_test))

sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 

            xticklabels = ['Non-default', 'Default'], 

            yticklabels = ['Non-default', 'Default'])

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title("Confusion Matrix - Logistic Regression with standardized data");
LR_imp = LogisticRegression(C=3.73, random_state=0)

LR_imp.fit(X_tr, y_tr)

y_pred = LR_imp.predict(X_t)

print('Accuracy:', metrics.accuracy_score(y_pred,y_t))



## 5-fold cross-validation 

cv_scores =cross_val_score(LR_imp, Ximp, y, cv=5)



# Print the 5-fold cross-validation scores

print()

print(classification_report(y_t, y_pred))

print()

print("Average 5-Fold CV Score: {}".format(round(np.mean(cv_scores),4)),

      ", Standard deviation: {}".format(round(np.std(cv_scores),4)))



plt.figure(figsize=(4,3))

ConfMatrix = confusion_matrix(y_t,LR_imp.predict(X_t))

sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 

            xticklabels = ['Non-default', 'Default'], 

            yticklabels = ['Non-default', 'Default'])

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title("Confusion Matrix - Logistic Regression (most important features)");
# Setup the parameters and distributions to sample from: param_dist

param_dist = {"max_depth": [1,2,3,4,5,6,7,8,9],

              "max_features": [1,2,3,4,5,6,7,8,9],

              "min_samples_leaf": [1,2,3,4,5,6,7,8,9],

              "criterion": ["gini", "entropy"]}



# Instantiate a Decision Tree classifier: tree

tree = DecisionTreeClassifier()



# Instantiate the RandomizedSearchCV object: tree_cv

tree_cv = RandomizedSearchCV(tree, param_distributions=param_dist, cv=5, random_state=0)



# Fit it to the data

tree_cv.fit(X_train, y_train)



# Print the tuned parameters and score

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
Tree = DecisionTreeClassifier(criterion= 'gini', max_depth= 7, 

                                     max_features= 9, min_samples_leaf= 2, 

                                     random_state=0)

Tree.fit(X_train, y_train)

y_pred = Tree.predict(X_test)

print('Accuracy:', metrics.accuracy_score(y_pred,y_test))



## 5-fold cross-validation 

cv_scores =cross_val_score(Tree, X, y, cv=5)



# Print the 5-fold cross-validation scores

print()

print(classification_report(y_test, y_pred))

print()

print("Average 5-Fold CV Score: {}".format(round(np.mean(cv_scores),4)), 

      ", Standard deviation: {}".format(round(np.std(cv_scores),4)))



plt.figure(figsize=(4,3))

ConfMatrix = confusion_matrix(y_test,Tree.predict(X_test))

sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 

            xticklabels = ['Non-default', 'Default'], 

            yticklabels = ['Non-default', 'Default'])

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title("Confusion Matrix - Decision Tree");
# Create the random grid

param_dist = {'n_estimators': [50,100,150,200,250],

               "max_features": [1,2,3,4,5,6,7,8,9],

               'max_depth': [1,2,3,4,5,6,7,8,9],

               "criterion": ["gini", "entropy"]}



rf = RandomForestClassifier()



rf_cv = RandomizedSearchCV(rf, param_distributions = param_dist, 

                           cv = 5, random_state=0, n_jobs = -1)



rf_cv.fit(X, y)



print("Tuned Random Forest Parameters: %s" % (rf_cv.best_params_))
Ran = RandomForestClassifier(criterion= 'gini', max_depth= 6, 

                                     max_features= 5, n_estimators= 150, 

                                     random_state=0)

Ran.fit(X_train, y_train)

y_pred = Ran.predict(X_test)

print('Accuracy:', metrics.accuracy_score(y_pred,y_test))



## 5-fold cross-validation 

cv_scores =cross_val_score(Ran, X, y, cv=5)



# Print the 5-fold cross-validation scores

print()

print(classification_report(y_test, y_pred))

print()

print("Average 5-Fold CV Score: {}".format(round(np.mean(cv_scores),4)),

      ", Standard deviation: {}".format(round(np.std(cv_scores),4)))



plt.figure(figsize=(4,3))

ConfMatrix = confusion_matrix(y_test,Ran.predict(X_test))

sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 

            xticklabels = ['Non-default', 'Default'], 

            yticklabels = ['Non-default', 'Default'])

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title("Confusion Matrix - Random Forest");
y_pred_proba_RF = Ran.predict_proba(X_test)[::,1]

fpr1, tpr1, _ = metrics.roc_curve(y_test,  y_pred_proba_RF)

auc1 = metrics.roc_auc_score(y_test, y_pred_proba_RF)



y_pred_proba_DT = Tree.predict_proba(X_test)[::,1]

fpr2, tpr2, _ = metrics.roc_curve(y_test,  y_pred_proba_DT)

auc2 = metrics.roc_auc_score(y_test, y_pred_proba_DT)



y_pred_proba_LR = LR.predict_proba(X_test)[::,1]

fpr3, tpr3, _ = metrics.roc_curve(y_test,  y_pred_proba_LR)

auc3 = metrics.roc_auc_score(y_test, y_pred_proba_LR)



y_pred_proba_LRS = LRS.predict_proba(Xstd_test)[::,1]

fpr4, tpr4, _ = metrics.roc_curve(ystd_test,  y_pred_proba_LRS)

auc4 = metrics.roc_auc_score(ystd_test, y_pred_proba_LRS)



y_pred_proba_LRimp = LR_imp.predict_proba(X_t)[::,1]

fpr5, tpr5, _ = metrics.roc_curve(y_t,  y_pred_proba_LRimp)

auc5 = metrics.roc_auc_score(y_t, y_pred_proba_LRimp)



plt.figure(figsize=(10,7))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr1,tpr1,label="Random Forest, auc="+str(round(auc1,2)))

plt.plot(fpr2,tpr2,label="Decision Tree, auc="+str(round(auc2,2)))

plt.plot(fpr3,tpr3,label="LogReg, auc="+str(round(auc3,2)))

plt.plot(fpr4,tpr4,label="LogReg(std), auc="+str(round(auc4,2)))

plt.plot(fpr5,tpr5,label="LogReg(Std&Imp), auc="+str(round(auc5,2)))

plt.legend(loc=4, title='Models', facecolor='white')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC', size=15)

plt.box(False)

plt.savefig('ImageName', format='png', dpi=200, transparent=True);
# Append different models

models = []



# Logistic Regression

models.append(('LogReg',

               LogisticRegression(C=3.73, random_state=0),'none'))



# Logistic Regression (with standardized data)

models.append(('LogReg(Std)',

               LogisticRegression(C=3.73, random_state=0),'Std'))



# Logistic Regression with standardized and important features

models.append(('LogReg(Std&Imp)',

               LogisticRegression(C=3.73, random_state=0),'imp'))



# Decision Tree

models.append(('Decision Tree', 

              DecisionTreeClassifier(criterion= 'entropy', max_depth= 4, 

                                     max_features= 7, min_samples_leaf= 8, 

                                     random_state=0),'none'))



# Random Forest Classifier

models.append(('Random Forest', 

              RandomForestClassifier(criterion= 'gini', max_depth= 6, 

                                     max_features= 5, n_estimators= 150, 

                                     random_state=0), 'none'))



# Evaluate each model

results = []

names = []

scoring = 'accuracy'



for name, model, Std in models:

    if Std == 'Std':

        cv_results = cross_val_score(model, stdX, y, cv=5, scoring=scoring)

        results.append(cv_results)

        names.append(name)

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)       

    elif Std == 'none':

        cv_results = cross_val_score(model, X, y, cv=5, scoring=scoring)

        results.append(cv_results)

        names.append(name)

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)

    else:

        cv_results = cross_val_score(model, Ximp, y, cv=5, scoring=scoring)

        results.append(cv_results)

        names.append(name)

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)
from matplotlib import pyplot

# Plot all the accuracy results vs. each model 

#(model type on the x-axis and accuracy on the y-axis).

fig = pyplot.figure(figsize=(10,5))

sns.set_context('notebook', font_scale=1.1)

fig.suptitle('Algorithm Comparison - Accuracy (cv=5)')

ax = fig.add_subplot(111)

pyplot.boxplot(results, showmeans=True)

ax.set_xticklabels(names)

ax.set_ylabel('Accuracy')

ax.set_ylim([0.75,1])

plt.box(False)

plt.savefig('ImageName', format='png', dpi=200, transparent=True);
from astropy.table import Table, Column

data_rows = [('Logistic Regression', 'Standardized', 0.79, 0.81, 0.77),

              ('Logistic Regression', 'Important features', 0.79, 0.81, 0.78),

              ('Decision Tree', 'original', 0.80, 0.82, 0.79),

             ('Random Forest', 'original', 0.80, 0.82, 0.80)

            ]

t = Table(rows=data_rows, names=('Model', 'Data', 'Precision', 'Recall', 'F1'))

print(t)