# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from imblearn import over_sampling as os

from imblearn import pipeline as pl

from imblearn.metrics import classification_report_imbalanced



# train test split

from sklearn.model_selection import train_test_split

# stand scaler

from sklearn.preprocessing import StandardScaler



# Area Under the Precision-Recall Curve 

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



# precision-recall curve and f1

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from sklearn.metrics import average_precision_score



# select k best

from sklearn.feature_selection import SelectKBest, f_classif



# models

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



# random seach

from sklearn.model_selection import RandomizedSearchCV



# confusion matrix

from sklearn.metrics import confusion_matrix

import itertools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import data

raw_data = pd.read_csv('../input/creditcard.csv', index_col=None)



df = pd.DataFrame(raw_data)



df.head()
df.info()
df.describe()
# view price data

df.Amount.hist();
# log transform price data

df['log_Amount'] = np.log1p(df.Amount)



# view log(price) data

df.log_Amount.hist(bins=20);
# view price data of fraudulent transactions

f, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)



sns.distplot(df.log_Amount.loc[df['Class']==1],

             kde=False, bins=20, ax=axes[0], axlabel='Fraud', color='r')

sns.distplot(df.log_Amount.loc[df['Class']==0],

             kde=False, bins=20, ax=axes[1], axlabel='Not Fraud', color='b')



plt.tight_layout()

plt.show()
# define data and target

data = df.drop('Class', axis=1)

target = df['Class']



# define training and test set

X_train, X_test, y_train, y_test = train_test_split(

    data, target, test_size=0.2, random_state=42, stratify=target)



# scale data

scaler = StandardScaler()

# fit transform X_train

X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

# transform X_test

X_test_scaled = scaler.transform(X_test)
LR = LogisticRegression()



# make pipeline

pipeline = pl.make_pipeline(os.SMOTE(random_state=42), LR)

# Train the classifier with balancing

pipeline.fit(X_train, y_train)



# Test the classifier and get the prediction

y_pred_bal = pipeline.predict(X_test)



# Show the classification report

print(classification_report_imbalanced(y_test, y_pred_bal))
# predict probabilities

probs = pipeline.predict_proba(X_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]



# calculate AUC

auc_score = roc_auc_score(y_test, probs)

print('AUC: %.3f' % auc_score)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, probs)



# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.', label='ROC Curve')

plt.title('Logistic Regression')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend()

plt.show()
# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, probs)



# calculate F1 score

f1 = f1_score(y_test, y_pred_bal)



# calculate precision-recall AUC

auc_score = auc(recall, precision)



# calculate average precision score

ap = average_precision_score(y_test, probs)

print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_score, ap))



# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')

plt.title('Logistic Regression')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.legend()

plt.show()
RFC = RandomForestClassifier(n_estimators=10)



# make pipeline

pipeline = pl.make_pipeline(os.SMOTE(random_state=42), RFC)

# Train the classifier with balancing

pipeline.fit(X_train, y_train)



# Test the classifier and get the prediction

y_pred_bal = pipeline.predict(X_test)



# Show the classification report

print(classification_report_imbalanced(y_test, y_pred_bal))
# predict probabilities

probs = pipeline.predict_proba(X_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]



# calculate AUC

auc_score = roc_auc_score(y_test, probs)

print('AUC: %.3f' % auc_score)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, probs)



# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.', label='ROC Curve')

plt.title('Random Forest Classifier')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend()

plt.show()
# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, probs)



# calculate F1 score

f1 = f1_score(y_test, y_pred_bal)



# calculate precision-recall AUC

auc_score = auc(recall, precision)



# calculate average precision score

ap = average_precision_score(y_test, probs)

print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_score, ap))



# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')

plt.title('Random Forest Classifier')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.legend()

plt.show()
CLF = GradientBoostingClassifier()



# make pipeline

pipeline = pl.make_pipeline(os.SMOTE(random_state=42), CLF)

# Train the classifier with balancing

pipeline.fit(X_train, y_train)



# Test the classifier and get the prediction

y_pred_bal = pipeline.predict(X_test)



# Show the classification report

print(classification_report_imbalanced(y_test, y_pred_bal))
# predict probabilities

probs = pipeline.predict_proba(X_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]



# calculate AUC

auc_score = roc_auc_score(y_test, probs)

print('AUC: %.3f' % auc_score)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, probs)



# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.', label='ROC Curve')

plt.title('Gradient Boosting Classifier')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend()

plt.show()
# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, probs)



# calculate F1 score

f1 = f1_score(y_test, y_pred_bal)



# calculate precision-recall AUC

auc_score = auc(recall, precision)



# calculate average precision score

ap = average_precision_score(y_test, probs)

print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_score, ap))



# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')

plt.title('Gradient Boosting Classifier')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.legend()

plt.show()
# import logistic regression from sklearn

LR_scaled = LogisticRegression()



# make pipeline

pipeline = pl.make_pipeline(os.SMOTE(random_state=42), LR_scaled)

# Train the classifier with balancing

pipeline.fit(X_train_scaled, y_train)



# Test the classifier and get the prediction

y_pred_bal = pipeline.predict(X_test_scaled)



# Show the classification report

print(classification_report_imbalanced(y_test, y_pred_bal))
# predict probabilities

probs = pipeline.predict_proba(X_test_scaled)

# keep probabilities for the positive outcome only

probs = probs[:, 1]



# calculate AUC

auc_score = roc_auc_score(y_test, probs)

print('AUC: %.3f' % auc_score)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, probs)



# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.', label='ROC Curve')

plt.title('Scaled Logistic Regression')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend()

plt.show()
# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, probs)



# calculate F1 score

f1 = f1_score(y_test, y_pred_bal)



# calculate precision-recall AUC

auc_score = auc(recall, precision)



# calculate average precision score

ap = average_precision_score(y_test, probs)

print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_score, ap))



# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')

plt.title('Scaled Logistic Regression')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.legend()

plt.show()
# import random forest classifier from sklearn

RFC_scaled = RandomForestClassifier(n_estimators=10)



# make pipeline

pipeline = pl.make_pipeline(os.SMOTE(random_state=42), RFC_scaled)

# Train the classifier with balancing

pipeline.fit(X_train_scaled, y_train)



# Test the classifier and get the prediction

y_pred_bal = pipeline.predict(X_test_scaled)



# Show the classification report

print(classification_report_imbalanced(y_test, y_pred_bal))
# predict probabilities

probs = pipeline.predict_proba(X_test_scaled)

# keep probabilities for the positive outcome only

probs = probs[:, 1]



# calculate AUC

auc_score = roc_auc_score(y_test, probs)

print('AUC: %.3f' % auc_score)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, probs)



# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.', label='ROC Curve')

plt.title('Scaled Random Forest Classifier')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend()

plt.show()
# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, probs)



# calculate F1 score

f1 = f1_score(y_test, y_pred_bal)



# calculate precision-recall AUC

auc_score = auc(recall, precision)



# calculate average precision score

ap = average_precision_score(y_test, probs)

print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_score, ap))



# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')

plt.title('Scaled Random Forest Classifier')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.legend()

plt.show()
# import gradient boosting classifier from sklearn

CLF_scaled = GradientBoostingClassifier()



# make pipeline

pipeline = pl.make_pipeline(os.SMOTE(random_state=42), CLF_scaled)

# Train the classifier with balancing

pipeline.fit(X_train_scaled, y_train)



# Test the classifier and get the prediction

y_pred_bal = pipeline.predict(X_test_scaled)



# Show the classification report

print(classification_report_imbalanced(y_test, y_pred_bal))
# predict probabilities

probs = pipeline.predict_proba(X_test_scaled)

# keep probabilities for the positive outcome only

probs = probs[:, 1]



# calculate AUC

auc_score = roc_auc_score(y_test, probs)

print('AUC: %.3f' % auc_score)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, probs)



# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.', label='ROC Curve')

plt.title('Scaled Gradient Boosting Classifier')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend()

plt.show()
# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, probs)



# calculate F1 score

f1 = f1_score(y_test, y_pred_bal)



# calculate precision-recall AUC

auc_score = auc(recall, precision)



# calculate average precision score

ap = average_precision_score(y_test, probs)

print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_score, ap))



# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')

plt.title('Scaled Gradient Boosting Classifier')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.legend()

plt.show()
# plot a heatmap

sns.heatmap(data.corr());
display(data.shape)

classifier = SelectKBest(f_classif, k=20).fit(data, target)
# Create new dataframe with only desired columns, or overwrite existing

feature_names = list(data.columns.values)



# Get columns to keep

mask = classifier.get_support() #list of booleans

new_features = data.columns[mask]



#        

data_new = data[new_features]

display(data_new.shape)

data_new.head()
# plot a heatmap

sns.heatmap(data_new.corr());
# Create correlation matrix

corr_matrix = data_new.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=4).astype(np.bool))



# Find index of feature columns with correlation greater than 0.90

to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]



display(data_new.shape)



# Drop correlated features 

for i in to_drop:

    data_new = data_new.drop(i, axis=1)



data_new.shape
# define training and test set with new features

X_train, X_test, y_train, y_test = train_test_split(

    data_new, target, test_size=0.2, random_state=42, stratify=target)
# sample training data

X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(

    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
# define our parameter ranges

n_estimators=[int(x) for x in np.linspace(start = 10, stop = 100, num = 3)]

criterion=['gini', 'entropy']

max_depth=[int(x) for x in np.linspace(start = 3, stop = 10, num = 3)]

max_depth.append(None)

min_samples_split=[int(x) for x in np.linspace(start = 2, stop = 5, num = 3)]

min_samples_leaf=[int(x) for x in np.linspace(start = 1, stop = 4, num = 3)]

max_features=['auto', 'sqrt']

# Create the random grid

param_grid = {'n_estimators':n_estimators,

              #'criterion':criterion,

              'max_depth':max_depth,

              'min_samples_split':min_samples_split,

              'min_samples_leaf':min_samples_leaf,

              'max_features':max_features

             }



print(param_grid)



# Initialize and fit the model.

model = RandomForestClassifier()

# parameter optimization

model = RandomizedSearchCV(model, param_grid, cv=3)

# make pipeline

pipeline = pl.make_pipeline(os.SMOTE(random_state=42), model)

# Train the classifier with balancing

pipeline.fit(X_train_sample, y_train_sample)



# get the best parameters

best_params = model.best_params_

print(best_params)
# refit model with best parameters

model_best = RandomForestClassifier(**best_params)

# make pipeline

pipeline_best = pl.make_pipeline(os.SMOTE(random_state=42), model_best)

# Train the classifier with balancing

pipeline_best.fit(X_train, y_train)

# Test the classifierand get the prediction

y_pred_bal = pipeline_best.predict(X_test)



# create a dictionary to hold our metrics

metrics_dict = {}
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        1#print('Confusion matrix, without normalization')



    #print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
#

y_pred_train = pipeline_best.predict(X_train)

cnf_matrix_tra = confusion_matrix(y_train, y_pred_train)



#

metrics_dict['Train recall metric'] = 100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])

print("Recall metric in the train dataset: {}%".format(metrics_dict['Train recall metric'])

     )



#

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')

plt.show()
cnf_matrix = confusion_matrix(y_test, y_pred_bal)



metrics_dict['Test recall metric']=100*cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])

print("Recall metric in the testing dataset: {}%".format(metrics_dict['Test recall metric'])

     )

#print("Precision metric in the testing dataset: {}%".format(

#    100*cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0]))

#     )

# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix , classes=class_names, title='Confusion matrix')

plt.show()
# predict probabilities

probs = pipeline.predict_proba(X_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]



# calculate AUC

auc_score = roc_auc_score(y_test, probs)

metrics_dict['ROC-AUC'] = auc_score

print('ROC-AUC: %.3f' % metrics_dict['ROC-AUC'])

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, probs)



# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.', label='ROC Curve')

plt.title('Logistic Regression')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend()

plt.show()
# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, probs)



# calculate F1 score

metrics_dict['F1 score'] = f1_score(y_test, y_pred_bal)



# calculate precision-recall AUC

metrics_dict['Precision-Recall AUC'] = auc(recall, precision)



# calculate average precision score

metrics_dict['Average Precision'] = average_precision_score(y_test, probs)

print('f1=%.3f auc=%.3f ap=%.3f' % (metrics_dict['F1 score'],

                                    metrics_dict['Precision-Recall AUC'],

                                    metrics_dict['Precision-Recall AUC']))



# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')

plt.title('Logistic Regression')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.legend()

plt.show()
# view metrics

for i in metrics_dict.items():

    print(i)