### General libraries ###

import pandas as pd

from pandas.api.types import CategoricalDtype

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import seaborn as sns

import numpy as np

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



##################################



### ML Models ###

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.tree.export import export_text

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.preprocessing import StandardScaler, LabelEncoder



##################################



### Metrics ###

from sklearn import metrics

from sklearn.metrics import f1_score,confusion_matrix, mean_squared_error, mean_absolute_error, classification_report, roc_auc_score, roc_curve, precision_score, recall_score
# Load the data.

data=pd.read_csv('../input/bank-additional-full.csv')



# Information

data.info()
# Remove "duration" column as proposed above in the description.

data=data.drop(['duration'], axis=1)
# Value counts for categorical columns 

for i in data.columns[data.dtypes=='object']:

    print(data[i].value_counts(),'\n')
# Check for duplicate rows.

print(f"There are {data.duplicated().sum()} duplicate rows in the data set.")



# Remove duplicate rows.

data=data.drop_duplicates()

print("The duplicate rows were removed.")
# Check for null values.

print(f"There are {data.isna().any().sum()} cells with null values in the data set.")
# Information about the data set after the cleaning.

data.info()
# Label encoding on category columns.

le=LabelEncoder()

for i in data.columns[data.dtypes=='object']:

    data[i]=le.fit_transform(data[i])
# Correlation matrix.

plt.figure(figsize=(20,20))

sns.set(font_scale=1.1)

sns.heatmap(data.corr(),annot=True, cmap='rainbow',linewidth=0.5, fmt='.2f')

plt.title('Correlation matrix');
# Distinguish attribute columns and class column.

X=data[data.columns[:-1]]

y=data['y']
# Split to train and test sets. 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)
# Standardization

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Initialize a Logistic Regression estimator.

logreg=LogisticRegression(multi_class='auto', random_state=25, n_jobs=-1)



# Train the estimator.

logreg.fit(X_train,y_train)
# Make predictions.

log_pred=logreg.predict(X_test)



# CV score for MLP

cv_logreg=cross_val_score(logreg, X_train, y_train, cv=10).mean()
# Accuracy: 1 is perfect prediction.

print('Accuracy: %.3f' % logreg.score(X_test, y_test))



# Cross-Validation accuracy

print('Cross-validation accuracy: %0.3f' % cv_logreg)



# Precision

print('Precision: %.3f' % precision_score(y_test, log_pred))



# Recall

print('Recall: %.3f' % recall_score(y_test, log_pred))



# f1 score: best value at 1 (perfect precision and recall) and worst at 0.

print('F1 score: %.3f' % f1_score(y_test, log_pred))
# Predict probabilities for the test data.

logreg_probs = logreg.predict_proba(X_test)



# Keep Probabilities of the positive class only.

logreg_probs = logreg_probs[:, 1]



# Compute the AUC Score.

auc_logreg = roc_auc_score(y_test, logreg_probs)

print('AUC: %.2f' % auc_logreg)
# Plot confusion matrix for Logistic Regression.

logreg_matrix = confusion_matrix(y_test,log_pred)

sns.set(font_scale=1.3)

plt.subplots(figsize=(8, 8))

sns.heatmap(logreg_matrix, annot=True, cbar=False, cmap='twilight',linewidth=0.5,fmt="d")

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.title('Confusion Matrix for Logistic Regression');
# Hyperparameters to be checked.

parameters = {'C':[0.0001, 0.001, 0.01, 1, 0.1, 10, 100, 1000],

              'penalty':['none','l2'] ,

              'solver':['lbfgs','sag','saga','newton-cg']

             }



# Logistic Regression estimator.

default_logreg=LogisticRegression(multi_class='auto', random_state=25, n_jobs=-1)



# GridSearchCV estimator.

gs_logreg = GridSearchCV(default_logreg, parameters, cv=10, n_jobs=-1, verbose=1)



# Train the GridSearchCV estimator and search for the best parameters.

gs_logreg.fit(X_train,y_train)
# Make predictions with the best parameters.

gs_log_pred=gs_logreg.predict(X_test)
# Best parameters.

print("Best Logistic Regression Parameters: {}".format(gs_logreg.best_params_))



# Cross validation accuracy for the best parameters.

print('Cross-validation accuracy: %0.3f' % gs_logreg.best_score_)



# Accuracy: 1 is perfect prediction.

print('Accuracy: %0.3f' % (gs_logreg.score(X_test,y_test)))



# Precision

print('Precision: %.3f' % precision_score(y_test, gs_log_pred))



# Recall

print('Recall: %.3f' % recall_score(y_test, gs_log_pred))



# f1 score: best value at 1 (perfect precision and recall) and worst at 0.

print('F1 score: %.3f' % f1_score(y_test, gs_log_pred))
# Predict probabilities for the test data.

gs_logreg_probs = gs_logreg.predict_proba(X_test)



# Keep Probabilities of the positive class only.

gs_logreg_probs = gs_logreg_probs[:, 1]



# Compute the AUC Score.

gs_logreg_auc = roc_auc_score(y_test, gs_logreg_probs)

print('AUC: %.2f' % gs_logreg_auc)
# Print confusion matrix for Logistic regression.

gs_logreg_matrix = confusion_matrix(y_test,gs_log_pred)

sns.set(font_scale=1.3)

plt.subplots(figsize=(8, 8))

sns.heatmap(gs_logreg_matrix,annot=True, cbar=False, cmap='twilight',linewidth=0.5,fmt="d")

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.title('Confusion Matrix for GridSearchCV Logistic Regression');
# Get the ROC curves.

logreg_fpr, logreg_tpr, logreg_thresholds = roc_curve(y_test, logreg_probs)

gs_logreg_fpr, gs_logreg_tpr, gs_logreg_thresholds = roc_curve(y_test, gs_logreg_probs)



# Plot the ROC curves.

plt.figure(figsize=(8,8))

plt.plot(logreg_fpr, logreg_tpr, color='black', label='LogReg ROC (AUC= %0.2f)'% auc_logreg)

plt.plot(gs_logreg_fpr, gs_logreg_tpr, color='red', linestyle='--',label='GridSearch+LogReg ROC (AUC= %0.2f)'% gs_logreg_auc)

plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='random')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic (ROC) Curves')

plt.legend()

plt.show()
# Initialize a decision tree estimator.

tr = tree.DecisionTreeClassifier(max_depth=3,random_state=25)



# Train the estimator.

tr.fit(X_train, y_train)
# Plot the tree.

fig=plt.figure(figsize=(23,15))

tree.plot_tree(tr.fit(X_train, y_train),feature_names=X.columns,filled=True,rounded=True,fontsize=16);

plt.title('Decision Tree');
# Print the tree in a simplified version.

r = export_text(tr, feature_names=X.columns.tolist())

print(r)
# Make predictions.

tr_pred=tr.predict(X_test)



# CV score for Decision tree

cv_tr=cross_val_score(tr, X_train, y_train, cv=10).mean()
# Accuracy: 1 is perfect prediction.

print('Accuracy: %.3f' % tr.score(X_test, y_test))



# Cross-Validation accuracy

print('Cross-validation accuracy: %0.3f' % cv_tr)



# Precision

print('Precision: %.3f' % precision_score(y_test, tr_pred))



# Recall

print('Precision: %.3f' % recall_score(y_test, tr_pred))



# f1 score: best value at 1 (perfect precision and recall) and worst at 0.

print('F1 score: %.3f' % f1_score(y_test, tr_pred))
# Predict propabilities for the test data.

tr_probs = tr.predict_proba(X_test)



# Keep Probabilities of the positive class only.

tr_probs = tr_probs[:, 1]



# Compute the AUC Score.

auc_tr = roc_auc_score(y_test, tr_probs)

print('AUC: %.2f' % auc_tr)
# Print confusion matrix for Decision tree.

tr_matrix = confusion_matrix(y_test,tr_pred)

sns.set(font_scale=1.3)

plt.subplots(figsize=(8,8))

sns.heatmap(tr_matrix,annot=True, cbar=False, cmap='twilight',linewidth=0.5,fmt="d")

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.title('Confusion Matrix for Decision tree');
# Hyperparameters to be checked.

parameters = {'criterion':['gini','entropy'],

              'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

             }



# MLP estimator.

default_tr = tree.DecisionTreeClassifier(random_state=25)



# GridSearchCV estimator.

gs_tree = GridSearchCV(default_tr, parameters, cv=10, n_jobs=-1,verbose=1)



# Train the GridSearchCV estimator and search for the best parameters.

gs_tree.fit(X_train,y_train)
# Make predictions with the best parameters.

gs_tree_pred=gs_tree.predict(X_test)
# Best parameters.

print("Best Decision tree Parameters: {}".format(gs_tree.best_params_))



# Cross validation accuracy for the best parameters.

print('Cross-validation accuracy: %0.3f' % gs_tree.best_score_)



# Accuracy: 1 is perfect prediction.

print('Accuracy: %0.3f' % (gs_tree.score(X_test,y_test)))



# Precision

print('Precision: %.3f' % precision_score(y_test, gs_tree_pred))



# Recall

print('Recall: %.3f' % recall_score(y_test, gs_tree_pred))



# f1 score: best value at 1 (perfect precision and recall) and worst at 0.

print('F1 score: %.3f' % f1_score(y_test, gs_tree_pred))
# Predict probabilities for the test data.

gs_tree_probs = gs_tree.predict_proba(X_test)



# Keep Probabilities of the positive class only.

gs_tree_probs = gs_tree_probs[:, 1]



# Compute the AUC Score.

gs_tree_auc = roc_auc_score(y_test, gs_tree_probs)

print('AUC: %.2f' % gs_tree_auc)
# Print confusion matrix for Decision tree.

gs_tr_matrix = confusion_matrix(y_test,gs_tree_pred)

sns.set(font_scale=1.3)

plt.subplots(figsize=(8, 8))

sns.heatmap(gs_tr_matrix,annot=True, cbar=False, cmap='twilight',linewidth=0.5,fmt="d")

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.title('Confusion Matrix for GridSearchCV Decision tree');
# Get the ROC Curves.

gs_tr_fpr, gs_tr_tpr, gs_tr_thresholds = roc_curve(y_test, gs_tree_probs)

tr_fpr, tr_tpr, tr_thresholds = roc_curve(y_test, tr_probs)



# Plot the ROC curves.

plt.figure(figsize=(8,8))

plt.plot(tr_fpr, tr_tpr, color='red', label='Decision tree ROC (AUC= %0.2f)'% auc_tr)

plt.plot(gs_tr_fpr, gs_tr_tpr, color='green', label='GridSearch+Decision tree ROC (AUC= %0.2f)'% gs_tree_auc)

plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='random')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic (ROC) Curves')

plt.legend()

plt.show()
# Initialize a Multi-layer Perceptron classifier.

mlp = MLPClassifier(hidden_layer_sizes=(100),max_iter=1000, random_state=25,shuffle=True,)



# Train the classifier.

mlp.fit(X_train, y_train)
# Make predictions.

mlp_pred = mlp.predict(X_test)



# CV score for MLP

cv_mlp=cross_val_score(mlp, X_train, y_train, cv=10).mean()
# Accuracy: 1 is perfect prediction.

print('Accuracy: %.3f' % mlp.score(X_test, y_test))



# Cross-Validation accuracy

print('Cross-validation accuracy: %0.3f' % cv_mlp)



# Precision

print('Precision: %.3f' % precision_score(y_test, mlp_pred))



# Recall

print('Recall: %.3f' % recall_score(y_test, mlp_pred))



# f1 score: best value at 1 (perfect precision and recall) and worst at 0.

print('F1 score: %.3f' % f1_score(y_test, mlp_pred))
# Predict probabilities for the test data.

mlp_probs = mlp.predict_proba(X_test)



# Keep probabilities of the positive class only.

mlp_probs = mlp_probs[:, 1]



# Compute the AUC Score.

auc_mlp = roc_auc_score(y_test, mlp_probs)

print('AUC: %.2f' % auc_mlp)
# Print confusion matrix for Multi-layer Perceptron.

matrix = confusion_matrix(y_test,mlp_pred)

sns.set(font_scale=1.3)

plt.subplots(figsize=(8, 8))

sns.heatmap(matrix,annot=True, cbar=False, cmap='twilight',linewidth=0.5,fmt="d")

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.title('Confusion Matrix for Multi-layer Perceptron');
# Hyperparameters to be checked.

parameters = {'activation':['logistic','relu'],'solver': ['lbfgs','adam','sgd'],

              'alpha':10.0 ** -np.arange(1,4),

              'hidden_layer_sizes':[(20),(4),(10),(14,2),(4,1),(10,5),(11,3)]}



# Decision tree estimator.

default_mlp = MLPClassifier(random_state=25)



# GridSearchCV estimator.

gs_mlp = GridSearchCV(default_mlp, parameters, cv=10, n_jobs=-1,verbose=1)



# Train the GridSearchCV estimator and search for the best parameters.

gs_mlp.fit(X_train,y_train)
# Make predictions with the best parameters.

gs_mlp_pred=gs_mlp.predict(X_test)
# Best parameters.

print("Best MLP Parameters: {}".format(gs_mlp.best_params_))



# Cross validation accuracy for the best parameters.

print('Cross-validation accuracy: %0.3f' % gs_mlp.best_score_)



# Accuracy: 1 is perfect prediction.

print('Accuracy: %0.3f' % (gs_mlp.score(X_test,y_test)))



# Precision

print('Precision: %.3f' % precision_score(y_test, gs_mlp_pred))



# Recall

print('Recall: %.3f' % recall_score(y_test, gs_mlp_pred))



# f1 score: best value at 1 (perfect precision and recall) and worst at 0.

print('F1 score: %.3f' % f1_score(y_test, gs_mlp_pred))
# Predict probabilities for the test data.

gs_mlp_probs = gs_mlp.predict_proba(X_test)



# Keep Probabilities of the positive class only.

gs_mlp_probs = gs_mlp_probs[:, 1]



# Compute the AUC Score.

gs_mlp_auc = roc_auc_score(y_test, gs_mlp_probs)

print('AUC: %.2f' % gs_mlp_auc)
# Plot confusion matrix for GridSearchCV Multi-layer Perceptron.

matrix = confusion_matrix(y_test,gs_mlp_pred)

plt.figure(figsize=(8,8))

sns.heatmap(matrix,annot=True, cbar=False, cmap='twilight',linewidth=0.5,fmt="d")

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.title('Confusion Matrix for GridSearchCV Multi-layer Perceptron');
# Get the ROC curves.

gs_mlp_fpr, gs_mlp_tpr,gs_mlp_thresholds = roc_curve(y_test, gs_mlp_probs)

mlp_fpr, mlp_tpr, mlp_thresholds = roc_curve(y_test, mlp_probs)



# Plot the ROC curve.

plt.figure(figsize=(8,8))

plt.plot(mlp_fpr, mlp_tpr, color='red', label='MLP ROC (AUC= %0.2f)'% auc_mlp)

plt.plot(gs_mlp_fpr, gs_mlp_tpr, color='green', label='GridSearch+MLP ROC (AUC= %0.2f)'% gs_mlp_auc)

plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='random')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic (ROC) Curves')

plt.legend()

plt.show()
metrics=['Accuracy','CV accuracy','Precision','Recall','F1','ROC AUC']



# Plot metrics.

fig = go.Figure(data=[

    go.Bar(name='Logistic Regression', x=metrics,

           y=[logreg.score(X_test, y_test),cv_logreg,precision_score(y_test, log_pred),recall_score(y_test, log_pred),f1_score(y_test, log_pred),auc_logreg]),

    go.Bar(name='Decision tree', x=metrics,

           y=[tr.score(X_test, y_test),cv_tr,precision_score(y_test, tr_pred),recall_score(y_test, tr_pred),f1_score(y_test, tr_pred),auc_tr]),

    go.Bar(name='Neural Network', x=metrics,

           y=[mlp.score(X_test, y_test),cv_mlp,precision_score(y_test, mlp_pred),recall_score(y_test, mlp_pred),f1_score(y_test, mlp_pred),auc_mlp]),

    go.Bar(name='GridSearchCV+Logistic Regression',x=metrics,

           y=[gs_logreg.score(X_test,y_test),gs_logreg.best_score_,precision_score(y_test, gs_log_pred),recall_score(y_test, gs_log_pred),f1_score(y_test, gs_log_pred),gs_logreg_auc]),

    go.Bar(name='GridSearchCV+Decision tree',x=metrics,

           y=[gs_tree.score(X_test,y_test),gs_tree.best_score_,precision_score(y_test, gs_tree_pred),recall_score(y_test, gs_tree_pred), f1_score(y_test, gs_tree_pred),gs_tree_auc]),

    go.Bar(name='GridSearchCV+Neural Network', x=metrics, 

           y=[gs_mlp.score(X_test,y_test),gs_mlp.best_score_,precision_score(y_test, gs_mlp_pred),recall_score(y_test, gs_mlp_pred), f1_score(y_test, gs_mlp_pred),gs_mlp_auc])

    ])



fig.update_layout(title_text='Metrics for all models',

                  barmode='group',xaxis_tickangle=-45,bargroupgap=0.05)

fig.show()
# Plot the ROC curve.

plt.figure(figsize=(8,8))

plt.plot(gs_mlp_fpr, gs_mlp_tpr, color='green', label='GridSearch+MLP ROC (AUC= %0.2f)'% gs_mlp_auc)

plt.plot(gs_tr_fpr, gs_tr_tpr, color='orange', label='GridSearch+Decision tree ROC (AUC= %0.2f)'% gs_tree_auc)

plt.plot(gs_logreg_fpr, gs_logreg_tpr, color='red',label='GridSearch+LogReg ROC (AUC= %0.2f)'% gs_logreg_auc)

plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='random')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic (ROC) Curves for GridSearch')

plt.legend()

plt.show()
d={

'': ['Logistic Regression','GridSearchCV + Logistic Regression','Decision Tree','GridSearchCV + Decision Tree','Neural Network (MLP)','GridSearchCV + Neural Network (MLP)'],

'Accuracy': [logreg.score(X_test, y_test), gs_logreg.score(X_test,y_test),tr.score(X_test, y_test),gs_tree.score(X_test,y_test),mlp.score(X_test, y_test),gs_mlp.score(X_test, y_test)],

'CV Accuracy': [cv_logreg, gs_logreg.best_score_, cv_tr,gs_tree.best_score_,cv_mlp,gs_mlp.best_score_],

'Precision': [precision_score(y_test, log_pred), precision_score(y_test, gs_log_pred),precision_score(y_test, tr_pred),precision_score(y_test, gs_tree_pred),precision_score(y_test, mlp_pred),precision_score(y_test, gs_mlp_pred)],

'Recall': [recall_score(y_test, log_pred), recall_score(y_test, gs_log_pred),recall_score(y_test, tr_pred),recall_score(y_test, gs_tree_pred),recall_score(y_test, mlp_pred),recall_score(y_test, gs_mlp_pred)],

'F1': [f1_score(y_test, log_pred), f1_score(y_test, gs_log_pred),f1_score(y_test, tr_pred),f1_score(y_test, gs_tree_pred),f1_score(y_test, mlp_pred),f1_score(y_test, gs_mlp_pred)],

'ROC AUC': [auc_logreg, gs_logreg_auc, auc_tr, gs_tree_auc, auc_mlp, gs_mlp_auc]

}



results=pd.DataFrame(data=d).round(3).set_index('')

results