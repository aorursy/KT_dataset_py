### General libraries ###

import pandas as pd

from pandas.api.types import CategoricalDtype

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import seaborn as sns

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import graphviz 

from graphviz import Source

from IPython.display import SVG



##################################



### ML Models ###

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.tree.export import export_text

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split, cross_val_score



##################################



### Metrics ###

from sklearn import metrics

from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, classification_report
# Load the data.

file = '../input/car.csv'

data = pd.read_csv(file)



# Information

data.info()
# Shape of the data set.

print("The data set has {} rows and {} columns.".format(data.shape[0],data.shape[1]))
# Check for missing values.

data.isna().any()
# Check for duplicate rows.

data.duplicated().any()
# Checking the values from each column.

for col in data.columns:

    print("Column:", col)

    print(data[col].value_counts(),'\n')
# Plotting the values of each column.

for i in data.columns:

    labels = data[i].unique()

    values = data[i].value_counts()

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_layout(title=go.layout.Title(text='Value distribution for column: "{}"'.format(i),x=.5))

    fig.show()
# Create category types.

buying_type = CategoricalDtype(['low','med','high','vhigh'], ordered=True)

maint_type = CategoricalDtype(['low','med','high','vhigh'], ordered=True)

doors_type = CategoricalDtype(['2','3','4','5more'], ordered=True)

persons_type = CategoricalDtype(['2','4','more'], ordered=True)

lug_boot_type = CategoricalDtype(['small','med','big'], ordered=True)

safety_type = CategoricalDtype(['low','med','high'], ordered=True)

class_type = CategoricalDtype(['unacc','acc','good','vgood'], ordered=True)



# Convert all categorical values to category type.

data.buying = data.buying.astype(buying_type)

data.maint = data.maint.astype(maint_type)

data.doors = data.doors.astype(doors_type)

data.persons = data.persons.astype(persons_type)

data.lug_boot = data.lug_boot.astype(lug_boot_type)

data.safety = data.safety.astype(safety_type)

data.class_val = data.class_val.astype(class_type)
# Convert categories into integers for each column.

data.buying=data.buying.replace({'low':0, 'med':1, 'high':2, 'vhigh':3})

data.maint=data.maint.replace({'low':0, 'med':1, 'high':2, 'vhigh':3})

data.doors=data.doors.replace({'2':0, '3':1, '4':2, '5more':3})

data.persons=data.persons.replace({'2':0, '4':1, 'more':2})

data.lug_boot=data.lug_boot.replace({'small':0, 'med':1, 'big':2})

data.safety=data.safety.replace({'low':0, 'med':1, 'high':2})

data.class_val=data.class_val.replace({'unacc':0, 'acc':1, 'good':2, 'vgood':3})
# The data set after the conversion.

data.head()
plt.figure(figsize=(10,6))

sns.set(font_scale=1.2)

sns.heatmap(data.corr(),annot=True, cmap='rainbow',linewidth=0.5)

plt.title('Correlation matrix');
# Choose attribute columns and class column.

X=data[data.columns[:-1]]

y=data['class_val']
# Split to train and test sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize a Logistic Regression classifier.

logreg=LogisticRegression(solver='saga', multi_class='auto', random_state=42, n_jobs=-1)



# Train the classifier.

logreg.fit(X_train,y_train)
# Make predictions.

log_pred=logreg.predict(X_test)



# CV score

logreg_cv = cross_val_score(logreg,X_train,y_train,cv=10)
# The mean squared error (relative error).

print("Mean squared error (MSE): %.3f" % mean_squared_error(y_test, log_pred))



# Explained average absolute error (average error).

print("Mean absolute error (MAE): %.3f" % mean_absolute_error(y_test, log_pred))



# Explained variance score: 1 is perfect prediction.

print('Accuracy: %.3f' % logreg.score(X_test, y_test))



# CV Accuracy

print('CV Accuracy: %.3f' % logreg_cv.mean())
# Plot confusion matrix for Logistic regression.

logreg_matrix = confusion_matrix(y_test,log_pred)

plt.figure(figsize=(8,8))

sns.set(font_scale=1.4)

sns.heatmap(logreg_matrix,annot=True, cbar=False, cmap='twilight',linewidth=0.5,fmt="d")

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.title('Confusion Matrix for Logistic Regression');
# Hyperparameters to be checked.

parameters = {'C':[0.0001,0.001, 0.01, 1, 0.1, 10, 100, 1000],

              'penalty':['none','l2'],

              'solver':['lbfgs','sag','saga','newton-cg']

             }



# Logistic Regression classifier.

default_logreg=LogisticRegression(multi_class='auto', random_state=42, n_jobs=-1)



# GridSearchCV estimator.

gs_logreg = GridSearchCV(default_logreg, parameters, cv=10, verbose=1)



# Train the GridSearchCV estimator and search for the best parameters.

gs_logreg.fit(X_train,y_train)
# Make predictions with the best parameters.

gs_log_pred=gs_logreg.predict(X_test)
# Best parameters.

print("Best Logistic Regression Parameters: {}".format(gs_logreg.best_params_))



# The mean squared error (relative error).

print("Mean squared error (MSE): %.3f" % mean_squared_error(y_test, gs_log_pred))



# Explained average absolute error (average error).

print("Mean absolute error (MAE): %.3f" % mean_absolute_error(y_test, gs_log_pred))



# Cross validation accuracy for the best parameters.

print('CV Accuracy: %0.3f' % gs_logreg.best_score_)



# Accuracy: 1 is perfect prediction.

print('Accuracy: %0.3f' % (gs_logreg.score(X_test,y_test)))
# Plot confusion matrix for GridSearchCV Logistic regression.

gs_logreg_matrix = confusion_matrix(y_test,log_pred)

plt.figure(figsize=(8,8))

sns.set(font_scale=1.4)

sns.heatmap(gs_logreg_matrix,annot=True, cbar=False, cmap='twilight',linewidth=0.5,fmt="d")

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.title('Confusion Matrix \nfor GridSearchCV Logistic Regression');
# Initialize a decision tree estimator.

tr = tree.DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)



# Train the estimator.

tr.fit(X_train, y_train)
# Plot the tree.

dot_data = tree.export_graphviz(tr, out_file=None, feature_names=X.columns,class_names=['unacc', 'acc', 'good', 'vgood'], filled=True, rounded=True, special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
# Print the tree in a simplified version.

r = export_text(tr, feature_names=X.columns.tolist())

print(r)
# Make predictions.

tr_pred=tr.predict(X_test)



# CV score

tr_cv = cross_val_score(tr,X_train,y_train,cv=10)
# The mean squared error (relative error).

print("Mean squared error (MSE): %.3f" % mean_squared_error(y_test, tr_pred))



# Explained average absolute error (average error).

print("Mean absolute error (MAE): %.3f" % mean_absolute_error(y_test, tr_pred))



# Explained variance score: 1 is perfect prediction.

print('Accuracy: %.3f' % tr.score(X_test, y_test))



# CV Accuracy

print('CV Accuracy: %.3f' % tr_cv.mean())
# Print confusion matrix for Decision tree.

tr_matrix = confusion_matrix(y_test,tr_pred)

plt.figure(figsize=(8,8))

sns.set(font_scale=1.4)

sns.heatmap(tr_matrix,annot=True, cbar=False, cmap='twilight',linewidth=0.5,fmt="d")

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.title('Confusion Matrix for Decision tree');
# Hyperparameters to be checked.

parameters = {'criterion':['gini','entropy'],

              'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

             }



# Default Decision tree estimator.

default_tr = tree.DecisionTreeClassifier(random_state=42)



# GridSearchCV estimator.

gs_tree = GridSearchCV(default_tr, parameters, cv=10, n_jobs=-1,verbose=1)



# Train the GridSearchCV estimator and search for the best parameters.

gs_tree.fit(X_train,y_train)
# Make predictions with the best parameters.

gs_tree_pred=gs_tree.predict(X_test)
# Best parameters.

print("Best Decision tree Parameters: {}".format(gs_tree.best_params_))



# The mean squared error (relative error).

print("Mean squared error (MSE): %.3f" % mean_squared_error(y_test, gs_tree_pred))



# Explained average absolute error (average error).

print("Mean absolute error (MAE): %.3f" % mean_absolute_error(y_test, gs_tree_pred))



# Cross validation accuracy for the best parameters.

print('CV accuracy: %0.3f' % gs_tree.best_score_)



# Accuracy: 1 is perfect prediction.

print('Accuracy: %0.3f' % (gs_tree.score(X_test,y_test)))
# Print confusion matrix for GridSearchCV Decision tree.

gs_tr_matrix = confusion_matrix(y_test,gs_tree_pred)

plt.figure(figsize=(8,8))

sns.set(font_scale=1.4)

sns.heatmap(gs_tr_matrix,annot=True, cbar=False, cmap='twilight', linewidth=0.5, fmt="d")

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.title('Confusion Matrix for GridSearchCV Decision tree');
# Initialize a Multi-layer Perceptron classifier.

mlp = MLPClassifier(hidden_layer_sizes=(5),max_iter=1000, random_state=42, shuffle=True, verbose=False)



# Train the classifier.

mlp.fit(X_train, y_train)
# Make predictions.

mlp_pred = mlp.predict(X_test)



# CV score

mlp_cv = cross_val_score(mlp,X_train,y_train,cv=10)
# The mean squared error (relative error).

print("Mean squared error (MSE): %.3f" % mean_squared_error(y_test, mlp_pred))



# Explained average absolute error (average error).

print("Mean absolute error (MAE): %.3f" % mean_absolute_error(y_test, mlp_pred))



# Explained variance score: 1 is perfect prediction.

print('Accuracy: %.3f' % mlp.score(X_test, y_test))



# CV Accuracy

print('CV Accuracy: %.3f' % mlp_cv.mean())
# Plot confusion matrix for MLP.

mlp_matrix = confusion_matrix(y_test,mlp_pred)

plt.figure(figsize=(8,8))

sns.set(font_scale=1.4)

sns.heatmap(mlp_matrix,annot=True, cbar=False, cmap='twilight',linewidth=0.5,fmt="d")

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.title('Confusion Matrix for MLP');
# Hyperparameters to be checked.

parameters = {'activation':['logistic','tanh','relu'],

              'solver': ['lbfgs','adam','sgd'],

              'alpha':10.0 ** -np.arange(1,3),

              'hidden_layer_sizes':[(5),(100),(3),(4),(3,1),(5,3)]}



# MLP estimator.

default_mlp = MLPClassifier(random_state=42)



# GridSearchCV estimator.

gs_mlp = GridSearchCV(default_mlp, parameters, cv=10, n_jobs=-1,verbose=1)



# Train the GridSearchCV estimator and search for the best parameters.

gs_mlp.fit(X_train,y_train)
# Make predictions with the best parameters.

gs_mlp_pred=gs_mlp.predict(X_test)
# Best parameters.

print("Best MLP Parameters: {}".format(gs_mlp.best_params_))



# The mean squared error (relative error).

print("Mean squared error (MSE): %.3f" % mean_squared_error(y_test, gs_mlp_pred))



# Explained average absolute error (average error).

print("Average absolute error (MAE): %.3f" % mean_absolute_error(y_test, gs_mlp_pred))



# Cross validation accuracy for the best parameters.

print('CV accuracy: %0.3f' % gs_mlp.best_score_)



# Accuracy: 1 is perfect prediction.

print('Accuracy: %0.3f' % (gs_mlp.score(X_test,y_test)))
# Print confusion matrix for GridSearchCV MLP.

gs_mlp_matrix = confusion_matrix(y_test,gs_mlp_pred)

plt.figure(figsize=(8,8))

sns.set(font_scale=1.4)

sns.heatmap(gs_mlp_matrix,annot=True, cbar=False, cmap='twilight',linewidth=0.5,fmt="d")

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.title('Confusion Matrix for GridSearchCV MLP');
# Ploting metrics

errors=['Accuracy','CV-accuracy','MSE', 'MAE']



fig = go.Figure(data=[

    go.Bar(name='Logistic Regression', x=errors, y=[logreg.score(X_test, y_test),logreg_cv.mean(),mean_squared_error(y_test, log_pred), mean_absolute_error(y_test, log_pred)]),

    go.Bar(name='Decision tree', x=errors, y=[tr.score(X_test, y_test),tr_cv.mean(),mean_squared_error(y_test, tr_pred), mean_absolute_error(y_test, tr_pred)]),

    go.Bar(name='MLP', x=errors, y=[mlp.score(X_test, y_test),mlp_cv.mean(),mean_squared_error(y_test, mlp_pred), mean_absolute_error(y_test, mlp_pred)]),

    go.Bar(name='GridSearchCV+Logistic Regression', x=errors, y=[gs_logreg.score(X_test, y_test),gs_logreg.best_score_,mean_squared_error(y_test, gs_log_pred), mean_absolute_error(y_test, gs_log_pred)]),

    go.Bar(name='GridSearchCV+Decision tree', x=errors, y=[gs_tree.score(X_test, y_test),gs_tree.best_score_,mean_squared_error(y_test, gs_tree_pred), mean_absolute_error(y_test, gs_tree_pred)]),

    go.Bar(name='GridSearchCV+MLP', x=errors, y=[gs_mlp.score(X_test, y_test),gs_mlp.best_score_,mean_squared_error(y_test, gs_mlp_pred), mean_absolute_error(y_test, gs_mlp_pred)])

])



fig.update_layout(

    title='Metrics for each model',

    xaxis_tickfont_size=14,    

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
d={

'': ['Logistic Regression','GridSearchCV + Logistic Regression','Decision Tree','GridSearchCV + Decision Tree','Neural Network (MLP)','GridSearchCV + Neural Network (MLP)'],

'Accuracy': [logreg.score(X_test, y_test), gs_logreg.score(X_test,y_test),tr.score(X_test, y_test),gs_tree.score(X_test,y_test),mlp.score(X_test, y_test),gs_mlp.score(X_test, y_test)],

'CV Accuracy': [logreg_cv.mean(), gs_logreg.best_score_, tr_cv.mean(),gs_tree.best_score_,mlp_cv.mean(),gs_mlp.best_score_],

'MSE': [mean_squared_error(y_test, log_pred),mean_squared_error(y_test, gs_log_pred),mean_squared_error(y_test, tr_pred), mean_squared_error(y_test, gs_tree_pred),mean_squared_error(y_test, mlp_pred),mean_squared_error(y_test, gs_mlp_pred)],

'MAE': [mean_absolute_error(y_test, log_pred),mean_absolute_error(y_test, gs_log_pred),mean_absolute_error(y_test, tr_pred), mean_absolute_error(y_test, gs_tree_pred),mean_absolute_error(y_test, mlp_pred),mean_absolute_error(y_test, gs_mlp_pred)]

}



results=pd.DataFrame(data=d).round(3).set_index('')

results