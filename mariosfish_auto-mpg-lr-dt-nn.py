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

from sklearn.linear_model import LinearRegression

from sklearn import tree

from sklearn.tree.export import export_text

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler



##################################



### Metrics ###

from sklearn import metrics

from sklearn.metrics import f1_score,confusion_matrix, mean_squared_error, mean_absolute_error, classification_report, roc_auc_score, roc_curve, precision_score, recall_score
# Read the data from the auto-mpg_data-original.csv file.

ds=pd.read_csv("../input/auto-mpg_data-original.csv")
# Examine the data types and the number of non-null items.

ds.info()
# Display the shape of the data.

print("The data frame has {} rows and {} columns.".format(ds.shape[0],ds.shape[1]))
# Statistics for the data set.

ds.describe().transpose()
# Check for duplicate rows.

print(f"There are {ds.duplicated().sum()} duplicate rows in the data set.")
# Number of null values for each attribute.

ds.isnull().sum()
# Rows with null value at the "mpg" column.

ds[ds['mpg'].isnull()]
# Store rows with a missing value at "mpg" attribute as a prediction set for future use.

prediction_set=ds[ds['mpg'].isnull()].drop('mpg',axis=1)
# Rows with null value at the "horsepower" column.

ds[ds['horsepower'].isnull()]
# Fill the null values of "horsepower" with the mean of "horsepower".

ds['horsepower']=ds['horsepower'].fillna(ds['horsepower'].mean())
# Remove rows from the data set with a null values.

new_ds=ds.dropna()
# Information about the new "clean" data set.

new_ds.info()



# Display the shape of the data.

print("\nThe new data frame has {} rows and {} columns.".format(new_ds.shape[0],new_ds.shape[1]))
# Statistics for the new data set.

new_ds.describe().transpose()
# Plotting a pairplot for the new data set.

plt.figure(figsize=(20,20))

sns.set(font_scale=1.1)

sns.pairplot(data=new_ds, diag_kind='kde', hue='origin')

plt.show();
# Plotting a correlation heatmap for the new data set.

corr = new_ds.corr()

plt.figure(figsize=(8,8))

sns.set(font_scale=1.1)

sns.heatmap(data=corr,annot=True,cmap='rainbow',linewidth=0.5)

plt.title('Correlation Matrix')

plt.show();
# Distinguish attribute columns and target column.

X=new_ds[new_ds.columns[1:-1]]

y=new_ds['mpg']
# Split to train and test sets. 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)
# Standardization

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Initialize a Logistic Regression estimator.

linreg=LinearRegression(n_jobs=-1)



# Train the estimator.

linreg.fit(X_train,y_train)
# Make predictions.

lin_pred=linreg.predict(X_test)



# Calculate CV score.

cv_lin_reg=cross_val_score(linreg, X_train, y_train, cv=10).mean()
# Mean squared error (relative error).

print("Mean squared error (MSE): %.2f" % mean_squared_error(y_test, lin_pred))



# Mean absolute error (average error).

print("Mean absolute error (MAE): %.2f" % mean_absolute_error(y_test, lin_pred))



# Cross-Validation accuracy

print('Cross-validation accuracy: %0.1f' % (cv_lin_reg*100),'%')



# Accuracy score: 1 is perfect prediction.

print('Accuracy: %.1f' % (linreg.score(X_test, y_test)*100),'%')
# Hyperparameters to be checked.

parameters = {'normalize':[True, False],

              'fit_intercept':[True, False]

             }



# Linear Regression estimator.

default_linreg=LinearRegression(n_jobs=-1)



# GridSearchCV estimator.

gs_linreg = GridSearchCV(default_linreg, parameters, cv=10, n_jobs=-1, verbose=1)



# Train the GridSearchCV estimator and search for the best parameters.

gs_linreg.fit(X_train,y_train)
# Make predictions with the best parameters.

gs_linreg_pred=gs_linreg.predict(X_test)
# Best parameters.

print("Best Linear Regression Parameters: {}".format(gs_linreg.best_params_))



# Mean squared error (relative error).

print("Mean squared error (MSE): %.2f" % mean_squared_error(y_test, gs_linreg_pred))



# Mean absolute error (average error).

print("Mean absolute error (MAE): %.2f" % mean_absolute_error(y_test, gs_linreg_pred))



# Cross validation accuracy for the best parameters.

print('Cross-validation accuracy: %0.1f' % (gs_linreg.best_score_*100),'%')



# Accuracy score: 1 is perfect prediction.

print('Accuracy: %.1f' % (gs_linreg.score(X_test, y_test)*100),'%')
# Initialize a decision tree estimator.

tr = tree.DecisionTreeRegressor(max_depth=3, random_state=25)



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



# Calculate CV score.

cv_tr_reg=cross_val_score(tr, X_train, y_train, cv=10).mean()
# Mean squared error (relative error).

print("Mean squared error (MSE): %.2f" % mean_squared_error(y_test, tr_pred))



# Mean absolute error (average error).

print("Mean absolute error (MAE): %.2f" % mean_absolute_error(y_test, tr_pred))



# Cross-Validation accuracy.

print('Cross-validation accuracy: %0.1f' % (cv_tr_reg*100),'%')



# Accuracy score: 1 is perfect prediction.

print('Accuracy: %.1f' % (tr.score(X_test, y_test)*100),'%')
# Hyperparameters to be checked.

parameters = {'criterion':['mse','friedman_mse','mae'],

              'splitter':['best','random'],

              'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],

              'min_samples_leaf':[2,3,5,10,20]

             }



# MLP estimator.

default_tr = tree.DecisionTreeRegressor(random_state=25)



# GridSearchCV estimator.

gs_tree = GridSearchCV(default_tr, parameters, cv=10, n_jobs=-1,verbose=1)



# Train the GridSearchCV estimator and search for the best parameters.

gs_tree.fit(X_train,y_train)
# Make predictions with the best parameters.

gs_tree_pred=gs_tree.predict(X_test)
# Best parameters.

print("Best Decision tree Parameters: {}".format(gs_tree.best_params_))



# Mean squared error (relative error).

print("Mean squared error (MSE): %.2f" % mean_squared_error(y_test, gs_tree_pred))



# Mean absolute error (average error).

print("Mean absolute error (MAE): %.2f" % mean_absolute_error(y_test, gs_tree_pred))



# Cross validation accuracy for the best parameters.

print('Cross-validation accuracy: %0.1f' % (gs_tree.best_score_*100),'%')



# Accuracy score: 1 is perfect prediction.

print('Accuracy: %.1f' % (gs_tree.score(X_test, y_test)*100),'%')
# Initialize a Multi-layer Perceptron classifier.

mlp = MLPRegressor(max_iter=1000, random_state=25,shuffle=True, verbose=False)



# Train the classifier.

mlp.fit(X_train, y_train)
# Make predictions.

mlp_pred = mlp.predict(X_test)



# Calculate CV score.

cv_mlp_reg=cross_val_score(mlp, X_train, y_train, cv=10).mean()
# Mean squared error (relative error).

print("Mean squared error (MSE): %.2f" % mean_squared_error(y_test, mlp_pred))



# Mean absolute error (average error).

print("Mean absolute error (MAE): %.2f" % mean_absolute_error(y_test, mlp_pred))



# Cross-Validation accuracy

print('Cross-validation accuracy: %0.1f' % (cv_mlp_reg*100),'%')



# Accuracy score: 1 is perfect prediction.

print('Accuracy: %.1f' % (mlp.score(X_test, y_test)*100),'%')
# Hyperparameters to be checked.

parameters = {'activation':['logistic','tanh','relu'],

              'solver': ['lbfgs','adam','sgd'],

              'alpha': 10.0 ** -np.arange(1,3),

              'learning_rate': ['constant', 'invscaling', 'adaptive'],

              'hidden_layer_sizes':[(7),(6),(14),(3),(7,3),(6,3),(14,7),(3,1)]}



# Decision tree estimator.

default_mlp = MLPRegressor(max_iter=1000, random_state=25,shuffle=True, verbose=False)



# GridSearchCV estimator.

gs_mlp = GridSearchCV(default_mlp, parameters, cv=10, n_jobs=-1,verbose=1)



# Train the GridSearchCV estimator and search for the best parameters.

gs_mlp.fit(X_train,y_train)
# Make predictions with the best parameters.

gs_mlp_pred=gs_mlp.predict(X_test)
# Best parameters.

print("Best MLP Regression Parameters: {}".format(gs_mlp.best_params_))



# Mean squared error (relative error).

print("Mean squared error (MSE): %.2f" % mean_squared_error(y_test, gs_mlp_pred))



# Mean absolute error (average error).

print("Mean absolute error (MAE): %.2f" % mean_absolute_error(y_test, gs_mlp_pred))



# Cross validation accuracy for the best parameters.

print('Cross-validation accuracy: %0.1f' % (gs_mlp.best_score_*100),'%')



# Accuracy score: 1 is perfect prediction.

print('Accuracy: %.1f' % (gs_mlp.score(X_test, y_test)*100),'%')
metrics=['MSE','MAE','CV accuracy','Accuracy']



# Plot metrics.

fig = go.Figure(data=[

    go.Bar(name='Linear Regression', x=metrics, y=[mean_squared_error(y_test, lin_pred),mean_absolute_error(y_test, lin_pred),cv_lin_reg,linreg.score(X_test, y_test)]),

    go.Bar(name='Decision tree', x=metrics, y=[mean_squared_error(y_test, tr_pred),mean_absolute_error(y_test, tr_pred),cv_tr_reg,tr.score(X_test, y_test)]),

    go.Bar(name='Neural Network', x=metrics, y=[mean_squared_error(y_test, mlp_pred),mean_absolute_error(y_test, mlp_pred),cv_mlp_reg,mlp.score(X_test, y_test)]),

    go.Bar(name='GridSearch+Linear Regression', x=metrics, y=[mean_squared_error(y_test, gs_linreg_pred),mean_absolute_error(y_test, gs_linreg_pred),gs_linreg.best_score_,gs_linreg.score(X_test, y_test)]),

    go.Bar(name='GridSearch+Decision tree', x=metrics, y=[mean_squared_error(y_test, gs_tree_pred),mean_absolute_error(y_test, gs_tree_pred),gs_tree.best_score_,gs_tree.score(X_test, y_test)]),

    go.Bar(name='GridSearch+Neural Network', x=metrics, y=[mean_squared_error(y_test, gs_mlp_pred),mean_absolute_error(y_test, gs_mlp_pred),gs_mlp.best_score_,gs_mlp.score(X_test, y_test)])

])



fig.update_layout(title_text='Results',

                  barmode='group',xaxis_tickangle=-45,bargroupgap=0.05)

fig.show()
d={

'': ['Linear Regression','GridSearchCV + Linear Regression','Decision Tree','GridSearchCV + Decision Tree','Neural Network (MLP)','GridSearchCV + Neural Network (MLP)'],

    'MSE': [mean_squared_error(y_test, lin_pred), mean_squared_error(y_test, gs_linreg_pred),mean_squared_error(y_test, tr_pred),mean_squared_error(y_test, gs_tree_pred),mean_squared_error(y_test, mlp_pred),mean_squared_error(y_test, gs_mlp_pred)],

    'MAE': [mean_absolute_error(y_test, lin_pred), mean_absolute_error(y_test, gs_linreg_pred),mean_absolute_error(y_test, tr_pred),mean_absolute_error(y_test, gs_tree_pred),mean_absolute_error(y_test, mlp_pred),mean_absolute_error(y_test, gs_mlp_pred)],

    'CV Accuracy': [cv_lin_reg, gs_linreg.best_score_, cv_tr_reg,gs_tree.best_score_,cv_mlp_reg,gs_mlp.best_score_],

    'Accuracy': [linreg.score(X_test, y_test), gs_linreg.score(X_test,y_test),tr.score(X_test, y_test),gs_tree.score(X_test,y_test),mlp.score(X_test, y_test),gs_mlp.score(X_test, y_test)]

}



results=pd.DataFrame(data=d).round(3).set_index('')

results