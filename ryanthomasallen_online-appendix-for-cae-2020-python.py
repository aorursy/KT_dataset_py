# NUMPY / PANDAS

import numpy as np

import pandas as pd



# SCI-KIT LEARN

import sklearn

from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, cross_validate, 

                                       RandomizedSearchCV,GridSearchCV)

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

import sklearn.tree as tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler,OneHotEncoder

from sklearn.pipeline import make_pipeline

from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.metrics import log_loss, roc_curve, auc



#PDPbox

from pdpbox import pdp



# MATPLOTLIB

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



# OTHER

from itertools import product

import copy

import graphviz



#Ignore warnings

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/simulated-data-for-ml-paper/simulated_TECHCO_data.csv")
df
df.time = df.time.astype(int)

df.is_male = df.is_male.astype(int)
df.set_index(['emp_id','time'],drop=False,inplace=True)
df.loc[df['turnover']=='Stayed', 'turnover'] = '0 Stayed'

df.loc[df['turnover']=='Left', 'turnover'] = '1 Left'

df.turnover

y = df.turnover
X = df.drop(['turnover'],axis=1)

cols = ["time","training_score","avg_literacy","is_male","logical_score","verbal_score","location_age","distance","similar_language"]

X = X[cols]
train_inds, test_inds = next(GroupShuffleSplit(test_size=.3,random_state=111).split(X,y,groups=df.emp_id))
X_train, X_test = X.iloc[train_inds], X.iloc[test_inds]

y_train, y_test = y.iloc[train_inds], y.iloc[test_inds]

employee_train = df.emp_id.iloc[train_inds]
folds = list(GroupKFold(n_splits=10).split(X_train,y_train,employee_train))
X_train_not_panel = X_train[X_train.time==1].drop(columns="time")

X_train_not_panel.corr()
pd.plotting.scatter_matrix(X_train_not_panel,figsize=(16,12),alpha=0.3);
def plot_roc(y_predictions,y_true,name,pos_label):

    fpr,tpr,thresholds = roc_curve(y_true,y_predictions,pos_label=pos_label)

    roc_auc = auc(fpr, tpr)

    plt.title(name)

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()
def pdplot(X, model, n, var_name,categorical_var=False,which_class = 1):        

    X_copy = copy.deepcopy(X)

    #For the continuous variables that will be plotted, create 40-interval arrays.

    if categorical_var == False:

        var_grid_vals = np.linspace(X_copy[var_name].min(), X_copy[var_name].max(), num=40)

    #For the categorical variables that will be plotted, create array of the unique values

    if categorical_var == True:

        var_grid_vals = list(set(X_copy[var_name]))

    

    samples=np.random.choice(len(X_copy), n, replace=False)

    

    predictions = pd.DataFrame()

      

    f=plt.figure()

    

    for sample in samples:

        x_vals = list()

        for i in var_grid_vals:

            X_copy[var_name]=i

            y_hat=model.predict_proba(X_copy.iloc[(sample-1):sample])[:,which_class][0]

            y_hat_log_odds = np.log(y_hat/(1-y_hat))

            predictions=predictions.append({'sample':sample,'x_val':i,'pred':y_hat_log_odds},ignore_index=True)

        sample_preds = predictions[predictions['sample']==sample]

        plt.plot(sample_preds.x_val, sample_preds.pred, c='C1', alpha=1.0, linewidth=0.1)

        

    preds_grouped = predictions.groupby(['x_val']).mean().reset_index()

    plt.plot(preds_grouped.x_val, preds_grouped.pred, c='C0', linestyle='--')

    plt.ylabel(r'Log Odds of Turnover Probability (log $\frac{h_\theta(x)}{1-h_\theta(x)}$)')

    plt.xlabel(var_name)

    plt.ylim(-8,0)
def plot_twoway_pdp(X,model,var1_name,var2_name,categorical_var1=False,categorical_var2=False,

                 var1_min=None,var1_max=None,

                 var2_min=None,var2_max=None,

                 which_class = 1):

    #Set the min and max value to plot for both variables. Default is the min and max of the variable

    if var1_min is None:

        var1_min=X[var1_name].min() 

    if var1_max is None:

        var1_max=X[var1_name].max() 

    if var2_min is None:

        var2_min=X[var2_name].min()

    if var2_max is None:

        var2_max=X[var2_name].max() 

        

    X_copy = copy.deepcopy(X)

    

    #For the continuous variables that will be plotted, create 40-interval arrays.

    if categorical_var1 == False:

        var1_grid_vals = np.linspace(var1_min, var1_max, num=40)

    if categorical_var2 == False:

        var2_grid_vals = np.linspace(var2_min, var2_max, num=40)

        

    #For the categorical variables that will be plotted, create array of the unique values

    if categorical_var1 == True:

        var1_grid_vals = list(set(X_copy[var1_name]))

    if categorical_var2 == True:

        var2_grid_vals = list(set(X_copy[var2_name]))

    

    predictions_from_grid = list()

    x_vals = list()

    y_vals =list()

    

    for i in var1_grid_vals:

        for j in var2_grid_vals:

            X_copy[var1_name]=i

            X_copy[var2_name]=j

            y_hats = model.predict_proba(X_copy)[:,which_class]

            predictions_from_grid.append(np.mean(y_hats))   

            x_vals.append(i)

            y_vals.append(j)



    plt.figure()

    plt.scatter(x_vals,y_vals,c=np.log(predictions_from_grid),marker='s',vmin=-8,vmax=-1)

    plt.xlabel(var1_name)

    plt.ylabel(var2_name)

    cbar = plt.colorbar(ticks=range(-8,0))

    cbar.ax.set_yticklabels(['$10^{'+str(i)+'}$' for i in range(-8,0) ])
#Make a dictionary of combinations of hyperparameters to try

random_grid = {'criterion': ['entropy','gini'],

               'max_depth': np.unique( np.exp(np.linspace(0, 10, 100)).astype(int) ),

               'min_samples_leaf': np.unique( np.exp(np.linspace(0, 8, 100)).astype(int) ),

               'min_impurity_decrease': np.exp(np.linspace(-9, -1, 100))}
dt_random_search = RandomizedSearchCV(estimator = DecisionTreeClassifier(), 

                                      param_distributions = random_grid,

                                      random_state=345, n_iter = 100,

                                      scoring='neg_log_loss',n_jobs=-1,

                                      cv =folds,return_train_score=True)

dt_random_search.fit(X=X_train,y=y_train)

dt_random_search.best_params_
dt=dt_random_search.best_estimator_
best_model_index = dt_random_search.best_index_

dt_train_score = dt_random_search.cv_results_['mean_train_score'][best_model_index]

dt_validation_score = dt_random_search.cv_results_['mean_test_score'][best_model_index]

dt_train_std = dt_random_search.cv_results_['std_train_score'][best_model_index]

dt_validation_std = dt_random_search.cv_results_['std_test_score'][best_model_index]
decisiontree_test_pred = dt.predict_proba(X_test)[:,1]
test_loss_dt = log_loss(y_test.values,decisiontree_test_pred)

print("Test Loss: %0.4f" % test_loss_dt)
decision_tree = graphviz.Source( tree.export_graphviz(dt, out_file=None, feature_names=X_train.columns, filled=False, rounded=True,impurity=True))

decision_tree
plot_roc(decisiontree_test_pred,y_test,"Decision Tree ROC Curve",pos_label="1 Left")
importances = dt.feature_importances_



plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances,

       color="b", align="center")

plt.xticks(range(X.shape[1]), cols,rotation=90)

plt.xlim([-1, X.shape[1]])

plt.show()
pdplot(X=X_train,var_name='time',n=500,model=dt,which_class = 1)
plot_twoway_pdp(X=X_train, model=dt,

             var1_name='time',var2_name='training_score',

             var2_min=2.5,which_class=1)
#Make a dictionary of which combinations of hyperparameters to try

random_grid = {'criterion': ['entropy'],

               'max_depth': np.unique( np.exp(np.linspace(0, 10, 100)).astype(int) ),

               'min_samples_leaf': np.unique( np.exp(np.linspace(0, 8, 100)).astype(int) ),

               'max_features': [None,'auto','log2'],

               'min_impurity_decrease': np.exp(np.linspace(-9, -1, 100))}
rf_random_search = RandomizedSearchCV(estimator = RandomForestClassifier(n_estimators=100), 

                                      param_distributions = random_grid,

                                      random_state=345, n_iter = 100,

                                      scoring='neg_log_loss',n_jobs=-1,

                                      cv =folds,return_train_score=True)

rf_random_search.fit(X=X_train,y=y_train)

rf_random_search.best_params_
rf = rf_random_search.best_estimator_
best_model_index = rf_random_search.best_index_

rf_train_score = rf_random_search.cv_results_['mean_train_score'][best_model_index]

rf_validation_score = rf_random_search.cv_results_['mean_test_score'][best_model_index]

rf_train_std = rf_random_search.cv_results_['std_train_score'][best_model_index]

rf_validation_std = rf_random_search.cv_results_['std_test_score'][best_model_index]
randomforest_test_pred = rf.predict_proba(X_test)[:,1]
test_loss_rf = log_loss(y_test.values,randomforest_test_pred)

print("Test Loss: %0.4f" % test_loss_rf)
plot_roc(randomforest_test_pred,y_test,"Random Forest ROC Curve",pos_label="1 Left")
importances = rf.feature_importances_



plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances,

       color="b", align="center")

plt.xticks(range(X.shape[1]), cols,rotation=90)

plt.xlim([-1, X.shape[1]])

plt.show()
pdplot(X=X_train,var_name='time',n=500,model=rf,which_class = 1)
plot_twoway_pdp(X=X_train, model=rf,

             var1_name='time',var2_name='training_score',

             var2_min=2.5,which_class=1)
nnet = MLPClassifier(random_state=345,max_iter=100)

scaler = StandardScaler()

nnet_pipeline = make_pipeline(scaler,nnet)
#Make a dictionary of which combinations of hyperparameters to try

random_grid = {'mlpclassifier__solver': ['adam'],

               'mlpclassifier__activation': ['tanh','relu','logistic'],

               'mlpclassifier__alpha': [0.01,0.05],

               'mlpclassifier__hidden_layer_sizes': [(30,20,10),(20,20),(25,15),(10,40,10),(20,20,20),(30,10)]}
# #Commented out for speed

# nnet_random_search = RandomizedSearchCV(estimator = nnet_pipeline, 

#                                       param_distributions = random_grid, n_iter = 25,

#                                       #random_state=345, 

#                                       scoring='neg_log_loss',n_jobs=-1,

#                                       cv =folds,return_train_score=True)

# nnet_random_search.fit(X=X_train,y=y_train)

# nnet_random_search.best_params_
nnet = MLPClassifier(solver='adam', 

                     activation='tanh',

                     alpha=1e-2, 

                     hidden_layer_sizes=(32,12), 

                     random_state=345)

nnet_pipeline = make_pipeline(scaler,nnet)
cv_results_nnet = cross_validate(

    estimator=nnet_pipeline,

    X=X_train,

    y=y_train,

    cv=folds,

    scoring='neg_log_loss',

    return_train_score=True

)

cv_val_score_nnet = -cv_results_nnet['test_score'].mean()



print("Mean cross-validation score (log loss) for neural network model: %0.4f" % cv_val_score_nnet)
nnet_pipeline.fit(X_train,y_train)

nnet_test_pred = nnet_pipeline.predict_proba(X_test)[:,1]
test_loss_nnet = log_loss(y_test.values,nnet_test_pred)

print("Test Loss: %0.4f" % test_loss_nnet)
plot_roc(nnet_test_pred,y_test,"Neural Network ROC Curve",pos_label="1 Left")
pdplot(X=X_train,var_name='time',n=500,model=nnet_pipeline,which_class = 1)
plot_twoway_pdp(X=X_train, model=nnet_pipeline,

             var1_name='time',var2_name='training_score',

             var2_min=2.5,which_class=1)
preprocess = make_column_transformer(

    (OneHotEncoder(), ["time"]),

remainder="passthrough")



logistic = LogisticRegression(C=1e8)



logistic_pipeline = make_pipeline(

    preprocess,

    logistic)
cv_results_logistic = cross_validate(

    estimator=logistic_pipeline,

    X=X_train,

    y=y_train,

    cv=folds,

    scoring='neg_log_loss',

    return_train_score=True

)

cv_val_score_logistic = -cv_results_logistic['test_score'].mean()



print("Mean cross-validation score (log loss) for logistic model: %0.4f" % cv_val_score_logistic)
logistic_pipeline.fit(X_train,y_train)

logistic_test_pred = logistic_pipeline.predict_proba(X_test)[:,1]



print("Test Loss: %0.4f" % log_loss(y_test.values,logistic_test_pred))
plot_roc(logistic_test_pred,y_test,"Logistic ROC Curve",pos_label="1 Left")
#Note since the variable time is categorical in our logistic_pipeline model, we set categorical_var=True in the pdplot function

pdplot(X=X_train,var_name='time',categorical_var=True,n=500,model=logistic_pipeline)
#Note since the variable time is categorical in our logistic_pipeline model, we set categorical_var1=True in the plot_twoway_pdp function

plot_twoway_pdp(X=X_train, model=logistic_pipeline,

             var1_name='time',var2_name='training_score',categorical_var1=True,

             var2_min=2.5,which_class=1)
x=[-dt_train_score,-rf_train_score,-cv_results_nnet['train_score'].mean(),-cv_results_logistic['train_score'].mean()]

x_err=[dt_train_std,rf_train_std,np.std(cv_results_nnet['train_score']),np.std(cv_results_logistic['train_score'])]

y=[-dt_validation_score,-rf_validation_score,-cv_results_nnet['test_score'].mean(),-cv_results_logistic['test_score'].mean()]

y_err=[dt_validation_std,rf_validation_std,np.std(cv_results_nnet['test_score']),np.std(cv_results_logistic['test_score'])]
plt.plot([-1,1],[-1,1], c='k', marker='None',linestyle='--',label='_nolegend_')

for ax, ax_err, ay, ay_err in zip(x, x_err, y, y_err):

    plt.errorbar(ax, ay, xerr=ax_err, yerr=ay_err, label='Training',marker='o', linestyle='None')

plt.xlim(0.06,0.07625)

plt.ylim(0.06,0.07625)

plt.xlabel('Training Loss')

plt.ylabel('Validation Loss')

plt.legend(['Decision Tree','Random Forest','Neural Network','Logistic Regression'], loc=4)
x=[-(dt_train_score-cv_results_logistic['train_score'].mean()),-(rf_train_score-cv_results_logistic['train_score'].mean()),-(cv_results_nnet['train_score'].mean()-cv_results_logistic['train_score'].mean()),0]

x_err=[np.std(dt_train_std-cv_results_logistic['train_score']),np.std(rf_train_std-cv_results_logistic['train_score']),np.std(cv_results_nnet['train_score']-cv_results_logistic['train_score']),0]

y=[-(dt_validation_score-cv_results_logistic['test_score'].mean()),-(rf_validation_score-cv_results_logistic['test_score'].mean()),-(cv_results_nnet['test_score'].mean()-cv_results_logistic['test_score'].mean()),0]

y_err=[np.std(dt_validation_std-cv_results_logistic['test_score']),np.std(rf_validation_std-cv_results_logistic['test_score']),np.std(cv_results_nnet['test_score']-cv_results_logistic['test_score']),0]
plt.plot([-1,1],[-1,1], c='k', marker='None',linestyle='--',label='_nolegend_')

for ax, ax_err, ay, ay_err in zip(x, x_err, y, y_err):

    plt.errorbar(ax, ay, xerr=ax_err, yerr=ay_err, label='Training',marker='o', linestyle='None')

plt.xlim(-0.010,0.002)

plt.ylim(-0.0082,0.0025)

plt.xlabel('Training Loss (Relative to Logistic)')

plt.ylabel('Validation Loss (Relative to Logistic)')

plt.legend(['Decision Tree','Random Forest','Neural Network','Logistic Regression'], loc=4)