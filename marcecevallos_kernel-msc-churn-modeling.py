import pandas as pd

import numpy as np

import os

import random

from numpy import mean

from pprint import pprint





# Visualization 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.gridspec as gridspec

from sklearn.model_selection import RandomizedSearchCV



# Modelization

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from scikitplot.estimators import plot_feature_importances

import category_encoders as ce

import statsmodels.formula.api as smf

import statsmodels.api as sm

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from sklearn.metrics import mean_squared_error 

from sklearn.model_selection import train_test_split

import eli5

from eli5.sklearn import PermutationImportance

from eli5.sklearn.explain_weights import explain_decision_tree, explain_rf_feature_importance

import xgboost as xgb

from eli5.xgboost import explain_weights_xgboost



# Warnings

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option("display.max_columns",200)
import os

print(os.listdir("../input"))
data = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")

data.head(10)
data.describe()
def dimensionality(data):

    print("The dataset has", data.shape[0], " observations, and ", data.shape[1], "columns")



dimensionality(data)
data.dtypes
def variable_types(data):

    print(data.dtypes)

    print("\nThere are", sum(data.dtypes=="object"), "qualitative variables and", sum(data.dtypes=="int64") + sum(data.dtypes=='float64'), "quantitative variables")

    

variable_types(data)
def filter_by_dtype(data, data_type):

    """filter a dataframe by columns with a certain data_type"""

    col_names = data.dtypes[data.dtypes == data_type].index

    return data[col_names]
### Filtering the numerical variables

data_numerical = pd.concat([filter_by_dtype(data, 'int64'), filter_by_dtype(data, 'float64')],axis=1)

data_numerical.head(10)
### Plot Distibutions ###

graph_1 = plt.figure(figsize = (15,20))

ax = graph_1.gca()

data_numerical.hist(ax = ax, bins = 15)

plt.show()
print(data.Surname.value_counts().sort_index().head())

print("\n")
len(data.Surname.unique())

data.Surname.value_counts().head(10)
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
data.head()
print(data.Geography.value_counts().sort_index())

print("\n")

print(data.Gender.value_counts().sort_index())
plt.figure(figsize=(12, 9))

plt.subplot(2,2,1)

sns.countplot((data.NumOfProducts), palette='colorblind')



plt.subplot(2,2,2)

sns.countplot(data.IsActiveMember)



plt.subplot(2,2,3)

sns.countplot(data.Exited)
plt.figure(figsize=(12, 9))



plt.subplot(2,2,1)

sns.countplot((data.Geography), palette='colorblind')



plt.subplot(2,2,2)

sns.countplot(data.Gender)



plt.subplot(2,2,3)

sns.countplot(data.Tenure)



plt.subplot(2,2,4)

sns.countplot(data.HasCrCard)
def variabletype(data):

    colname=data.columns

    coltype=data.dtypes

    variabletype=[]

    for i in data:

        if (data[i].nunique()>11) and (data[i].dtype=='int64' or data[i].dtype=='float64'):

            variabletype.append('Continuous')

        else:

            variabletype.append('Class')

    #variabletype

    dict={'ColumnName':colname,

         'Column_dtype':coltype,

          'Variable_Type':variabletype}

    return pd.DataFrame(dict)

df1=variabletype(data)

df1
def correlation_matrix(data):

    sns.set(style="dark", palette='colorblind')

    corr = data.corr('spearman')

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    return sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,

                square=True, annot = corr.round(2), linewidths=.5, cbar_kws={"shrink": .5})
correlation_matrix(data)
# Salary to Balance Ratio

data['BalanceToSalaryRatio']=data.EstimatedSalary/data.Balance



# Score to Balance Ratio

data['ScoreToBalance']=data.CreditScore/data.Balance



# age to Salary ratio

data['SalaryToAge']=data.Age/data.EstimatedSalary



# Products to Balance

data['ProductsToBalance']=data.NumOfProducts/data.Balance
data.head()
data=data.replace([np.inf, -np.inf], 0)
X=data[data.columns.difference(['Exited'])]

y=data['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42

                                                   )
print(dimensionality(X_train))

print(dimensionality(X_test))
print('The dataset has' , y_train.shape[0], 'observations')

print('The dataset has' , y_test.shape[0], 'observations')
print(X_train.isnull().sum())

print(X_test.isnull().sum())
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(X_train)
numCols=[]

levCols=[]

response=[]

for i in X_train.columns:

    if (X_train[i].dtype=='int64' or X_train[i].dtype=='float64'):

        numCols.append(i)

    else:

        levCols.append(i)

        

print(numCols)

print(levCols)
for i in numCols:

    if (i=='NumOfProducts' or i=='HasCrCard' or i=='IsActiveMember' or i=='Tenure' ):

        levCols.append(i)



elements_to_remove= ['NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Tenure', 'Exited']

for j in elements_to_remove:

    if j in numCols:

        numCols.remove(j)
print(numCols)

print(levCols)

response = 'Exited'

response
encoder=ce.BinaryEncoder(cols=levCols)
data_encoded=encoder.fit_transform(X_train[levCols])

X_train=pd.concat([X_train, data_encoded], axis=1)

X_train.head()
data_encoded2=encoder.transform(X_test[levCols])

X_test=pd.concat([X_test, data_encoded2], axis=1)

X_test.head()
#Gender

X_train['Gender'].replace(to_replace='Male', value='0', regex=True, inplace=True)

X_train['Gender'].replace(to_replace='Female', value='1', regex=True, inplace=True)

#Geography

X_train['Geography'].replace(to_replace='Spain', value='0', regex=True, inplace=True)

X_train['Geography'].replace(to_replace='Germany', value='1', regex=True, inplace=True)

X_train['Geography'].replace(to_replace='France', value='2', regex=True, inplace=True)
X_train['Gender'] = X_train.Gender.astype(int)

X_train['Geography'] = X_train.Geography.astype(int)
X_train.head()
#Gender

X_test['Gender'].replace(to_replace='Male', value='0', regex=True, inplace=True)

X_test['Gender'].replace(to_replace='Female', value='1', regex=True, inplace=True)

#Geography

X_test['Geography'].replace(to_replace='Spain', value='0', regex=True, inplace=True)

X_test['Geography'].replace(to_replace='Germany', value='1', regex=True, inplace=True)

X_test['Geography'].replace(to_replace='France', value='2', regex=True, inplace=True)
X_test['Gender'] = X_test.Gender.astype(int)

X_test['Geography'] = X_test.Geography.astype(int)
X_test.head()
X_train1= X_train.copy()

X_train2= X_train.copy()

X_test1= X_test.copy()

X_test2=  X_test.copy()

columns_to_standarize=['Age', 'Balance', 'BalanceToSalaryRatio', 'CreditScore', 'EstimatedSalary',

                      'ProductsToBalance', 'SalaryToAge', 'ScoreToBalance']

#X_train2[columns_to_standarize] = X_train2[columns_to_standarize].apply(lambda x: (x-x.mean()/x.std()))
sc = StandardScaler()

X_train2 = sc.fit_transform(X_train2[columns_to_standarize])

X_test2 = sc.transform(X_test2[columns_to_standarize])
X_train2=pd.DataFrame(X_train2, columns=columns_to_standarize)
X_test2=pd.DataFrame(X_test2, columns=columns_to_standarize)
X_train1.reset_index(drop=True, inplace=True)

X_test1.reset_index(drop=True, inplace=True)

X_train2.reset_index(drop=True, inplace=True)

X_test2.reset_index(drop=True, inplace=True)
X_train1.drop(['Age', 'Balance', 'BalanceToSalaryRatio', 'CreditScore', 'EstimatedSalary',

                      'ProductsToBalance', 'SalaryToAge', 'ScoreToBalance'],axis=1, inplace=True)

X_test1.drop(['Age', 'Balance', 'BalanceToSalaryRatio', 'CreditScore', 'EstimatedSalary',

                      'ProductsToBalance', 'SalaryToAge', 'ScoreToBalance'],axis=1, inplace=True)
X_train1=pd.concat([X_train1, X_train2], axis=1)

missing_data(X_train1)
X_test1=pd.concat([X_test1, X_test2], axis=1)

missing_data(X_test1)
X_train_standarized=X_train1.copy()

X_test_standarized=X_test1.copy()
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state= 42)

X_train_standarized_upsampled, y_train_standarized_upsampled = ros.fit_sample(X_train_standarized, y_train)

y_vals_, counts_ = np.unique(y_train, return_counts=True)

y_vals_ros_, counts_ros_ = np.unique(y_train_standarized_upsampled, return_counts=True)

print(' Classes in the train set originally were:',dict(zip(y_vals_, counts_)),'\n',

      'Classes in the rebalanced train set are now:',dict(zip(y_vals_ros_, counts_ros_)))
#X_test_standarized
features=X_train.columns

features
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 50, stop = 700, num = 10)]

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(start=10, stop=100, num = 10)]

# Creating the random grid

random_grid = {'n_estimators': n_estimators,

               'max_depth': max_depth}

pprint(random_grid)
# Using the random grid to search for best hyperparameters

# First creating the base model to tune

random_forest = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

random_forest_random = RandomizedSearchCV(estimator = random_forest, param_distributions = random_grid, n_iter = 20, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

random_forest_random.fit(X_train[features], y_train)
random_forest_random.best_params_
random_forest_random_tunned = RandomForestClassifier(max_depth=40, n_estimators=700, random_state=42)

random_forest_random_tunned.fit(X_train, y_train)

plot_feature_importances(random_forest_random_tunned, feature_names=features, figsize=(40, 20));
X_train.head()
from sklearn.model_selection import RandomizedSearchCV

from pprint import pprint
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 50, stop = 700, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(start=10, stop=100, num = 10)]

# Creating the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth}

pprint(random_grid)
# Using the random grid to search for best hyperparameters

# First creating the base model to tune

xg_boost = xgb.XGBClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

xg_boost_random = RandomizedSearchCV(estimator = xg_boost, param_distributions = random_grid, n_iter = 20, 

                                     cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

xg_boost_random.fit(X_train, y_train)
xg_boost_random.best_params_
xg_boost_random_tunned = xgb.XGBClassifier(n_estimators=50, max_depth=20, random_state=42)

xg_boost_random_tunned.fit(X_train, y_train)

explain_weights_xgboost(xg_boost_random_tunned)
overlapping_variables=['Gender', 'Geography', 'HasCrCard', 'IsActiveMember', 'NumOfProducts', 'Tenure']
features = ['Age', 'Balance', 'BalanceToSalaryRatio', 'CreditScore',

       'EstimatedSalary', 'ProductsToBalance', 'SalaryToAge', 'ScoreToBalance',

        'Gender_0', 'Gender_1', 'Geography_0', 'Geography_1',

       'Geography_2', 'HasCrCard_0', 'HasCrCard_1', 'IsActiveMember_0',

       'IsActiveMember_1', 'NumOfProducts_0', 'NumOfProducts_1',

       'NumOfProducts_2', 'Tenure_0', 'Tenure_1', 'Tenure_2', 'Tenure_3',

       'Tenure_4']

len(features)
random_forest_selected_features = RandomForestClassifier(max_depth=40, n_estimators=122, random_state=42)

random_forest_selected_features.fit(X_train[features], y_train)

plot_feature_importances(random_forest_selected_features, feature_names=features, figsize=(40, 20));
xg_boost_random_selected_features = xgb.XGBClassifier(n_estimators=50, max_depth=20, random_state=42)

xg_boost_random_selected_features.fit(X_train[features], y_train)

explain_weights_xgboost(xg_boost_random_selected_features)
col_names =  ['Model', 'Precision', 'Recall', 'F1-score', 'Accuracy']

model_comparison = pd.DataFrame(columns = col_names)

model_comparison
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report



# logistic regression object 

lr = LogisticRegression() 

# Training the model on train set 

lr.fit(X_train[features], y_train) 

# Predicting on test set

predictions_logistic_regression = lr.predict(X_test[features]) 



print(classification_report(y_test, predictions_logistic_regression)) 



#Extracting the results

logistic_regression_report = classification_report(y_test, predictions_logistic_regression, output_dict=True )

precision_logistic_regression =  logistic_regression_report['macro avg']['precision'] 

recall_logistic_regression = logistic_regression_report['macro avg']['recall']    

f1_score_logistic_regression = logistic_regression_report['macro avg']['f1-score']

accuracy_logistic_regression = logistic_regression_report['accuracy']
cm_logistic_regression = confusion_matrix(y_test, predictions_logistic_regression)



sns.heatmap(cm_logistic_regression, annot = True, fmt = 'd')
def logistic_regression_cv(X_train, y_train, features, k):

    train_roc_auc, test_roc_auc, iteration = [], [], []

    i = 1

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train, test in kf.split(X_train.index.values):

        # Logistic regression model    

        lr = LogisticRegression()        

        lr.fit(X_train.iloc[train][features],y_train.iloc[train]) 

        #Predictions on train and test set from cross val

        preds_train=lr.predict(X_train.iloc[train][features])

        preds_test = lr.predict(X_train.iloc[test][features])

        train_roc_auc.append(roc_auc_score(y_train.iloc[train], preds_train))

        test_roc_auc.append(roc_auc_score(y_train.iloc[test], preds_test))

        iteration.append(i)

        i+=1  

    columns = {'Iteration': iteration, 'Train ROC AUC': train_roc_auc, 'Test ROC AUC': test_roc_auc}

    results = pd.DataFrame.from_dict(columns)

    results2 = results.drop(['Iteration'], axis=1)

    results2.boxplot()

    results.loc[len(results)] = ["Mean", np.mean(train_roc_auc), np.mean(test_roc_auc)]

    display(results)
logistic_regression_cv(X_train, y_train, features, 5)
logistic_regression_result= ['logistic regression', precision_logistic_regression, recall_logistic_regression, f1_score_logistic_regression,

                                 accuracy_logistic_regression]

model_comparison.loc[len(model_comparison)] = logistic_regression_result

model_comparison
sns.countplot(y_train)

y_train.value_counts()
# Dividing by 0's and 1's

y_train_0 = y_train[y_train == 0]

y_train_1 = y_train[y_train== 1]
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)

X_train_res, y_train_res = ros.fit_sample(X_train, y_train)

y_vals, counts = np.unique(y_train, return_counts=True)

y_vals_ros, counts_ros = np.unique(y_train_res, return_counts=True)

print(' Classes in the train set originally were:',dict(zip(y_vals, counts)),'\n',

      'Classes in the rebalanced train set now are:',dict(zip(y_vals_ros, counts_ros)))
# logistic regression object with over sampling

lr2 = LogisticRegression() 

# Training the model on train set 

lr2.fit(X_train_res[features], y_train_res) 

# Predicting on test set

predictions_logistic_regression_up = lr2.predict(X_test[features]) 



# print classification report 

print(classification_report(y_test, predictions_logistic_regression_up)) 



#Extracting the results

logistic_regression_report_up = classification_report(y_test, predictions_logistic_regression_up, output_dict=True )

precision_logistic_regression_up =  logistic_regression_report_up['macro avg']['precision'] 

recall_logistic_regression_up = logistic_regression_report_up['macro avg']['recall']    

f1_score_logistic_regression_up = logistic_regression_report_up['macro avg']['f1-score']

accuracy_logistic_regression_up = logistic_regression_report_up['accuracy']
cm_logistic_regression_up = confusion_matrix(y_test, predictions_logistic_regression_up)



sns.heatmap(cm_logistic_regression_up, annot = True, fmt = 'd')
logistic_regression_cv(X_train_res, y_train_res, features, 5)
logistic_regression_result_up= ['logistic regression upsampled', precision_logistic_regression_up, 

                             recall_logistic_regression_up, f1_score_logistic_regression_up,

                            accuracy_logistic_regression_up]

model_comparison.loc[len(model_comparison)] = logistic_regression_result_up

model_comparison
#X_train_standarized_upsampled

#y_train_standarized_upsampled

#X_test_standarized
# logistic regression object with over sampling

lr2 = LogisticRegression() 

# Training the model on train set 

lr2.fit(X_train_standarized_upsampled[features], y_train_standarized_upsampled) 

# Predicting on test set

predictions_logistic_regression_std_up = lr2.predict(X_test_standarized[features]) 



# print classification report 

print(classification_report(y_test, predictions_logistic_regression_std_up)) 



#Extracting the results

logistic_regression_report_std_up = classification_report(y_test, predictions_logistic_regression_std_up, output_dict=True )

precision_logistic_regression_std_up =  logistic_regression_report_std_up['macro avg']['precision'] 

recall_logistic_regression_std_up = logistic_regression_report_std_up['macro avg']['recall']    

f1_score_logistic_regression_std_up = logistic_regression_report_std_up['macro avg']['f1-score']

accuracy_logistic_regression_std_up = logistic_regression_report_std_up['accuracy']
cm_logistic_regression_std_up = confusion_matrix(y_test, predictions_logistic_regression_std_up)



sns.heatmap(cm_logistic_regression_std_up, annot = True, fmt = 'd')
logistic_regression_cv(X_train_standarized_upsampled, y_train_standarized_upsampled, features, 5)
logistic_regression_result_std_up= ['logistic regression standarized upsampled', precision_logistic_regression_std_up, 

                                    recall_logistic_regression_std_up, f1_score_logistic_regression_std_up,

                                 accuracy_logistic_regression_std_up]

model_comparison.loc[len(model_comparison)] = logistic_regression_result_std_up

model_comparison
# Creating the parameter grid

# Number of features to consider at every split

max_features = list(range(1,X_train.shape[1]))

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(start= 10, stop= 100, num = 10)]

#min_samples_split

min_samples_split=[int(x) for x in np.linspace(start = 0.1, stop = 10, num = 10)]     

#min_samples_leaf                       

min_samples_leaf=[int(x) for x in np.linspace(start = 0.1, stop = 10, num = 10)]                      

                           

# Creating the random grid

random_grid = {

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf

              }

#pprint(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

decision_tree_classifier = DecisionTreeClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

decision_tree_classifier_random = RandomizedSearchCV(estimator = decision_tree_classifier, 

                                                     param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, 

                                                     random_state=42, n_jobs = -1)

# Fit the random search model

decision_tree_classifier_random.fit(X_train[features], y_train)
decision_tree_classifier_random.best_params_

best_parameters = decision_tree_classifier_random.best_params_

pd.DataFrame(best_parameters.values(),best_parameters.keys(),columns=['Tuned Parameters'])
from sklearn.tree import DecisionTreeClassifier



# Defining the model:

decision_tree_classifier_tunned = DecisionTreeClassifier(min_samples_split= 10,

                                                         min_samples_leaf= 10,

                                                         max_features= 19,

                                                         max_depth= 10, 

                                                         random_state=42)

# Training the model:

decision_tree_classifier_tunned.fit(X_train[features], y_train)



# Predicting on test set:

tree_predictions = decision_tree_classifier_tunned.predict(X_test[features])



# Printing the classification report 

print(classification_report(y_test, tree_predictions)) 



#Extracting the results

decision_tree_report = classification_report(y_test, tree_predictions, output_dict=True )

precision_decision_tree =  decision_tree_report['macro avg']['precision'] 

recall_decision_tree = decision_tree_report['macro avg']['recall']    

f1_score_decision_tree = decision_tree_report['macro avg']['f1-score']

accuracy_decision_tree = decision_tree_report['accuracy']
cm_decision_tree = confusion_matrix(y_test, tree_predictions)



sns.heatmap(cm_decision_tree, annot = True, fmt = 'd')
def decision_tree_classifier_cv(X_train, y_train, features, k):

    train_roc_auc, test_roc_auc, iteration = [], [], []

    i = 1

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train, test in kf.split(X_train.index.values):

        # Model    

        decision_tree_classifier = DecisionTreeClassifier(min_samples_split= 10,

                                                         min_samples_leaf= 10,

                                                         max_features= 19,

                                                         max_depth= 10, 

                                                         random_state=42)       

        decision_tree_classifier.fit(X_train.iloc[train][features],y_train.iloc[train]) 

        #Predictions on train and test set from cross val

        preds_train=decision_tree_classifier.predict(X_train.iloc[train][features])

        preds_test = decision_tree_classifier.predict(X_train.iloc[test][features])

        train_roc_auc.append(roc_auc_score(y_train.iloc[train], preds_train))

        test_roc_auc.append(roc_auc_score(y_train.iloc[test], preds_test))

        iteration.append(i)

        i+=1  

    columns = {'Iteration': iteration, 'Train ROC AUC': train_roc_auc, 'Test ROC AUC': test_roc_auc}

    results = pd.DataFrame.from_dict(columns)

    results2 = results.drop(['Iteration'], axis=1)

    results2.boxplot()

    results.loc[len(results)] = ["Mean", np.mean(train_roc_auc), np.mean(test_roc_auc)]

    display(results)
decision_tree_classifier_cv(X_train, y_train, features, 5)
decision_tree_result= ['decision tree', precision_decision_tree, 

                                    recall_decision_tree, f1_score_decision_tree,

                                 accuracy_decision_tree]

model_comparison.loc[len(model_comparison)] = decision_tree_result

model_comparison
# First create the base model to tune

decision_tree_classifier_up = DecisionTreeClassifier()

# Using the random grid to search for best hyperparameters

decision_tree_classifier_random_up = RandomizedSearchCV(estimator = decision_tree_classifier_up, 

                                    param_distributions = random_grid, n_iter = 100, cv = 3, 

                                    verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

decision_tree_classifier_random_up.fit(X_train_res[features], y_train_res)



decision_tree_best_parameters_up = decision_tree_classifier_random_up.best_params_

pd.DataFrame(decision_tree_best_parameters_up.values(),decision_tree_best_parameters_up.keys(),columns=['Tuned Parameters'])
# Defining the model:

decision_tree_classifier_up = DecisionTreeClassifier(min_samples_split= 6,

                                                         min_samples_leaf= 1,

                                                         max_features= 21,

                                                         max_depth= 40, 

                                                         random_state=42)

# Training the model:

decision_tree_classifier_up.fit(X_train_res[features], y_train_res)



# Predicting on test set:

tree_predictions_up = decision_tree_classifier_up.predict(X_test[features])



# Printing the classification report 

print(classification_report(y_test, tree_predictions_up)) 



#Extracting the results

decision_tree_report_up = classification_report(y_test, tree_predictions_up, output_dict=True )

precision_decision_tree_up =  decision_tree_report_up['macro avg']['precision'] 

recall_decision_tree_up = decision_tree_report_up['macro avg']['recall']    

f1_score_decision_tree_up = decision_tree_report_up['macro avg']['f1-score']

accuracy_decision_tree_up = decision_tree_report_up['accuracy']
cm_decision_tree_up = confusion_matrix(y_test, tree_predictions_up)



sns.heatmap(cm_decision_tree_up, annot = True, fmt = 'd')
decision_tree_classifier_cv(X_train_res, y_train_res, features, 5)
decision_tree_result_up= ['decision tree upsampled', precision_decision_tree_up, 

                                    recall_decision_tree_up, f1_score_decision_tree_up,

                                    accuracy_decision_tree_up]



model_comparison.loc[len(model_comparison)] = decision_tree_result_up

model_comparison
# First create the base model to tune

decision_tree_classifier_std_up = DecisionTreeClassifier()



# Using the random grid to search for best hyperparameters

decision_tree_classifier_random_std_up = RandomizedSearchCV(estimator = decision_tree_classifier_std_up, 

                                    param_distributions = random_grid, n_iter = 100, cv = 3, 

                                    verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

decision_tree_classifier_random_std_up.fit(X_train_standarized_upsampled[features], y_train_standarized_upsampled)



decision_tree_best_parameters_std_up = decision_tree_classifier_random_std_up.best_params_

pd.DataFrame(decision_tree_best_parameters_std_up.values(),decision_tree_best_parameters_std_up.keys(),

             columns=['Tuned Parameters'])
# Defining the model

decision_tree_classifier_std_up = DecisionTreeClassifier(min_samples_split= 6,

                                                         min_samples_leaf= 1,

                                                         max_features= 21,

                                                         max_depth= 40,

                                                         random_state=42)

# Training the model with oversampling

decision_tree_classifier_std_up.fit(X_train_standarized_upsampled[features], y_train_standarized_upsampled)



# Predicting on test set:

tree_pred_std_up = decision_tree_classifier_std_up.predict(X_test_standarized[features])



# Printing the classification report 

print(classification_report(y_test, tree_pred_std_up)) 



#Extracting the results

decision_tree_report_std_up = classification_report(y_test, tree_pred_std_up, output_dict=True )

precision_decision_tree_std_up =  decision_tree_report_std_up['macro avg']['precision'] 

recall_decision_tree_std_up = decision_tree_report_std_up['macro avg']['recall']    

f1_score_decision_tree_std_up = decision_tree_report_std_up['macro avg']['f1-score']

accuracy_decision_tree_std_up = decision_tree_report_std_up['accuracy']
cm_decision_tree_std_up = confusion_matrix(y_test, tree_pred_std_up)



sns.heatmap(cm_decision_tree_std_up, annot = True, fmt = 'd')
decision_tree_classifier_cv(X_train_standarized_upsampled, y_train_standarized_upsampled, features, 5)
decision_tree_result_std_up= ['decision tree standarized upsampled', precision_decision_tree_std_up, 

                                    recall_decision_tree_std_up, f1_score_decision_tree_std_up,

                                    accuracy_decision_tree_std_up]



model_comparison.loc[len(model_comparison)] = decision_tree_result_std_up

model_comparison
# Creating the parameter grid



n_estimators = [int(x) for x in np.linspace(start = 50, stop = 700, num = 10)]

# Number of features to consider at every split

max_features = list(range(1,X_train[features].shape[1]))

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(start= 10, stop= 100, num = 10)]

#min_samples_split

min_samples_split=[int(x) for x in np.linspace(start = 1, stop = 10, num = 10)]     

#min_samples_leaf                       

min_samples_leaf=[int(x) for x in np.linspace(start = 1, stop = 10, num = 10)]                      

                           

# Creating the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf

              }

#pprint(random_grid)
# Using  random grid to search for best hyperparameters

# First create the base model to tune

random_forest_classifier = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

random_forest_classifier_random = RandomizedSearchCV(estimator = random_forest_classifier, 

                                                     param_distributions = random_grid, n_iter = 100, 

                                                     cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

random_forest_classifier_random.fit(X_train[features], y_train)
random_forest_classifier_random.best_params_

best_parameters = random_forest_classifier_random.best_params_

pd.DataFrame(best_parameters.values(),best_parameters.keys(),columns=['Tuned Parameters'])
from sklearn.ensemble import RandomForestClassifier



# Defining the model:

random_forest_classifier_tunned = RandomForestClassifier(n_estimators=627, min_samples_split=8, min_samples_leaf=7,

                                                         max_features= 20, max_depth= 70,  random_state=42)

# Training the model:

random_forest_classifier_tunned.fit(X_train[features], y_train)



# Predicting on test set:

random_tree_pred= random_forest_classifier_tunned.predict(X_test[features])



# Printing the classification report 

print(classification_report(y_test, random_tree_pred)) 



#Extracting the results

random_tree_report = classification_report(y_test, random_tree_pred, output_dict=True )

precision_random_tree =  random_tree_report['macro avg']['precision'] 

recall_random_tree = random_tree_report['macro avg']['recall']    

f1_score_random_tree = random_tree_report['macro avg']['f1-score']

accuracy_random_tree = random_tree_report['accuracy']
cm_random_tree = confusion_matrix(y_test, random_tree_pred)



sns.heatmap(cm_random_tree, annot = True, fmt = 'd')
def random_forest_classifier_cv(X_train, y_train, features, k):

    train_roc_auc, test_roc_auc, iteration = [], [], []

    i = 1

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train, test in kf.split(X_train.index.values):

        # Model    

        random_forest_classifier = RandomForestClassifier(n_estimators=627, min_samples_split=8, min_samples_leaf=7,

                                                         max_features= 20, max_depth= 70,  random_state=42)

        random_forest_classifier.fit(X_train.iloc[train][features],y_train.iloc[train]) 

        #Predictions on train and test set from cross val

        preds_train=random_forest_classifier.predict(X_train.iloc[train][features])

        preds_test = random_forest_classifier.predict(X_train.iloc[test][features])

        train_roc_auc.append(roc_auc_score(y_train.iloc[train], preds_train))

        test_roc_auc.append(roc_auc_score(y_train.iloc[test], preds_test))

        iteration.append(i)

        i+=1  

    columns = {'Iteration': iteration, 'Train ROC AUC': train_roc_auc, 'Test ROC AUC': test_roc_auc}

    results = pd.DataFrame.from_dict(columns)

    results2 = results.drop(['Iteration'], axis=1)

    results2.boxplot()

    results.loc[len(results)] = ["Mean", np.mean(train_roc_auc), np.mean(test_roc_auc)]

    display(results)
random_forest_classifier_cv(X_train, y_train, features, 5)
random_tree_result = ['random forest', precision_random_tree, 

                              recall_random_tree, f1_score_random_tree,

                              accuracy_random_tree]



model_comparison.loc[len(model_comparison)] = random_tree_result

model_comparison
# Using the random grid to search for best hyperparameters

# First create the base model to tune

random_forest_classifier_up = RandomForestClassifier()



random_forest_classifier_random_up = RandomizedSearchCV(estimator = random_forest_classifier_up, 

                                                     param_distributions = random_grid, 

                                                     n_iter = 100, cv = 3, verbose=2, random_state=42, 

                                                     n_jobs = -1)



# Fit the random search model

random_forest_classifier_random_up.fit(X_train_res[features], y_train_res)



random_forest_best_parameters_up = random_forest_classifier_random_up.best_params_

pd.DataFrame(random_forest_best_parameters_up.values(),random_forest_best_parameters_up.keys(),columns=['Tuned Parameters'])
# Defining the model:

random_forest_classifier_oversampled = RandomForestClassifier(n_estimators=555, min_samples_split=3, min_samples_leaf=1,

                                       max_features= 3, max_depth= 90,  random_state=42)



# Training the model with oversampling:

random_forest_classifier_oversampled.fit(X_train_res[features], y_train_res)



# Predicting on test set:

random_tree_pred_up =random_forest_classifier_oversampled.predict(X_test[features])



# Printing the classification report 

print(classification_report(y_test, random_tree_pred_up)) 



#Extracting the results

random_tree_report_up = classification_report(y_test, random_tree_pred_up, output_dict=True )

precision_random_tree_up= random_tree_report_up['macro avg']['precision'] 

recall_random_tree_up   = random_tree_report_up['macro avg']['recall']    

f1_score_random_tree_up = random_tree_report_up['macro avg']['f1-score']

accuracy_random_tree_up = random_tree_report_up['accuracy']
cm_random_tree_up = confusion_matrix(y_test, random_tree_pred_up)



sns.heatmap(cm_random_tree_up, annot = True, fmt = 'd')
random_forest_classifier_cv(X_train_res, y_train_res, features, 5)
random_tree_result_up= ['random forest upsampled', precision_random_tree_up, 

                              recall_random_tree_up, f1_score_random_tree_up,

                              accuracy_random_tree_up]



model_comparison.loc[len(model_comparison)] = random_tree_result_up

model_comparison
# Using the random grid to search for best hyperparameters

# First create the base model to tune

random_forest_classifier_std_up = RandomForestClassifier()



random_forest_classifier_random_std_up = RandomizedSearchCV(estimator = random_forest_classifier_std_up, 

                                                     param_distributions = random_grid, 

                                                     n_iter = 100, cv = 3, verbose=2, random_state=42, 

                                                     n_jobs = -1)



# Fit the random search model

random_forest_classifier_random_std_up.fit(X_train_standarized_upsampled[features], y_train_standarized_upsampled)



random_forest_best_parameters_std_up = random_forest_classifier_random_std_up.best_params_

pd.DataFrame(random_forest_best_parameters_std_up.values(),random_forest_best_parameters_std_up.keys(),columns=['Tuned Parameters'])
# Defining the model:

random_forest_classifier_std_up = RandomForestClassifier(n_estimators=555, min_samples_split=3, min_samples_leaf=1,

                                       max_features= 3, max_depth= 90,  random_state=42)



# Training the model with oversampling and standarization:

random_forest_classifier_std_up.fit(X_train_standarized_upsampled[features], y_train_standarized_upsampled)



# Predicting on test set:

random_tree_pred_std_up=random_forest_classifier_std_up.predict(X_test_standarized[features])



# Printing the classification report 

print(classification_report(y_test, random_tree_pred_std_up)) 



#Extracting the results

random_tree_report_std_up = classification_report(y_test, random_tree_pred_std_up, output_dict=True )

precision_random_tree_std_up= random_tree_report_std_up['macro avg']['precision'] 

recall_random_tree_std_up   = random_tree_report_std_up['macro avg']['recall']    

f1_score_random_tree_std_up = random_tree_report_std_up['macro avg']['f1-score']

accuracy_random_tree_std_up= random_tree_report_std_up['accuracy']
cm_random_tree_std_up = confusion_matrix(y_test, random_tree_pred_std_up)



sns.heatmap(cm_random_tree_std_up, annot = True, fmt = 'd')
random_forest_classifier_cv(X_train_standarized_upsampled, y_train_standarized_upsampled, features, 5)
random_tree_result_std_up= ['random forest standarized upsampled', precision_random_tree_std_up, 

                        recall_random_tree_std_up, f1_score_random_tree_std_up,

                        accuracy_random_tree_std_up]



model_comparison.loc[len(model_comparison)] = random_tree_result_std_up

model_comparison
kernel = ['linear', 'rbf', 'poly']



# Creating the random grid

random_grid = {'C': [0.1, 1, 10, 100, 1000],  

              'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}  



#pprint(random_grid)
from sklearn.svm import SVC

# Using  random grid to search for best hyperparameters

# First create the base model to tune

svm_classifier = SVC(random_state=42)

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

random_svm = RandomizedSearchCV(estimator = svm_classifier, param_distributions = random_grid, n_iter = 100, 

                                cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

random_svm.fit(X_train[features], y_train)
random_svm.best_params_

best_parameters = random_svm.best_params_

pd.DataFrame(best_parameters.values(),best_parameters.keys(),columns=['Tuned Parameters'])
from sklearn.svm import SVC



# Defining the model: SVM

svm_tunned = SVC(gamma=1, C=0.1 ,random_state=42)



# Training the model:

svm_tunned.fit(X_train[features], y_train)



# Predicting on test set:

svm_pred=svm_tunned.predict(X_test[features])



# Printing the classification report 

print(classification_report(y_test, svm_pred)) 



#Extracting the results

svm_report = classification_report(y_test, svm_pred, output_dict=True )

precision_svm= svm_report['macro avg']['precision'] 

recall_svm   = svm_report['macro avg']['recall']    

f1_score_svm = svm_report['macro avg']['f1-score']

accuracy_svm= svm_report['accuracy']
cm_svm = confusion_matrix(y_test, svm_pred)



sns.heatmap(cm_svm, annot = True, fmt = 'd')
def svm_cv(X_train, y_train, features, k):

    train_roc_auc, test_roc_auc, iteration = [], [], []

    i = 1

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train, test in kf.split(X_train.index.values):

        # Model    

        svm_classifier = SVC(gamma=1, C=0.1 ,random_state=45)

        svm_classifier.fit(X_train.iloc[train][features], y_train.iloc[train]) 

        #Predictions on train and test set from cross val

        preds_train= svm_classifier.predict(X_train.iloc[train][features])

        preds_test = svm_classifier.predict(X_train.iloc[test][features])

        train_roc_auc.append(roc_auc_score(y_train.iloc[train], preds_train))

        test_roc_auc.append(roc_auc_score(y_train.iloc[test], preds_test))

        iteration.append(i)

        i+=1  

    columns = {'Iteration': iteration, 'Train ROC AUC': train_roc_auc, 'Test ROC AUC': test_roc_auc}

    results = pd.DataFrame.from_dict(columns)

    results2 = results.drop(['Iteration'], axis=1)

    results2.boxplot()

    results.loc[len(results)] = ["Mean", np.mean(train_roc_auc), np.mean(test_roc_auc)]

    display(results)
svm_cv(X_train, y_train, features, 5)
svm_result= ['svm', precision_svm, recall_svm, f1_score_svm, accuracy_svm]



model_comparison.loc[len(model_comparison)] = svm_result

model_comparison
# Using the random grid to search for best hyperparameters



# First create the base model to tune

svm_classifier_up = SVC(random_state=42)

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

random_svm_up = RandomizedSearchCV(estimator = svm_classifier_up, param_distributions = random_grid, n_iter = 100, 

                                cv = 3, verbose=2, random_state=42, n_jobs = -1)



# Fit the random search model

random_svm_up.fit(X_train_res[features], y_train_res)



random_svm_best_parameters_up = random_svm_up.best_params_

pd.DataFrame(random_svm_best_parameters_up.values(),random_svm_best_parameters_up.keys(),columns=['Tuned Parameters'])
# Defining the model:

svm_up = SVC(gamma=1, C=1 ,random_state=42)



# Training the model with upsampling:

svm_up.fit(X_train_res[features], y_train_res)



# Predicting on test set:

svm_pred_up =svm_up.predict(X_test[features])



# Printing the classification report 

print(classification_report(y_test, svm_pred_up)) 



#Extracting the results

svm_report_up = classification_report(y_test, svm_pred_up, output_dict=True )

precision_svm_up = svm_report_up['macro avg']['precision'] 

recall_svm_up   = svm_report_up['macro avg']['recall']    

f1_score_svm_up = svm_report_up['macro avg']['f1-score']

accuracy_svm_up= svm_report_up['accuracy']
cm_svm_up = confusion_matrix(y_test, svm_pred_up)



sns.heatmap(cm_svm_up, annot = True, fmt = 'd')
svm_cv(X_train_res, y_train_res, features, 5)
svm_result_up= ['svm upsampled', precision_svm_up, recall_svm_up, f1_score_svm_up, accuracy_svm_up]



model_comparison.loc[len(model_comparison)] = svm_result_up

model_comparison
# Using the random grid to search for best hyperparameters



# First create the base model to tune

svm_classifier_std_up = SVC(random_state=42)

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

random_svm_std_up = RandomizedSearchCV(estimator = svm_classifier_std_up, param_distributions = random_grid, n_iter = 100, 

                                cv = 3, verbose=2, random_state=42, n_jobs = -1)



# Fit the random search model

random_svm_std_up.fit(X_train_standarized_upsampled[features], y_train_standarized_upsampled)



random_svm_best_parameters_std_up = random_svm_std_up.best_params_

pd.DataFrame(random_svm_best_parameters_std_up.values(),

             random_svm_best_parameters_std_up.keys(),columns=['Tuned Parameters'])
# Defining the model:

svm_classifier_std_up = SVC(gamma=1, C=10 ,random_state=42)



# Training the model with oversampling:

svm_classifier_std_up.fit(X_train_standarized_upsampled[features], y_train_standarized_upsampled)



# Predicting:

svm_pred_std_up=svm_classifier_std_up.predict(X_test_standarized[features])



# Printing the classification report 

print(classification_report(y_test, svm_pred_std_up)) 



#Extracting the results

svm_report_std_up = classification_report(y_test, svm_pred_std_up, output_dict=True )

precision_svm_std_up = svm_report_std_up['macro avg']['precision'] 

recall_svm_std_up   = svm_report_std_up['macro avg']['recall']    

f1_score_svm_std_up = svm_report_std_up['macro avg']['f1-score']

accuracy_svm_std_up= svm_report_std_up['accuracy']
cm_svm_std_up = confusion_matrix(y_test, svm_pred_std_up)



sns.heatmap(cm_svm_std_up, annot = True, fmt = 'd')
svm_cv(X_train_standarized_upsampled, y_train_standarized_upsampled, features, 5)
svm_result_std_up= ['svm standarized upsampled', precision_svm_std_up, 

                    recall_svm_std_up, f1_score_svm_std_up, accuracy_svm_std_up]



model_comparison.loc[len(model_comparison)] = svm_result_std_up

model_comparison
# Defining the model: Xg-boost

xg_boost_tunned= xgb.XGBClassifier(n_estimators=50, max_depth=20, random_state=42)



#Training the model:

xg_boost_tunned.fit(X_train[features], y_train)



#Predicting:

xg_boost_pred=xg_boost_tunned.predict(X_test[features])



# Printing the classification report 

print(classification_report(y_test, xg_boost_pred)) 



#Extracting the results

xg_boost_report = classification_report(y_test, xg_boost_pred, output_dict=True )

precision_xg_boost = xg_boost_report['macro avg']['precision'] 

recall_xg_boost   = xg_boost_report['macro avg']['recall']    

f1_score_xg_boost = xg_boost_report['macro avg']['f1-score']

accuracy_xg_boost= xg_boost_report['accuracy']
cm_xg_boost = confusion_matrix(y_test, xg_boost_pred)



sns.heatmap(cm_xg_boost, annot = True, fmt = 'd')
def score_xg_boost_cv(X_train, y_train, features, k):

    train_roc_auc, test_roc_auc, iteration = [], [], []

    i = 1

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train, test in kf.split(X_train.index.values):

        # Model    

        xg_boost_ = xgb.XGBClassifier(n_estimators=50, max_depth=20, random_state=45)

        xg_boost_.fit(X_train.iloc[train][features],y_train.iloc[train]) 

        #Predictions on train and test set from cross val

        preds_train=xg_boost_.predict(X_train.iloc[train][features])

        preds_test = xg_boost_.predict(X_train.iloc[test][features])

        train_roc_auc.append(roc_auc_score(y_train.iloc[train], preds_train))

        test_roc_auc.append(roc_auc_score(y_train.iloc[test], preds_test))

        iteration.append(i)

        i+=1  

    columns = {'Iteration': iteration, 'Train ROC AUC': train_roc_auc, 'Test ROC AUC': test_roc_auc}

    results = pd.DataFrame.from_dict(columns)

    results2 = results.drop(['Iteration'], axis=1)

    results2.boxplot()

    results.loc[len(results)] = ["Mean", np.mean(train_roc_auc), np.mean(test_roc_auc)]

    display(results)
score_xg_boost_cv(X_train, y_train, features, 5)
xg_boost_result= ['XgBoost', precision_xg_boost, 

                    recall_xg_boost, f1_score_xg_boost, accuracy_xg_boost]



model_comparison.loc[len(model_comparison)] = xg_boost_result

model_comparison
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 50, stop = 700, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(start=10, stop=100, num = 10)]

# Creating the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth}

#pprint(random_grid)
# Using the random grid to search for best hyperparameters

# First creating the base model to tune

xg_boost_up = xgb.XGBClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

xg_boost_random_up = RandomizedSearchCV(estimator = xg_boost_up, param_distributions = random_grid, n_iter = 20, 

                                     cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

xg_boost_random_up.fit(X_train_res, y_train_res)



xg_boost_up_best_parameters_up = xg_boost_random_up.best_params_

pd.DataFrame(xg_boost_up_best_parameters_up.values(),xg_boost_up_best_parameters_up.keys(),columns=['Tuned Parameters'])
# Defining the model: Xg-boost

xg_boost_tunned_up= xgb.XGBClassifier(n_estimators=411, max_features='sqrt' ,max_depth=10, random_state=42)



#Training the model:

xg_boost_tunned_up.fit(X_train_res[features], y_train_res)



#Predicting:

xg_boost_pred_up=xg_boost_tunned_up.predict(X_test[features])



# Printing the classification report 

print(classification_report(y_test, xg_boost_pred_up)) 



#Extracting the results

xg_boost_report_up = classification_report(y_test, xg_boost_pred_up, output_dict=True )

precision_xg_boost_up = xg_boost_report_up['macro avg']['precision'] 

recall_xg_boost_up   = xg_boost_report_up['macro avg']['recall']    

f1_score_xg_boost_up = xg_boost_report_up['macro avg']['f1-score']

accuracy_xg_boost_up = xg_boost_report_up['accuracy']
cm_xg_boost_up = confusion_matrix(y_test, xg_boost_pred_up)



sns.heatmap(cm_xg_boost_up, annot = True, fmt = 'd')
score_xg_boost_cv(X_train_res, y_train_res, features, 5)
xg_boost_result_up= ['XgBoost upsampled', precision_xg_boost_up, 

                    recall_xg_boost_up, f1_score_xg_boost_up, accuracy_xg_boost_up]



model_comparison.loc[len(model_comparison)] = xg_boost_result_up

model_comparison
# Using the random grid to search for best hyperparameters



# First creating the base model to tune

xg_boost_std_up = xgb.XGBClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

xg_boost_random_std_up = RandomizedSearchCV(estimator = xg_boost_std_up, param_distributions = random_grid, n_iter = 20, 

                                     cv = 3, verbose=2, random_state=42, n_jobs = -1)



# Fit the random search model

xg_boost_random_std_up.fit(X_train_standarized_upsampled[features], y_train_standarized_upsampled)



xg_boost_best_parameters_std_up = xg_boost_random_std_up.best_params_

pd.DataFrame(xg_boost_best_parameters_std_up.values(),xg_boost_best_parameters_std_up.keys(),columns=['Tuned Parameters'])
# Defining the model: Xg-boost

xg_boost_tunned_std_up= xgb.XGBClassifier(n_estimators=338, max_features='sqrt', max_depth=90, random_state=42)



#Training the model:

xg_boost_tunned_std_up.fit(X_train_standarized_upsampled[features], y_train_standarized_upsampled)



#Predicting:

xg_boost_pred_std_up=xg_boost_tunned_up.predict(X_test_standarized[features])



# Printing the classification report 

print(classification_report(y_test, xg_boost_pred_std_up)) 



#Extracting the results

xg_boost_report_std_up = classification_report(y_test, xg_boost_pred_std_up, output_dict=True )

precision_xg_boost_std_up = xg_boost_report_std_up['macro avg']['precision'] 

recall_xg_boost_std_up  = xg_boost_report_std_up['macro avg']['recall']    

f1_score_xg_boost_std_up = xg_boost_report_std_up['macro avg']['f1-score']

accuracy_xg_boost_std_up = xg_boost_report_std_up['accuracy']
cm_xg_boost_std_up = confusion_matrix(y_test, xg_boost_pred_std_up)



sns.heatmap(cm_xg_boost_std_up, annot = True, fmt = 'd')
score_xg_boost_cv(X_train_standarized_upsampled, y_train_standarized_upsampled, features, 5)
xg_boost_result_std_up= ['XgBoost standarized upsampled', precision_xg_boost_std_up, 

                    recall_xg_boost_std_up, f1_score_xg_boost_std_up, accuracy_xg_boost_std_up]



model_comparison.loc[len(model_comparison)] = xg_boost_result_std_up

model_comparison
import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense #Dense module is for the layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Set callback functions to early stop training and save the best model so far

callbacks = [EarlyStopping(monitor='val_loss', patience=2),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout



# Initializing the model

ann = Sequential()



# Adding the input layer and the first hidden layer

ann.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 25))





# Adding the second hidden layer

ann.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))





# Adding the output layer

ann.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN | means applying Stochastic Gradient Descent on ann

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Training the model:

ann.fit(X_train[features], y_train, batch_size = 10, epochs = 100, verbose = 0, callbacks=callbacks)



#Predicting

y_pred_ann= ann.predict(X_test[features])



# Printing the classification report 

#As y_pred_ann is continuous and our target variable is binary, to follow the results are compared to boolean values:

y_pred_ann_bin=(y_pred_ann>0.5)



print(classification_report(y_test, y_pred_ann_bin)) 



#Extracting the results

ann_report = classification_report(y_test, y_pred_ann_bin, output_dict=True )

precision_ann = ann_report['macro avg']['precision'] 

recall_ann  = ann_report['macro avg']['recall']    

f1_score_ann = ann_report['macro avg']['f1-score']

accuracy_ann = ann_report['accuracy']
cm_ann = confusion_matrix(y_test, y_pred_ann_bin)



sns.heatmap(cm_xg_boost_std_up, annot = True, fmt = 'd')
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_ann)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='ANN')

plt.title('ROC curve')

plt.show()
def ann_cv(X_train, y_train, features, k):

    train_roc_auc, test_roc_auc, iteration = [], [], []

    i = 1

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train, test in kf.split(X_train.index.values):

        

        ann_ = Sequential()

        # Model    

        ann_.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 25))



        # Adding the second hidden layer

        ann_.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))



        # Adding the output layer

        ann_.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

        

        #Predictions on train and test set from cross val

        preds_train=ann_.predict(X_train.iloc[train][features])

        preds_test = ann_.predict(X_train.iloc[test][features])

        train_roc_auc.append(roc_auc_score(y_train.iloc[train], preds_train))

        test_roc_auc.append(roc_auc_score(y_train.iloc[test], preds_test))

        iteration.append(i)

        i+=1  

    columns = {'Iteration': iteration, 'Train ROC AUC': train_roc_auc, 'Test ROC AUC': test_roc_auc}

    results = pd.DataFrame.from_dict(columns)

    results2 = results.drop(['Iteration'], axis=1)

    results2.boxplot()

    results.loc[len(results)] = ["Mean", np.mean(train_roc_auc), np.mean(test_roc_auc)]

    display(results)
ann_cv(X_train, y_train, features, 5)
ann_result= ['Ann', precision_ann, recall_ann, f1_score_ann, accuracy_ann]



model_comparison.loc[len(model_comparison)] = ann_result

model_comparison
# Initializing the model

ann_up = Sequential()



# Adding the input layer and the first hidden layer

ann_up.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 25))



# Adding the second hidden layer

ann_up.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

ann_up.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN | means applying Stochastic Gradient Descent on ann

ann_up.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Training the model:

ann_up.fit(X_train_res[features], y_train_res, batch_size = 10, epochs = 100, verbose = 0, callbacks=callbacks)



#Predicting

y_pred_ann_up= ann_up.predict(X_test[features])



# Printing the classification report 

#As y_pred_ann is continuous and our target variable is binary, to follow the results are compared to boolean values:

y_pred_ann_bin_up=(y_pred_ann_up>0.5)



print(classification_report(y_test, y_pred_ann_bin_up)) 



#Extracting the results

ann_report_up = classification_report(y_test, y_pred_ann_bin_up, output_dict=True )

precision_ann_up = ann_report_up['macro avg']['precision'] 

recall_ann_up  = ann_report_up['macro avg']['recall']    

f1_score_ann_up = ann_report_up['macro avg']['f1-score']

accuracy_ann_up = ann_report_up['accuracy']
cm_ann_up = confusion_matrix(y_test, y_pred_ann_bin_up)



sns.heatmap(cm_ann_up, annot = True, fmt = 'd')
ann_cv(X_train_res, y_train_res, features, 5)
ann_result_up= ['Ann upsampled', precision_ann_up, recall_ann_up, f1_score_ann_up, accuracy_ann_up]



model_comparison.loc[len(model_comparison)] = ann_result_up

model_comparison
# Initializing the model

ann_std_up = Sequential()



# Adding the input layer and the first hidden layer

ann_std_up.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 25))



# Adding the second hidden layer

ann_std_up.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

ann_std_up.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN | means applying Stochastic Gradient Descent on ann

ann_std_up.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Training the model:

ann_std_up.fit(X_train_standarized_upsampled[features], y_train_standarized_upsampled, batch_size = 10, epochs = 100, verbose = 0, callbacks=callbacks)



#Predicting

y_pred_ann_std_up= ann_std_up.predict(X_test_standarized[features])



# Printing the classification report 

#As y_pred_ann is continuous and our target variable is binary, to follow the results are compared to boolean values:

y_pred_ann_bin_std_up=(y_pred_ann_std_up>0.5)



print(classification_report(y_test, y_pred_ann_bin_std_up)) 



#Extracting the results

ann_report_std_up = classification_report(y_test, y_pred_ann_bin_std_up, output_dict=True )

precision_ann_std_up = ann_report_std_up['macro avg']['precision'] 

recall_ann_std_up  = ann_report_std_up['macro avg']['recall']    

f1_score_ann_std_up = ann_report_std_up['macro avg']['f1-score']

accuracy_ann_std_up = ann_report_std_up['accuracy']
cm_ann_std_up = confusion_matrix(y_test, y_pred_ann_bin_std_up)



sns.heatmap(cm_ann_std_up, annot = True, fmt = 'd')
fpr, tpr, thresholds = roc_curve(y_test, y_pred_ann_std_up)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='ANN')

plt.title('ROC curve')

plt.show()
ann_cv(X_train_standarized_upsampled, y_train_standarized_upsampled, features, 5)
ann_result_std_up= ['Ann standarized upsampled', precision_ann_std_up, recall_ann_std_up, 

                    f1_score_ann_std_up, accuracy_ann_std_up]



model_comparison.loc[len(model_comparison)] = ann_result_std_up

model_comparison
# Tuning the ANN

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier



def build_classifier(optimizer):

    classifier = Sequential()

    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 25))

    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier



classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [9 , 25, 32],

              'epochs': [100, 200],

              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 5)
grid_search = grid_search.fit(X_train[features], y_train, verbose = 0)

best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_
print('Best parameters after tuning are: {}'.format(best_parameters))

print('Best accuracy after tuning is: {}'.format(best_accuracy))
from sklearn.ensemble import AdaBoostClassifier



# Defining the model: Ada-Boost

svc=SVC(probability=True, kernel='linear')

ada_boost = AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1,random_state=42)



#Training the model:

ada_boost.fit(X_train_standarized_upsampled[features], y_train_standarized_upsampled)



#Predicting:

ada_boost_pred=ada_boost.predict(X_test_standarized[features])



# Printing the classification report 

print(classification_report(y_test, ada_boost_pred)) 



#Extracting the results

ada_boost_report = classification_report(y_test, ada_boost_pred, output_dict=True )

precision_ada_boost = ada_boost_report['macro avg']['precision'] 

recall_ada_boost  = ada_boost_report['macro avg']['recall']    

f1_score_ada_boost = ada_boost_report['macro avg']['f1-score']

accuracy_ada_boost = ada_boost_report['accuracy']
cm_ada_boost = confusion_matrix(y_test, ada_boost_pred)



sns.heatmap(cm_ada_boost, annot = True, fmt = 'd')
ada_boost_result= ['AdaBoost standarized upsampled', precision_ada_boost, recall_ada_boost, 

                    f1_score_ada_boost, accuracy_ada_boost]



model_comparison.loc[len(model_comparison)] = ada_boost_result

model_comparison
from sklearn.base import BaseEstimator, TransformerMixin, clone, RegressorMixin

class WeightedAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models, weights):

        self.models = models

        self.weights = weights

        assert sum(self.weights)==1

        

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)

        return self

    

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.sum(predictions*self.weights, axis=1)
# Defining the model: Weighted Average

weighted_average_ = WeightedAveragedModels([decision_tree_classifier_up, svm_classifier_std_up, xg_boost_tunned_up],

                                           [0.4, 0.4, 0.2])



#Training the model:

weighted_average_.fit(X_train_standarized_upsampled[features], y_train_standarized_upsampled)



#Predicting:

weighted_average_pred = weighted_average_.predict(X_test_standarized[features])



#As weighted_average_pred is continuous and our target variable is binary, to follow the results are compared to boolean values:

weighted_average_pred_bin=(weighted_average_pred>0.5)





# Printing the classification report 

print(classification_report(y_test, weighted_average_pred_bin)) 



#Extracting the results

weighted_average_report = classification_report(y_test, weighted_average_pred_bin, output_dict=True )

precision_weighted_average = weighted_average_report['macro avg']['precision'] 

recall_weighted_average  = weighted_average_report['macro avg']['recall']    

f1_score_weighted_average = weighted_average_report['macro avg']['f1-score']

accuracy_weighted_average = weighted_average_report['accuracy']
cm_weighted_average = confusion_matrix(y_test, weighted_average_pred_bin)



sns.heatmap(cm_weighted_average, annot = True, fmt = 'd')
weighted_average_result= ['Weighted Average standarized upsampled', precision_weighted_average, recall_weighted_average, 

                    f1_score_weighted_average, accuracy_weighted_average]



model_comparison.loc[len(model_comparison)] = weighted_average_result

model_comparison
multiplot = plt.figure(figsize = (18,18))



m_1 = multiplot.add_subplot(3,3,1)

m_1.set_title("Logistic Regression")

sns.heatmap(cm_logistic_regression, annot = True, fmt = 'd')



m_2 = multiplot.add_subplot(3,3,2)

m_2.set_title("Logistic Regression upsampled")

sns.heatmap(cm_logistic_regression_up, annot = True, fmt = 'd')



m_3 = multiplot.add_subplot(3,3,3)

m_3.set_title("Logistic Regression upsampled standarized")

sns.heatmap(cm_logistic_regression_std_up, annot = True, fmt = 'd')



m_4 = multiplot.add_subplot(3,3,4)

m_4.set_title("Decision Trees")

sns.heatmap(cm_decision_tree, annot = True, fmt = 'd')



m_5 = multiplot.add_subplot(3,3,5)

m_5.set_title("Decision Trees upsampled")

sns.heatmap(cm_decision_tree_up, annot = True, fmt = 'd')



m_6 = multiplot.add_subplot(3,3,6)

m_6.set_title("Decision Trees standarized upsampled")

sns.heatmap(cm_decision_tree_std_up, annot = True, fmt = 'd')



m_7 = multiplot.add_subplot(3,3,7)

m_7.set_title("Random Forest")

sns.heatmap(cm_random_tree, annot = True, fmt = 'd')



m_8 = multiplot.add_subplot(3,3,8)

m_8.set_title("Random Forest upsampled")

sns.heatmap(cm_random_tree_up, annot = True, fmt = 'd')



m_9 = multiplot.add_subplot(3,3,9)

m_9.set_title("Random Forest standarized upsampled")

sns.heatmap(cm_random_tree_std_up, annot = True, fmt = 'd')
multiplot = plt.figure(figsize = (18,18))



m_10 = multiplot.add_subplot(3,3,1)

m_10.set_title("SVM")

sns.heatmap(cm_svm, annot = True, fmt = 'd')



m_11 = multiplot.add_subplot(3,3,2)

m_11.set_title("SVM upsampled")

sns.heatmap(cm_svm_up, annot = True, fmt = 'd')



m_12 = multiplot.add_subplot(3,3,3)

m_12.set_title("SVM standarized upsampled")

sns.heatmap(cm_svm_std_up, annot = True, fmt = 'd')



m_13 = multiplot.add_subplot(3,3,4)

m_13.set_title("XgBoost")

sns.heatmap(cm_xg_boost, annot = True, fmt = 'd')



m_14 = multiplot.add_subplot(3,3,5)

m_14.set_title("XgBoost upsampled")

sns.heatmap(cm_xg_boost_up, annot = True, fmt = 'd')



m_15 = multiplot.add_subplot(3,3,6)

m_15.set_title("XgBoost upsampled standarized")

sns.heatmap(cm_xg_boost_std_up, annot = True, fmt = 'd')



m_16 = multiplot.add_subplot(3,3,7)

m_16.set_title("Ann")

sns.heatmap(cm_ann, annot = True, fmt = 'd')



m_17 = multiplot.add_subplot(3,3,8)

m_17.set_title("Ann upsampled")

sns.heatmap(cm_ann_up, annot = True, fmt = 'd')



m_18 = multiplot.add_subplot(3,3,9)

m_18.set_title("Ann standarized upsampled")

sns.heatmap(cm_ann_std_up, annot = True, fmt = 'd')

model_comparison
model_comparison.sort_values(by='Accuracy', ascending=False, na_position='first')
multiplot = plt.figure(figsize = (18,18))



e_1 = multiplot.add_subplot(3,3,1)

e_1.set_title("Ensemble: AdaBoost std and upsampled with SVC as base estimator")

sns.heatmap(cm_ada_boost, annot = True, fmt = 'd')



e_2 = multiplot.add_subplot(3,3,2)

e_2.set_title("Ensemble: Weighted Average std and upsampled")

sns.heatmap(cm_weighted_average, annot = True, fmt = 'd')
accuracy_models=[accuracy_logistic_regression, accuracy_logistic_regression_up, accuracy_logistic_regression_std_up, 

                 accuracy_decision_tree, accuracy_decision_tree, accuracy_decision_tree_up, accuracy_decision_tree_std_up,

                accuracy_random_tree, accuracy_random_tree_up, accuracy_random_tree_std_up, accuracy_svm, accuracy_svm_up,

                accuracy_svm_up, accuracy_svm_std_up, accuracy_xg_boost, accuracy_xg_boost_up, accuracy_xg_boost_std_up,

                accuracy_ann, accuracy_ann_up, accuracy_ann_std_up, accuracy_ada_boost, accuracy_weighted_average]

labels=['accuracy_logistic_regression', 'accuracy_logistic_regression_up', 'accuracy_logistic_regression_std_up', 

                 'accuracy_decision_tree', 'accuracy_decision_tree', 'accuracy_decision_tree_up', 'accuracy_decision_tree_std_up',

                'accuracy_random_tree', 'accuracy_random_tree_up', 'accuracy_random_tree_std_up', 'accuracy_svm, accuracy_svm_up',

                'accuracy_svm_up', 'accuracy_svm_std_up', 'accuracy_xg_boost', 'accuracy_xg_boost_up', 'accuracy_xg_boost_std_up',

                'accuracy_ann', 'accuracy_ann_up', 'accuracy_ann_std_up', 'accuracy_ada_boost', 'accuracy_weighted_average']

y_pos = np.arange(len(accuracy_models))
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt



plt.barh(y_pos, accuracy_models, align='center', alpha=0.5)

plt.yticks(y_pos, labels)

plt.ylabel('Accuracy performance')

plt.title('ML algorithms')

plt.show()