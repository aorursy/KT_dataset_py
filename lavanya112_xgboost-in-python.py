# importing the required modules



import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
# importing the data



df = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df['customerID'].count()
len(df['customerID'].unique())
df.drop('customerID', axis= 1, inplace = True)
df.columns
df['MultipleLines'].unique()
miss = []

for var in df.columns:

    if df[var].isnull().values.any() == True:

        miss.append(var)
print(var)
df.describe()
df.info
df.head()
df.columns = df.columns.str.replace(' ', '_')

df.head()
df.dtypes
print(df['TotalCharges'].unique())

print(len(df['TotalCharges'].unique()))
#df['TotalCharges']= pd.to_numeric(df['TotalCharges'])

# throws error 
len(df.loc[df['TotalCharges'] == ' '])
df.loc[df['TotalCharges'] == ' ']
df.loc[(df['TotalCharges'] == ' '), 'TotalCharges'] = 0
df.loc[df['tenure'] == 0]
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df.dtypes
df.replace(' ', '_', regex= True, inplace= True)

df.head()
df.columns
X = df.drop('Churn', axis = 1).copy() 

# ALTERNATE:

# X = df_no_missing.iloc[:,:-1]



X.head()
y = df['Churn'].copy()

y.head()
X.dtypes
print(df.columns)

for x in df.columns:

    print(df[x].unique())
pd.get_dummies(X, columns= ['Contract']).head()
X_encoded = pd.get_dummies(X, columns= ['gender', 

                                        'Partner', 

                                        'Dependents', 

                                        'PhoneService', 

                                        'MultipleLines', 

                                        'InternetService',

                                        'OnlineSecurity',

                                        'OnlineBackup',

                                        'DeviceProtection',

                                        'TechSupport',

                                        'StreamingTV',

                                        'StreamingMovies',

                                        'Contract',

                                        'PaperlessBilling',

                                        'PaymentMethod'])



X_encoded.head()
y.unique()
# REPLACING YESs WITH 1s and NOs with 0s

y = y.str.replace('Yes', '1')

y = y.str.replace('No', '0')

y.unique()
y = pd.to_numeric(y)
sum(y)/len(y)
X_train, X_test, y_train, y_test= train_test_split(X_encoded, y, random_state = 42, stratify = y)
print(sum(y_train)/len(y_train))

print(sum(y_test)/len(y_test))
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', missing= None, seed= 42)

xgb_clf.fit(X_train,

            y_train,

            verbose= True,

            early_stopping_rounds= 10,

            eval_metric= 'aucpr',

            eval_set= [(X_test, y_test)])
plot_confusion_matrix(xgb_clf, 

                      X_test,

                      y_test,

                      values_format= 'd',

                      display_labels=["Did not leave", "Left"])
# NOTE: When data is imbalanced, the XGBoost manual says

# If you care only about the overall performance matric (AUC) of your predictions

# -> Balance the positive and negative weights via scale_pos_weight

# -> Use AUC for evaluation.

# Running GridSearchCV() sequentially on subsets of parameter options, rather than all at once in order

# to optimize parameters in a short period of time.



## ROUND 1



param_grid= {

    'max_depth': [3, 4, 5],

    'learning_rate': [0.1, 0.01, 0.05],

    'gamma': [0, 0.25, 1.0],

    'reg_lambda': [0, 1.0, 10.0],

    'scale_pos_weight': [1, 3, 5]

}



## ROUND 2



param_grid= {

    'max_depth': [4],

    'learning_rate': [0.1, 0.5, 1],

    'gamma': [0.25],

    'reg_lambda': [10.0, 20, 100],

    'scale_pos_weight': [3]

}





# In order to speed up the Cross Validation, for each tree we are using a random subset of the actual data ie. we are not using 

# all the data. We are only using 90 % and that is randomly selected per tree. We are also only selecting per tree 50 % of the 

# columns in that dataset so for every tree we create, we select a different 50 % of the column and that helps us with overfitting

# issues as well as speeding things up considerably. Other than that we are just using AUC score and we are not doing a lot of 

# Cross Validation (not 10 fold only 3 fold).



optimal_params= GridSearchCV(

    estimator= xgb.XGBClassifier(objective= 'binary:logistic',

                                 seed= 42,

                                 subsample= 0.9,

                                 colsample_bytree= 0.5),

    param_grid= param_grid,

    scoring= 'roc_auc',

    verbose= 0,

    n_jobs= 10,

    cv= 3

)



optimal_params.fit(X_train,

                   y_train,

                   early_stopping_rounds= 10,

                   eval_metric= 'auc',

                   eval_set= [(X_test, y_test)],

                   verbose= False)



print(optimal_params.best_params_)

xgb_clf= xgb.XGBClassifier(seed= 42,

                           objective= 'binary:logistic',

                           gamma= 0.25,

                           learn_rate= 0.1,

                           max_depth= 4,

                           reg_lambda= 10,

                           scale_pos_weight= 3,

                           subsample= 0.9,

                           colsample_bytree= 0.5)



xgb_clf.fit(X_train,

            y_train,

            verbose= True,

            early_stopping_rounds= 10,

            eval_metric= 'aucpr',

            eval_set= [(X_test, y_test)])
plot_confusion_matrix(xgb_clf,

                      X_test,

                      y_test,

                      values_format= 'd',

                      display_labels= ["Did not leave", "Left"])
xgb_clf= xgb.XGBClassifier(seed= 42,

                           objective= 'binary:logistic',

                           learn_rate= 0.1,

                           max_depth= 4,

                           reg_lambda= 10,

                           scale_pos_weight= 3,

                           subsample= 0.9,

                           colsample_bytree= 0.5,

                           n_estimators= 1)

# n_estimators set to 1 so that we can get gain, cover etc.

xgb_clf.fit(X_train, y_train)
bst= xgb_clf.get_booster()



for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):

    print('%s: ' % importance_type, bst.get_score(importance_type= importance_type))

    

    

node_params= {'shape': 'box',  # makes the node fancy

              'style': 'filled, rounded',

              'fillcolor': '#78cbe'

             }



leaf_params= {'shape': 'box',

              'style': 'filled',

              'fillcolor': '#e48038'}



# NOTE: num_trees is NOT the number of trees to plot, but the specific tree that we are going to plot

# The default value is 0, but let's set it just to show it since it is counter-intuitive.

# xgb.to_graphviz(xgb_clf, num_trees= 0, size= "10, 10")



xgb.to_graphviz(xgb_clf, num_trees= 0, size= "10, 10",

                condition_node_params= node_params,

                leaf_node_params= leaf_params)



# TO SAVE THE FIGURE (in jupyter notebook):

# graph_data= xgb.to_graphviz(xgb_clf, num_trees= 0, size= "10, 10",

#                 condition_node_params= node_params,

#                 leaf_node_params= leaf_params)

# graph_data.view(filename= 'insert arbitrary file name as required')