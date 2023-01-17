import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.svm import SVC

from sklearn.model_selection import KFold

import statsmodels.api as sm

from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

import tensorflow as tf

from sklearn.metrics import roc_auc_score

from sklearn.datasets import make_classification

from tensorflow.keras.models import Sequential 

from tensorflow.keras.layers import Dense 

from tensorflow.keras.callbacks import Callback, EarlyStopping

import re

from sklearn.impute import KNNImputer

from tensorflow.keras import layers

from sklearn.feature_selection import VarianceThreshold

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

import category_encoders as ce

from statsmodels.stats.outliers_influence import variance_inflation_factor

# load the data

train_set = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

test_set = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")

# check the training set

train_set.head()
# lets segregate the features into categorical and numeric columns

cat_vars = []

num_vars = []

for col in train_set:

    if train_set[col].dtypes == 'O':

        cat_vars.append(col)

    else:

        num_vars.append(col)



# removing id and target from the list

num_vars.remove("id")

num_vars.remove("target")
# lets further segregate categorical variables in ordinal, nominal,binary and date variables

bin_vars = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

ord_vars = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5'] 

nom_vars = ['nom_0','nom_1','nom_2','nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

dat_vars = ['day','month']
# lets save training and test set "id"s for future use

train_set_id = train_set.id

test_set_id = test_set.id



# save training set target 

train_set_target = train_set.target



# lets drop targe and ids

train_set.drop(["id","target"],1,inplace= True)

test_set.drop("id",1,inplace= True)



#create master dataset from training and test datasets

master=pd.concat((train_set,test_set)).reset_index(drop=True)

master.shape
# Let's display the variables having null values

null_cols = []

for col in train_set.columns:

    if train_set[col].isnull().sum() > 0 :

        print('%s %s %d %s' %(col,"have",train_set[col].isnull().sum(),"null values"))    

        null_cols.append(col)
# lets take a look at the ordinal variables

for col in bin_vars:

    print(col, ":\n",train_set[col].value_counts(),"\n")
# missing values in bin_0/1/2 can be replaced by 0.0

master[['bin_0','bin_1','bin_2']] = master[['bin_0','bin_1','bin_2']].replace(np.nan, 0.0)



# missing values in bin_3 can be replaced with "F"

master['bin_3'] = master['bin_3'].replace(np.nan, "F")



# missing values in bin_4 can be replaced with "N"

master['bin_4'] = master['bin_4'].replace(np.nan, "N")
# lets take a look at the ordinal variables

for col in ord_vars:

    print(col, ":",train_set[col].value_counts())
# since values are evenly distributed for ordinal values, we will replace the missing values by "missing"

master[ord_vars] = master[ord_vars].replace(np.nan,"missing")

# lets take a look at the ordinal variables

for col in nom_vars:

    print(col, ":",train_set[col].value_counts())
# "Red" is most frequent value for nom_0 variable, lets replace missing values with "Red"

master['nom_0'] = master['nom_0'].replace(np.nan,"Red")



# "Theremin" is most frequent value for nom_0 variable, lets replace missing values with "Theremin"

master['nom_4'] = master['nom_4'].replace(np.nan,"Theremin")



# its difficult to make a call for other variables, so, let replace with "missing"

master[['nom_1','nom_2','nom_3','nom_5','nom_6','nom_7','nom_8','nom_9']] = master[['nom_1','nom_2','nom_3','nom_5','nom_6','nom_7','nom_8','nom_9']].replace(np.nan,"missing")
# lets take a look at the ordinal variables

for col in dat_vars:

    print(col, ":",train_set[col].value_counts())
# lets replace all the null values by -1

master[dat_vars] = master[dat_vars].replace(np.nan,0.0)
# lets confirm if there are any missing values leftr

master.isnull().sum()
# lets encode binary variables



master['bin_3'] = master['bin_3'].map({'F': 1, 'T': 0})



master['bin_4'] = master['bin_4'].map({'Y': 1, 'N': 0})
# lets apply label encoding to ordinal variables 

lbl = preprocessing.LabelEncoder()

for col in ord_vars:

    master[col] = lbl.fit_transform(master[col].astype(str).values)
#for col in high_card_nom_vars:

#    top10 = master[col].value_counts().sort_values(ascending =  False).head(10).index

#    print(top10)

#    for label in top10:

#        master[label] = np.where(master[col]==label,1,0)
# lets divide nominal variables into low and high cardinality variables 

low_card_nom_vars = []

high_card_nom_vars = []



for col in nom_vars:

    if train_set[col].nunique()>10:

        high_card_nom_vars.append(col)

    else:

        low_card_nom_vars.append(col)
# lets take a quick look at the lists created

low_card_nom_vars, high_card_nom_vars
# lets apply one hot encoding for low cardinality nominal variables

dummies = pd.get_dummies(master[low_card_nom_vars], drop_first=True)



# concat dummy variables with X

master = pd.concat([master, dummies], axis=1)



# drop categorical variables for which we already created the dummy variables

master.drop(low_card_nom_vars,1,inplace = True)
# lets apply hashing encoding on high cardinality nominal variables

ce_hash = ce.HashingEncoder(cols = high_card_nom_vars)

master = ce_hash.fit_transform(master)
master.columns
master.shape

# lets first create train_set and test_set back from the master dataset



train_set = master[:train_set.shape[0]]



test_set = master[train_set.shape[0]:]



# lets confirm the shape of train and test datasets

train_set.shape, test_set.shape
const_fltr = VarianceThreshold(threshold = 0)

const_fltr.fit(train_set)
constant_columns = [column for column in train_set.columns

                    if column not in train_set.columns[const_fltr.get_support()]]



print(len(constant_columns))
# lets check the variables with 

quasi_const_fltr = VarianceThreshold(threshold = 0.01)

quasi_const_fltr.fit(train_set)
qconstant_columns = [column for column in train_set.columns

                    if column not in train_set.columns[quasi_const_fltr.get_support()]]



print(len(qconstant_columns))
train_set_T = train_set.T

train_set_T.shape
# Removing duplicate columns using the given method iscomputationally costly since we have to take the transpose 

# of the data matrix before we can remove duplicate features



#print(train_set_T.duplicated().sum())
corrmat = train_set.corr()

sns.heatmap(corrmat)

def get_corrdata(data,threshold):

    corr_col = set()

    corrmat = data.corr()

    for i in range(len(corrmat.columns)):

        for j in range(i):

            if abs(corrmat.iloc[i,j]) > threshold:

                colname = corrmat.columns[i]

                corr_col.add(colname)

    return corr_col       
# selecting feauters with high colinearity between them

corr_feat = get_corrdata(train_set,0.80)

corr_feat

#roc_auc = []

#for col in train_set.columns:

#    rfc = RandomForestClassifier(class_weight = "balanced")

#    rfc.fit(train_set[col].to_frame(),train_set_target)

#    y_pred = rfc.predict(train_set[col].to_frame())

#    roc_auc.append(roc_auc_score(y_pred,train_set_target))
#roc_values = pd.Series(roc_auc)

#roc_values.index= train_set.columns

#roc_values
# lets take a copy of the updated train and test data sets. Why?

# because I plan to build Decision Tree & Random Forest models, which do not need scaled data and I am going to scale the

# data in the next step to be used by other models

train_set_copy = train_set.copy()

test_set_copy = test_set.copy()
# scaling the numeric features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_set[num_vars] = scaler.fit_transform(train_set[num_vars]) # apply 

test_set[num_vars] = scaler.fit_transform(test_set[num_vars])

# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(train_set,train_set_target, train_size=0.7,test_size=0.3,random_state=100)
# lets create a dataframe to save model's performance

model_df = pd.DataFrame(columns = ["model_name","training roc_auc","test_roc_auc"])
# Confusion matrix function

def cm(pred,true):

    confusion = metrics.confusion_matrix(pred, true)

    print(confusion)

    score = round(metrics.roc_auc_score(pred,true),2)

    print("roc_auc score:",score)

    return(score)

# saving the list of current features

col = X_train.columns
# lets build a GLM and check the p-values of the features

# class_weight="balanced" will take care of the class imbalance in the dataset

X_train_sm = sm.add_constant(X_train)

logm1 = sm.GLM(y_train,X_train_sm, class_weight="balanced",family = sm.families.Binomial())

logm1.fit().summary()
col = col.drop('bin_3', 1)



X_train_sm = sm.add_constant(X_train[col])

# class_weight="balanced" will take care of the class imbalance in the dataset

logm14 = sm.GLM(y_train,X_train_sm, class_weight="balanced",family = sm.families.Binomial())

res = logm14.fit()

res.summary()
col = col.drop('nom_1_missing', 1)



X_train_sm = sm.add_constant(X_train[col])

# class_weight="balanced" will take care of the class imbalance in the dataset

logm14 = sm.GLM(y_train,X_train_sm, class_weight="balanced",family = sm.families.Binomial())

res = logm14.fit()

res.summary()
col = col.drop('nom_3_India', 1)



X_train_sm = sm.add_constant(X_train[col])

# class_weight="balanced" will take care of the class imbalance in the dataset

logm14 = sm.GLM(y_train,X_train_sm, class_weight="balanced",family = sm.families.Binomial())

res = logm14.fit()

res.summary()
# p-values looks good now, lets check VIF



vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# make predictions

y_train_pred = res.predict(X_train_sm)



X_test_sm = sm.add_constant(X_test[col])

y_test_pred = res.predict(X_test_sm)
X_train = X_train[col]



X_test = X_test[col]
logModel = LogisticRegression(class_weight = "balanced",solver = "saga")

res = logModel.fit(X_train,y_train)
# make predictions

y_train_pred_prob = res.predict_proba(X_train)

y_train_pred = res.predict(X_train)



y_test_pred_prob = res.predict_proba(X_test)

y_test_pred = res.predict(X_test)
print("training scores:")

train_score = cm(y_train_pred,y_train)



print("\ntest scores:\n")

test_score = cm(y_test_pred,y_test)



model_df.loc[0] = ["LogisticReg-Default",train_score,test_score]
# set up cross validation scheme

#folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)



# specify range of hyperparameters

#params = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge



## using Logistic regression for class imbalance, class_weight = balanced will take care of class imbalance in the dataset

#model = LogisticRegression(class_weight='balanced', solver = "saga")

#model_cv = GridSearchCV(estimator = model, param_grid = params, 

#                        scoring= 'roc_auc', 

#                        cv = folds, 

#                        return_train_score=True, verbose = 1)            

#model_cv.fit(X_train, y_train)
## reviewing the results

#cv_results = pd.DataFrame(model_cv.cv_results_)

#cv_results
## segerigating results for L1 and L2 regression and plotting them differently

#cv_results_penalty_l1 = cv_results.loc[cv_results['param_penalty']=='l1']

#cv_results_penalty_l2 = cv_results.loc[cv_results['param_penalty']=='l2']
## plotting results for Logistic regression with L1 panelty

#plt.figure(figsize=(8, 6))

#plt.plot(cv_results_penalty_l1['param_C'], cv_results_penalty_l1['mean_test_score'])

#plt.plot(cv_results_penalty_l1['param_C'], cv_results_penalty_l1['mean_train_score'])

#plt.xlabel('C')

#plt.ylabel('roc_auc')

#plt.legend(['test roc_auc', 'train roc_auc'], loc='upper right')

#plt.xscale('log')
## plotting results for Logistic regression with L2 panelty

#plt.figure(figsize=(8, 6))

#plt.plot(cv_results_penalty_l2['param_C'], cv_results_penalty_l2['mean_test_score'])

#plt.plot(cv_results_penalty_l2['param_C'], cv_results_penalty_l2['mean_train_score'])

#plt.xlabel('C')

#plt.ylabel('roc_auc')

#plt.legend(['test roc_auc', 'train roc_auc'], loc='upper right')

#plt.xscale('log')
## checking best score

#best_score = model_cv.best_score_

#best_param = model_cv.best_params_



#print(" The highest test roc_auc is {0} at {1}".format(best_score, best_param))
## preparing final model based on best score

model=LogisticRegression(C=0.1,penalty="l2",class_weight="balanced",solver="saga")

model.fit(X_train,y_train)
# make predictions

y_train_pred_prob = model.predict_proba(X_train)

y_train_pred = model.predict(X_train)



y_test_pred_prob = model.predict_proba(X_test)

y_test_pred = model.predict(X_test)
print("training scores:")

train_score = cm(y_train_pred,y_train)



print("\ntest scores:\n")

test_score = cm(y_test_pred,y_test)



model_df.loc[1] = ["LogisticReg-Regularize",train_score,test_score]
def auc(y_true, y_pred):

    def fallback_auc(y_true, y_pred):

        try:

            return metrics.roc_auc_score(y_true, y_pred)

        except:

            return 0.5

    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)
model = Sequential()

model.add(Dense(300, activation='relu',input_dim = X_train.shape[1]))

layers.Dropout(0.3)

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])

# reserving 20% of the training data for validation purpose in each epoch

model.fit(X_train, y_train,validation_split =0.2,batch_size = 128, epochs = 20)

y_train_pred = model.predict_classes(X_train)

y_test_pred = model.predict_classes(X_test)
train_score = cm(y_train_pred,y_train)

test_score = cm(y_test_pred,y_test)

model_df.loc[2] = ["ANN",train_score,test_score]
y_test_pred = model.predict_classes(test_set[col])
# lets prepare for the prediction submission

sub = pd.DataFrame()

sub['Id'] = test_set_id

sub['target'] = y_test_pred

sub.to_csv('submission_ann.csv',index=False)

# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(train_set_copy,train_set_target, train_size=0.7,test_size=0.3,random_state=100)
# Decision tree with default parameters

model = DecisionTreeClassifier(class_weight = "balanced")

model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)
train_score = cm(y_train_pred,y_train)

test_score = cm(y_test_pred,y_test)

model_df.loc[3] = ["Decision Tree",train_score,test_score]
# function to fine tune different hyperparameters

def dtree_hyper_param_tuning(X_train,y_train,param,values,score):

    print(param)

    # specify number of folds for k-fold CV

    n_folds = KFold(n_splits = 5, shuffle = True, random_state = 4)



    # parameters to build the model on

    parameters = {param: values}



# instantiate the model

    dtree = DecisionTreeClassifier(class_weight='balanced',criterion = "gini", 

                                   random_state = 101)



    # fit tree on training data

    tree = GridSearchCV(dtree, parameters, 

                        cv=n_folds, 

                       scoring="roc_auc",

                       return_train_score=True, verbose = 1)

    tree.fit(X_train, y_train)



    # scores of GridSearch CV

    scores = tree.cv_results_

    pd.DataFrame(scores).head()



    # plotting accuracies with max_depth

    plt.figure()

    plt.plot(scores[score], 

             scores["mean_train_score"], 

             label="training roc_auc")

    plt.plot(scores[score], 

             scores["mean_test_score"], 

             label="test roc_auc")

    plt.xlabel(param)

    plt.ylabel("auc_roc")

    plt.legend()

    plt.show()
# fine tune max depth

#dtree_hyper_param_tuning(X_train,y_train,"max_depth",range(10,50,10), "param_max_depth")
# fine tune minimum samples leaf

#dtree_hyper_param_tuning(X_train,y_train,"min_samples_leaf",[500,1000,2000],"param_min_samples_leaf")
# fine tune minimum samples split

#dtree_hyper_param_tuning(X_train,y_train,"min_samples_split",[500,1000,2000],"param_min_samples_split")
# putting all the tuned parameters together to find out the best fit

#param_grid = {

#    'max_depth': [10,15,20],

#    'min_samples_leaf': [500,700,900],

#    'min_samples_split': [200,400,600],

#    'criterion': ["entropy", "gini"]

#}



#n_folds = 5



# Instantiate the grid search model

#dtree = DecisionTreeClassifier(class_weight='balanced')

#grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 

#scoring="roc_auc", cv = n_folds,

#return_train_score=True, verbose = 1)



# Fit the grid search to the data

#grid_search.fit(X_train,y_train)
# cv results

#cv_results = pd.DataFrame(grid_search.cv_results_)

#cv_results.head()
# printing the optimal recall score and hyperparameters

#print("best roc_aus score", grid_search.best_score_)

#print(grid_search.best_estimator_)
# Decision tree with best parameters

model = DecisionTreeClassifier(max_depth = 20, min_samples_leaf = 500,min_samples_split = 200,class_weight = "balanced")

model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)
train_score = cm(y_train_pred,y_train)

test_score = cm(y_test_pred,y_test)

model_df.loc[4] = ["Decision Tree Tuned",train_score,test_score]
# snapshot of all the models created

model_df