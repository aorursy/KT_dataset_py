import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

raw_data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")

raw_data.head()
raw_data.describe(include='all')
#mapping senior citizen 1 to yes and 0 to no for visulization purposes and check info to identify data types

raw_data['SeniorCitizen']=raw_data['SeniorCitizen'].map({1:'Yes', 0:'No'})

raw_data.info()
#Seems No null values but some elements in TotalCharges columns are spaces (''), those do not not showup as nulls 

raw_data.isnull().sum()
# change '' to NaN

raw_data_with_nan  = raw_data.replace(' ', np.nan)

# now we have 11 null elements

raw_data_with_nan.isnull().sum()
#mising values (null) percentage is small(0.15%). We can drop all those rows

print ("Missing Values Percentage: {}%".format(11/raw_data_with_nan.shape[0]*100))

data_no_mv = raw_data_with_nan.dropna(axis=0)

#change data type of TotalCharges, object to float for further analysis 

data_no_mv['TotalCharges'] = data_no_mv['TotalCharges'].astype(float)



#customer id does not effect on analysis. It will drop from the dataset

data_no_mv_no_id = data_no_mv.drop(['customerID'],axis=1)
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

#collect all columns with datatype 'Object'

object_cols = list(data_no_mv_no_id.select_dtypes(include=['object']).columns)

#remove "Churn" column

object_cols.remove('Churn')

fig, axes = plt.subplots(4, 4, figsize=(20, 20), sharex=True)

i=0

for colname in object_cols:

  i=i+1

  ax1 = fig.add_subplot(4,4,i)

  sns.countplot(x='Churn', hue=colname, data=data_no_mv_no_id)

plt.show()
plt.figure(figsize=(20,5))

tenure_count = data_no_mv_no_id['tenure'].value_counts()

sns.barplot(tenure_count.index, tenure_count.values, alpha=0.9)

plt.title('Frequency distribution of Tenures', fontsize='17')

plt.xlabel('Number of months', fontsize='15')

plt.ylabel('Number of occurences',fontsize='15')

plt.show()
churn_count = data_no_mv_no_id['Churn'].value_counts()

plt.figure(figsize=(4,2))

plt.bar(churn_count.index,churn_count.values)

plt.xlabel('Churn')

plt.ylabel('Number of occurences')

plt.title('frequency of target values')

plt.show()
sns.distplot(data_no_mv_no_id['MonthlyCharges'])

plt.show()

sns.distplot(data_no_mv_no_id['TotalCharges'])

plt.show()
# change categorical variables to numerical variables (one-hot). Drop first column for each 

# category to avoid extra correlinearity.

data_pre_processed = pd.get_dummies(data_no_mv_no_id,drop_first=True)

#separate input and targets

inputs = data_pre_processed.drop('Churn_Yes', axis=1)

targets = data_pre_processed['Churn_Yes']
# Import the scaling module to scale data

from sklearn.preprocessing import StandardScaler

# Create a scaler object

scaler = StandardScaler()

# Fit the inputs (calculate the mean and standard deviation feature-wise)

scaler.fit(inputs)

# scale input data

inputs_scaled = scaler.transform(inputs)

# Import the module for the split

from sklearn.model_selection import train_test_split

# Split the variables with an 80-20 split and some random state 

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=1)
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()

#fit data to logistic model

log_model.fit(x_train,y_train)

# get prediction on train data itself to measure the performance of the model 

y_hat = log_model.predict(x_train)

#import confusion matrix

from sklearn.metrics import confusion_matrix

#create confusion matrix on train data

print('confusion matrix for training data = ', confusion_matrix(y_hat,y_train))

#import accurracy score 

from sklearn.metrics import accuracy_score

#compute accuray score of model on training data

print('acuracy score for training data = ', accuracy_score(y_hat,y_train))
#prediction on test data 

predictions = log_model.predict(x_test)

print('confusion matrix for test data = ', confusion_matrix(predictions,y_test))

logistic_acc = accuracy_score(predictions,y_test)

print('acuracy score for training data = ',logistic_acc)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

leaf_nodes_list = range(2,400)

score_list = []

for n_nodes in leaf_nodes_list:

  rf_clf = RandomForestClassifier( max_leaf_nodes=n_nodes)

  scores = cross_val_score(rf_clf, inputs, targets, scoring = "neg_mean_squared_error", cv=10)

  score_list.append(-scores.mean())

print('minimum validation error = ',min(score_list))

rfc_leaf_node = leaf_nodes_list[score_list.index(min(score_list))]

print('minimum error reaches for the number of leaf_nodes = ', rfc_leaf_node)

plt.plot(leaf_nodes_list,score_list)

plt.xlabel('Number of leaf nodes',fontsize=15)

plt.ylabel('Validation Score',fontsize=15)

plt.title('Random Forest Model',fontsize=15)

plt.show()
from sklearn.ensemble import RandomForestClassifier

#pre-test shows minmum between 50 and 200

leaf_nodes_list = range(50,200)

score_list = []

std_list=[]

for n_nodes in leaf_nodes_list:

    rf_clf = RandomForestClassifier( max_leaf_nodes=n_nodes)

    scores = cross_val_score(rf_clf, inputs, targets, scoring = "neg_mean_squared_error", cv=10)

    score_list.append(-scores.mean())

    std_list.append(scores.std())

print('minimum validation error = ',min(score_list))

rfc_min_score_leaf_node = leaf_nodes_list[score_list.index(min(score_list))]

print('minimum error reaches for the number of leaf_nodes = ', rfc_min_score_leaf_node )

print('minimum validation std = ',min(std_list))

rfc_min_std_leaf_node=leaf_nodes_list[std_list.index(min(std_list))]

print('minimum std reaches for the number of leaf_nodes = ', rfc_min_std_leaf_node)

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=False)

fig.tight_layout()

ax1 = fig.add_subplot(1,2,1)

plt.plot(leaf_nodes_list,score_list)

plt.xlabel('Number of leaf nodes',fontsize=15)

plt.title('Mean Validation Score',fontsize=15)

ax1 = fig.add_subplot(1,2,2)

plt.plot(leaf_nodes_list,std_list)

plt.xlabel('Number of leaf nodes',fontsize=15)

plt.title('STD of Validation Score',fontsize=15)

plt.show()
Tree_x_train, Tree_x_test, Tree_y_train, Tree_y_test = train_test_split(inputs, targets, test_size=0.3, random_state=15)

rf_clf = RandomForestClassifier(random_state=30,max_leaf_nodes=rfc_min_score_leaf_node)

#fit data to random forest model

rf_clf.fit(Tree_x_train,Tree_y_train)

#make predictions of train data itself

rf_clf_hat = rf_clf.predict(Tree_x_test)

print('Prediction with the leaf node corresponding to the minimum validation score')

#accuracy score

display(accuracy_score(rf_clf_hat,Tree_y_test))

#confusion matrix

print(confusion_matrix(rf_clf_hat,Tree_y_test))
Tree_x_train, Tree_x_test, Tree_y_train, Tree_y_test = train_test_split(inputs, targets, test_size=0.3, random_state=15)

rf_clf = RandomForestClassifier(random_state=30,max_leaf_nodes=rfc_min_std_leaf_node)

#fit data to random forest model

rf_clf.fit(Tree_x_train,Tree_y_train)

#make predictions of train data itself

rf_clf_hat = rf_clf.predict(Tree_x_test)

print('Prediction with the leaf node corresponding to the minimum standard diviation of validation score')

#accuracy score

display(accuracy_score(rf_clf_hat,Tree_y_test))

#confusion matrix

print(confusion_matrix(rf_clf_hat,Tree_y_test))
from sklearn.tree import DecisionTreeClassifier

leaf_nodes_list = range(2,100)

score_list = []

for n_nodes in leaf_nodes_list:

  dt_clf = DecisionTreeClassifier( max_leaf_nodes=n_nodes)

  scores = cross_val_score(dt_clf, inputs, targets, scoring = "neg_mean_squared_error", cv=10)

  score_list.append(-scores.mean())

print('minimum validation error = ',min(score_list))

dtc_leaf_node = leaf_nodes_list[score_list.index(min(score_list))]

print('minimum error reaches for the number of leaf_nodes = ', dtc_leaf_node)

plt.plot(leaf_nodes_list,score_list)

plt.xlabel('Number of leaf nodes',fontsize=15)

plt.ylabel('Validation Score',fontsize=15)

plt.title('Decission Tree Classifier')

plt.show()
dt_clf = DecisionTreeClassifier(random_state = 1, max_leaf_nodes=dtc_leaf_node)

#fit data to the decision tree model

dt_clf.fit(Tree_x_train,Tree_y_train)

#make a prediction

dt_clf_hat= dt_clf.predict(Tree_x_test)

print('predictions using the number of leaf nodes corresponding to the minimum validation score ')

#accuray score

print('accuary score = ' ,accuracy_score(Tree_y_test,dt_clf_hat))

#confusion matrix

print('confusion matrix = ',confusion_matrix(dt_clf_hat,Tree_y_test))
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

leaf_nodes_list = range(2,100)

score_list = []

for n_nodes in leaf_nodes_list:

  dec_tree_model = DecisionTreeRegressor( max_leaf_nodes=n_nodes)

  scores = cross_val_score(dec_tree_model, inputs, targets, scoring = "neg_mean_squared_error", cv=10)

  score_list.append(-scores.mean())

print('minimum validation error = ',min(score_list))

dtr_leaf_node = leaf_nodes_list[score_list.index(min(score_list))]

print('minimum error reaches for the number of leaf_nodes = ', dtr_leaf_node)

plt.plot(leaf_nodes_list,score_list)

plt.xlabel('Number of leaf nodes',fontsize=15)

plt.ylabel('Validation Score',fontsize=15)

plt.title('Decission Tree Regressor')

plt.show()
dt_reg = DecisionTreeRegressor(random_state = 1, max_leaf_nodes=dtr_leaf_node)

#fit data to the decision tree model

dt_reg.fit(Tree_x_train,Tree_y_train)

#make a prediction

dt_reg_hat= dt_reg.predict(Tree_x_test).round()

print('predictions using the number of leaf nodes corresponding to the minimum validation score ')

#accuray score

print('accuracy score = ', accuracy_score(Tree_y_test,dt_reg_hat.round()))

#confusion matrix

print('confusion matrix = ' ,confusion_matrix(dt_reg_hat.round(),Tree_y_test))
feature_cols = list(inputs.columns)

from sklearn.tree import export_graphviz

from sklearn import tree

from graphviz import Source

from IPython.display import SVG,display



graph = Source(tree.export_graphviz(dt_reg,out_file=None,

                                        rounded=True,proportion = False,

                                        feature_names = feature_cols, 

                                        precision  = 2,

                                        class_names=["Not churn","Churn"],

                                        filled = True                         

                                       )

                  )



display(graph)
from sklearn.ensemble import RandomForestRegressor

leaf_nodes_list = range(2,200)

score_list = []

for n_nodes in leaf_nodes_list:

  rf_reg = RandomForestRegressor( max_leaf_nodes=n_nodes)

  scores = cross_val_score(rf_reg, inputs, targets, scoring = "neg_mean_squared_error", cv=10)

  score_list.append(-scores.mean())

print('minimum validation error = ',min(score_list))

rfr_leaf_node = leaf_nodes_list[score_list.index(min(score_list))]

print('minimum error reaches for the number of leaf_nodes = ', rfr_leaf_node)

plt.plot(leaf_nodes_list,score_list)

plt.xlabel('Number of leaf nodes',fontsize=15)

plt.ylabel('Validation Score',fontsize=15)

plt.title('Random Forest Regressor',fontsize=15)

plt.show()
rf_reg = RandomForestRegressor(random_state=1,max_leaf_nodes=rfr_leaf_node)

#fit data to random forest model

rf_reg.fit(Tree_x_train,Tree_y_train)

#make predictions 

rf_reg_hat = rf_reg.predict(Tree_x_test)

print('predictions using the number of leaf nodes corresponding to the minimum validation score ')

#accuracy score

print('accuracy socre = ',accuracy_score(rf_reg_hat.round(),Tree_y_test))

#confusion matrix

print('confusion matrix = ',confusion_matrix(rf_reg_hat.round(),Tree_y_test))