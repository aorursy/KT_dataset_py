# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.cm as cm
# visulaisation

from matplotlib.pyplot import xticks

%matplotlib inline
# Data display coustomization

pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', -1)
class color:

   PURPLE = '\033[95m'

   CYAN = '\033[96m'

   DARKCYAN = '\033[36m'

   BLUE = '\033[94m'

   GREEN = '\033[92m'

   YELLOW = '\033[93m'

   RED = '\033[91m'

   BOLD = '\033[1m'

   UNDERLINE = '\033[4m'

   END = '\033[0m'

# import all libraries and dependencies for machine learning

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

from math import isnan
#To perform CART-RF-ANN

from sklearn.utils import resample

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn import tree

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import roc_curve,roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

#Data Loading

ins=pd.read_csv('../input/insurance-part2-datacsv/insurance_part2_data (1).csv')

ins.head()
print(color.BOLD + color.DARKCYAN+'This datasets contains information on tour insurance  and their claiming status respectively.')
ins.info()

print("\n")

print("Check for null values:")

print(ins.isnull().sum())

print("\n")

print(color.BOLD + color.DARKCYAN+'Dataset has',ins.shape[0],'rows and',ins.shape[1],'columns.')

print(color.BOLD + color.DARKCYAN+'Dataset has 2 columns are of integer type,2 of float data type and 6 are of object data type.')

print(color.BOLD + color.DARKCYAN+'Dataset has no null values.')
print(color.BOLD + color.DARKCYAN+"Number of duplicates in dataset :",ins.duplicated(['Age','Agency_Code','Type','Claimed','Commision','Channel','Duration','Sales','Product Name','Destination']).sum())



ins[ins.duplicated(subset=None)]
ins.drop_duplicates(keep=False,inplace=True)

print(color.BOLD + color.DARKCYAN+'Duplicates are removed')

print("Number of duplicates in dataset :",ins.duplicated(['Age','Agency_Code','Type','Claimed','Commision','Channel','Duration','Sales','Product Name','Destination']).sum())
print(color.BOLD+"Unique values per column in dataset :\n")

for column in ins[['Age', 'Agency_Code', 'Type', 'Claimed', 'Commision', 'Channel', 

                   'Duration', 'Sales', 'Product Name', 'Destination']]:

    print(color.BOLD + color.DARKCYAN,column.upper(),': ',ins[column].nunique())

    print(color.BOLD + color.DARKCYAN,ins[column].value_counts().sort_values())

    print('\n')
ins.Duration = ins.Duration.replace(to_replace = -1, value =0)
for column in ins:

    if ins[column].dtype == 'object':

        ins[column] = pd.Categorical(ins[column]).codes 
print(color.BOLD+color.DARKCYAN+'Post performing EDA dataset statistics are as below :')

ins.describe(include="all")
#historgrams

plt.tight_layout()

ins.hist(figsize=(20,30))

plt.show()
#checking correlations

plt.figure(figsize = (8, 8))

sns.heatmap(ins.corr(), annot = True)

plt.show()
plt.figure(figsize=(15,15))

ins[['Age','Agency_Code','Type','Claimed','Commision','Channel','Duration','Sales','Product Name','Destination']].boxplot(vert=0)
Q1=ins['Age'].quantile(0.25)

Q3=ins['Age'].quantile(0.75)

IQR=Q3-Q1

lr = Q1-1.5*IQR

ur = Q3+1.5*IQR

ins['Age']=np.where(ins['Age']>ur,ur,ins['Age'])

ins['Age']=np.where(ins['Age']<lr,lr,ins['Age'])

   
Q1=ins['Commision'].quantile(0.25)

Q3=ins['Commision'].quantile(0.75)

IQR=Q3-Q1

lr = Q1-1.5*IQR

ur = Q3+1.5*IQR

ins['Commision']=np.where(ins['Commision']>ur,ur,ins['Commision'])

ins['Commision']=np.where(ins['Commision']<lr,lr,ins['Commision'])
Q1=ins['Duration'].quantile(0.25)

Q3=ins['Duration'].quantile(0.75)

IQR=Q3-Q1

lr = Q1-1.5*IQR

ur = Q3+1.5*IQR

ins['Duration']=np.where(ins['Duration']>ur,ur,ins['Duration'])

ins['Duration']=np.where(ins['Duration']<lr,lr,ins['Duration'])
Q1=ins['Sales'].quantile(0.25)

Q3=ins['Sales'].quantile(0.75)

IQR=Q3-Q1

lr = Q1-1.5*IQR

ur = Q3+1.5*IQR

ins['Sales']=np.where(ins['Sales']>ur,ur,ins['Sales'])

ins['Sales']=np.where(ins['Sales']<lr,lr,ins['Sales'])
plt.figure(figsize=(15,15))

ins[['Age','Agency_Code','Type','Claimed','Commision','Channel','Duration','Sales','Product Name','Destination']].boxplot(vert=0)
print(color.BOLD+color.DARKCYAN,ins.Claimed.value_counts())

print('%1s',7721/(7721+3827))

print('%0s',3827/(7721+3827))
X = ins.drop("Claimed", axis=1)



y = ins.pop("Claimed")


X_train, X_test, train_labels, test_labels = train_test_split(X, y, test_size=.30, random_state=42)

print(color.BOLD+color.DARKCYAN+'X_train',X_train.shape)

print(color.BOLD+color.DARKCYAN+'X_test',X_test.shape)

print(color.BOLD+color.DARKCYAN+'train_labels',train_labels.shape)

print(color.BOLD+color.DARKCYAN+'test_labels',test_labels.shape)
param_grid = {

    

    #Parameters used initially :

        #'max_depth': [3,5,8,9,10,15,20],

        # 'max_features': [4,6],

        #'min_samples_leaf': [5,10,20,30,50],

        #'min_samples_split': [20,30,50,100],

    

   #Parameters finalized after obtaining best parameters:



    'criterion': ['gini'],

    'max_depth': [5,7,10],

    'max_features': [4,6],

    'min_samples_leaf': [3,5,10],

    'min_samples_split':[50,100,300]

    }



dt_model = DecisionTreeClassifier()



grid_search = GridSearchCV(estimator = dt_model, param_grid = param_grid, cv = 10)
print(color.BOLD+color.DARKCYAN+'Parameter used to build Cart model:')

grid_search.fit(X_train, train_labels)
print(color.BOLD+color.DARKCYAN+'Parameter finzilzed and applied to build to Cart model:')

print(grid_search.best_params_)

best_grid_dt = grid_search.best_estimator_

best_grid_dt
## Generating Tree

train_char_label = ['No', 'Yes']

tree_regularized = open('tree_regularized.dot','w')

dot_data = tree.export_graphviz(best_grid_dt, out_file= tree_regularized , feature_names = list(X_train), class_names = list(train_char_label))



tree_regularized.close()

dot_data
print(color.BOLD+'Variable Importance:\n')

print (color.BOLD+color.DARKCYAN,pd.DataFrame(best_grid_dt.feature_importances_,columns = ["Imp"], index = X_train.columns).sort_values('Imp',ascending=False))

ytrain_predict = best_grid_dt.predict(X_train)

ytest_predict = best_grid_dt.predict(X_test)
print(color.BOLD+color.DARKCYAN+'Getting the Predicted Classes and Probs:')

ytest_predict

ytest_predict_prob=best_grid_dt.predict_proba(X_test)

ytest_predict_prob

pd.DataFrame(ytest_predict_prob).head()
# predict probabilities

probs = best_grid_dt.predict_proba(X_train)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# calculate AUC

cart_train_auc = roc_auc_score(train_labels, probs)

print(color.BOLD+color.DARKCYAN+'AUC: %.3f' % cart_train_auc)

# calculate roc curve

cart_train_fpr, cart_train_tpr, cart_train_thresholds = roc_curve(train_labels, probs)

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(cart_train_fpr, cart_train_tpr)
print(color.BOLD+color.DARKCYAN+'Confusion matrix on the train data:')

confusion_matrix(train_labels,ytrain_predict) #comparing actuals & predicted

sns.heatmap(confusion_matrix(train_labels,ytrain_predict),annot=True, fmt='d',cbar=False, cmap='rainbow')

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

plt.title('Confusion Matrix')

plt.show()
print(color.BOLD+'Classification Report on the train data:\n')

print(color.BOLD+color.DARKCYAN+classification_report(train_labels,ytrain_predict))
cart_metrics=classification_report(train_labels, ytrain_predict,output_dict=True)

df=pd.DataFrame(cart_metrics).transpose()

cart_train_f1=round(df.loc["1"][2],2)

cart_train_recall=round(df.loc["1"][1],2)

cart_train_precision=round(df.loc["1"][0],2)

print(color.BOLD+'CART Train data metrices:\n')

print (color.BOLD+color.DARKCYAN+'CART Train Precision ',cart_train_precision)

print (color.BOLD+color.DARKCYAN+'CART Train Recall ',cart_train_recall)

print (color.BOLD+color.DARKCYAN+'CART Train F1 score ',cart_train_f1)
# predict probabilities

probs = best_grid_dt.predict_proba(X_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# calculate AUC

cart_test_auc = roc_auc_score(test_labels, probs)

print(color.BOLD+color.DARKCYAN+'AUC: %.3f' % cart_test_auc)

# calculate roc curve

cart_test_fpr, cart_test_tpr, cart_testthresholds = roc_curve(test_labels, probs)

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(cart_test_fpr, cart_test_tpr)
print(color.BOLD+color.DARKCYAN+'Confusion matrix on the test data:')

confusion_matrix(test_labels,ytest_predict) #comparing actuals & predicted

sns.heatmap(confusion_matrix(test_labels,ytest_predict),annot=True, fmt='d',cbar=False, cmap='rainbow')

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

plt.title('Confusion Matrix')

plt.show()
print(color.BOLD+'Classification Report on the test data:\n')

print(color.BOLD+color.DARKCYAN+classification_report(test_labels,ytest_predict))
cart_metrics=classification_report(test_labels, ytest_predict,output_dict=True)

df=pd.DataFrame(cart_metrics).transpose()

cart_test_precision=round(df.loc["1"][0],2)

cart_test_recall=round(df.loc["1"][1],2)

cart_test_f1=round(df.loc["1"][2],2)

print(color.BOLD+'CART Test data metrices:\n')

print (color.BOLD+color.DARKCYAN+'CART Test Precision',cart_test_precision)

print (color.BOLD+color.DARKCYAN+'CART Test Recall ',cart_test_recall)

print (color.BOLD+color.DARKCYAN+'CART Test F1 score ',cart_test_f1)
#Train Data Accuracy

cart_train_acc=best_grid_dt.score(X_train,train_labels) 

#Test Data Accuracy

cart_test_acc=best_grid_dt.score(X_test,test_labels)

param_grid = {

    

#Parameters used initially :



    #'max_depth': [5,10,15,20],

    #'max_features': [4,6,8,10],

    #'min_samples_leaf': [3,5,10],

    #'min_samples_split': [20,50,100],

    #'n_estimators': [100,200]



#Parameters finalized after obtaining best parameters:



'max_depth': [5,10,20],

    'max_features': [4,6],

    'min_samples_leaf': [4,6],

    'min_samples_split': [20,30,50],

    'n_estimators': [100]

}

rfcl = RandomForestClassifier(random_state=42)



grid_search_rfcl = GridSearchCV(estimator = rfcl, param_grid = param_grid, cv = 10)

print(color.BOLD+color.DARKCYAN+'Parameter used to build Cart model:')

grid_search_rfcl.fit(X_train, train_labels)
print(color.BOLD+color.DARKCYAN+'Parameter finzilzed and applied to build to Cart model:')



print(grid_search_rfcl.best_params_)

grid_search_rfcl=grid_search_rfcl.best_estimator_

grid_search_rfcl
print(color.BOLD+'Variable Importance:\n')

print (color.BOLD+color.DARKCYAN,pd.DataFrame(best_grid_dt.feature_importances_,columns = ["Imp"], index = X_train.columns).sort_values('Imp',ascending=False))

rf_ytrain_predict = grid_search_rfcl.predict(X_train)

rf_ytest_predict = grid_search_rfcl.predict(X_test)
print(color.BOLD+color.DARKCYAN+'Getting the Predicted Classes and Probs:')



ytest_predict

ytest_predict_prob=grid_search_rfcl.predict_proba(X_test)

ytest_predict_prob

pd.DataFrame(ytest_predict_prob).head()
rf_fpr, rf_tpr,_=roc_curve(train_labels,grid_search_rfcl.predict_proba(X_train)[:,1])

plt.figure(figsize=(12,7))

plt.plot(rf_fpr,rf_tpr, marker='x', label='Random Forest')

plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.show()

rf_train_auc=roc_auc_score(train_labels,grid_search_rfcl.predict_proba(X_train)[:,1])

print(color.BOLD+color.DARKCYAN+'Area under Curve is', rf_train_auc)
print(color.BOLD+color.DARKCYAN+'Confusion matrix on the train data:')

confusion_matrix(train_labels,rf_ytrain_predict) #comparing actuals & predicted

sns.heatmap(confusion_matrix(train_labels,rf_ytrain_predict),annot=True, fmt='d',cbar=False, cmap='rainbow')

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

plt.title('Confusion Matrix')

plt.show()
print(color.BOLD+'Classification Report on the train data:\n')

print(color.BOLD+color.DARKCYAN+classification_report(train_labels,rf_ytrain_predict))
rf_metrics=classification_report(train_labels, rf_ytrain_predict,output_dict=True)



df=pd.DataFrame(rf_metrics).transpose()

rf_train_precision=round(df.loc["1"][0],2)

rf_train_recall=round(df.loc["1"][1],2)

rf_train_f1=round(df.loc["1"][2],2)

print(color.BOLD+'Random Forest Train data metrices:\n')

print (color.BOLD+color.DARKCYAN+'RF Train Precision',rf_train_precision)

print (color.BOLD+color.DARKCYAN+'RF Train Recall',rf_train_recall)

print (color.BOLD+color.DARKCYAN+'RF Train F1 score ',rf_train_f1)
rft_fpr, rft_tpr,_=roc_curve(test_labels,grid_search_rfcl.predict_proba(X_test)[:,1])

plt.figure(figsize=(12,7))

plt.plot(rft_fpr,rft_tpr, marker='x', label='Random Forest')

plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.show()

rf_test_auc= roc_auc_score(test_labels,grid_search_rfcl.predict_proba(X_test)[:,1])

print(color.BOLD+color.DARKCYAN+'Area under Curve is',rf_test_auc)
print(color.BOLD+color.DARKCYAN+'Confusion matrix on the test data:')

confusion_matrix(test_labels,rf_ytest_predict) #comparing actuals & predicted

sns.heatmap(confusion_matrix(test_labels,rf_ytest_predict),annot=True, fmt='d',cbar=False, cmap='rainbow')

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

plt.title('Confusion Matrix')



plt.show()
print(color.BOLD+'Classification Report on the test data:\n')

print(color.BOLD+color.DARKCYAN+classification_report(test_labels,rf_ytest_predict))
rf_metrics=classification_report(test_labels, rf_ytest_predict,output_dict=True)

df=pd.DataFrame(rf_metrics).transpose()

rf_test_precision=round(df.loc["1"][0],2)

rf_test_recall=round(df.loc["1"][1],2)

rf_test_f1=round(df.loc["1"][2],2)

print(color.BOLD+'Random Forest Test data metrices:\n')

print (color.BOLD+color.DARKCYAN+'RF Test Precision ',rf_test_precision)

print (color.BOLD+color.DARKCYAN+'RF Test Recall ',rf_test_recall)

print (color.BOLD+color.DARKCYAN+'RF Test F1 score ',rf_test_f1)
rf_test_acc=grid_search_rfcl.score(X_test,test_labels)
#Test Data Accuracy

rf_train_acc=grid_search_rfcl.score(X_test,test_labels)

param_grid = {

  

    #Parameters used initially :

    

    #'hidden_layer_sizes' :[100,500,1000,1500],

    #'max_iter' :[200,500,1000],

    #'tol':[0.0001,0.1,0.01],

    

   #Parameters finalized after obtaining best parameters:

    

    'hidden_layer_sizes' :[500,1500],

    'max_iter' :[500,1500],

    'tol':[0.0001],

    

 }



clf = MLPClassifier(random_state=42)



grid_searchnn = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 10)
print(color.BOLD+color.DARKCYAN+'Parameter used to build ANN model:')

# Fit the model on the training data

grid_searchnn.fit(X_train, train_labels)
print(color.BOLD+color.DARKCYAN+'Parameter finzilzed and applied to build to ANN model:')

print(grid_searchnn.best_params_)

grid_searchnn.best_estimator_ 

# use the model to predict the training data

NN_train_pred = grid_searchnn.predict(X_train)

NN_test_pred = grid_searchnn.predict(X_test)
print(color.BOLD+color.DARKCYAN+'Getting the Predicted Classes and Probs:')

NN_test_pred

NN_test_pred_prob=best_grid_dt.predict_proba(X_test)

pd.DataFrame(NN_test_pred_prob).head()
from sklearn.metrics import roc_curve,roc_auc_score

nn_fpr,nn_tpr,_=roc_curve(train_labels,grid_searchnn.predict_proba(X_train)[:,1])

plt.figure(figsize=(12,7))

plt.plot(nn_fpr,nn_tpr, marker='x', label='ANN')

plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.show()

nn_train_auc=roc_auc_score(train_labels,grid_searchnn.predict_proba(X_train)[:,1])

print(color.BOLD+color.DARKCYAN+'Area under Curve is',nn_train_auc )
print(color.BOLD+color.DARKCYAN+'Confusion matrix on the train data:')

confusion_matrix(train_labels,NN_train_pred) #comparing actuals & predicted

sns.heatmap(confusion_matrix(train_labels,NN_train_pred),annot=True, fmt='d',cbar=False, cmap='rainbow')

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

plt.title('Confusion Matrix')

plt.show()
nn_train_acc=grid_searchnn.score(X_train,train_labels)
print(color.BOLD+'Classification Report on the train data:\n')

print(color.BOLD+color.DARKCYAN+classification_report(train_labels,NN_train_pred))
nn_metrics=classification_report(train_labels, NN_train_pred,output_dict=True)

df=pd.DataFrame(nn_metrics).transpose()

nn_train_precision=round(df.loc["1"][0],2)

nn_train_recall=round(df.loc["1"][1],2)

nn_train_f1=round(df.loc["1"][2],2)

print(color.BOLD+'ANN Train data metrices:\n')

print (color.BOLD+color.DARKCYAN+'ANN Train Data Precision ',nn_train_precision)

print (color.BOLD+color.DARKCYAN+'ANN Train Data Recall  ',nn_train_recall)

print (color.BOLD+color.DARKCYAN+'ANN Train Data F1 score  ',nn_train_f1)
from sklearn.metrics import roc_curve,roc_auc_score

nnt_fpr, nnt_tpr,_=roc_curve(test_labels,grid_searchnn.predict_proba(X_test)[:,1])

plt.figure(figsize=(12,7))

plt.plot(nnt_fpr,nnt_tpr, marker='x', label='ANN')

plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.show()

nn_test_auc=roc_auc_score(train_labels,grid_searchnn.predict_proba(X_train)[:,1])

print(color.BOLD+color.DARKCYAN+'Area under Curve is',nn_test_auc )
print(color.BOLD+color.DARKCYAN+'Confusion matrix on the test data:')

confusion_matrix(test_labels,NN_test_pred) #comparing actuals & predicted

sns.heatmap(confusion_matrix(test_labels,NN_test_pred),annot=True, fmt='d',cbar=False, cmap='rainbow')

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

plt.title('Confusion Matrix')

plt.show()
nn_test_acc=grid_searchnn.score(X_test,test_labels)
print(color.BOLD+'Classification Report on the test data:\n')

print(color.BOLD+color.DARKCYAN+classification_report(test_labels,NN_test_pred))
nn_metrics=classification_report(test_labels, NN_test_pred,output_dict=True)

df=pd.DataFrame(nn_metrics).transpose()

nn_test_precision=round(df.loc["1"][0],2)

nn_test_recall=round(df.loc["1"][1],2)

nn_test_f1=round(df.loc["1"][2],2)

print(color.BOLD+'ANN Test data metrices:\n')

print (color.BOLD+color.DARKCYAN+'ANN Test Data Precision ',nn_test_precision)

print (color.BOLD+color.DARKCYAN+'ANN Test Data Recall ',nn_test_recall)

print (color.BOLD+color.DARKCYAN+'ANN Test Data F1 score ',nn_test_f1)


index=['Accuracy', 'AUC', 'Recall','Precision','F1 Score']

data = pd.DataFrame({'CART Train':[cart_train_acc,cart_train_auc,cart_train_recall,cart_train_precision,cart_train_f1],

        'CART Test':[cart_test_acc,cart_test_auc,cart_test_recall,cart_test_precision,cart_test_f1],

       'Random Forest Train':[rf_train_acc,rf_train_auc,rf_train_recall,rf_train_precision,rf_train_f1],

        'Random Forest Test':[rf_test_acc,rf_test_auc,rf_test_recall,rf_test_precision,rf_test_f1],

       'Neural Network Train':[nn_train_acc,nn_train_auc,nn_train_recall,nn_train_precision,nn_train_f1],

        'Neural Network Test':[nn_test_acc,nn_test_auc,nn_test_recall,nn_test_precision,nn_test_f1]},index=index)

round(data,2)
plt.plot([0, 1], [0, 1], linestyle='--')

plt.plot(cart_train_fpr, cart_train_tpr,color='red',label="CART")

plt.plot(rf_fpr,rf_tpr,color='green',label="RF")

plt.plot(nn_fpr,nn_tpr,color='black',label="NN")

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--')

plt.plot(cart_test_fpr, cart_test_tpr,color='red',label="CART")

plt.plot(rft_fpr,rft_tpr,color='green',label="RF")

plt.plot(nnt_fpr,nnt_tpr,color='black',label="NN")

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower right')