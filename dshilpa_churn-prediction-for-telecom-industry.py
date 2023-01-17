#importing the required packages/libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler





import warnings # supress warnings

warnings.filterwarnings('ignore')
#importing the data set

Telecom_dataset=pd.read_csv("/kaggle/input/telecom_churn_case_study.csv")
Telecom_dataset.head()
print(Telecom_dataset.shape)

Telecom_dataset.describe()
Telecom_dataset.info()
#Checking for null values

Telecom_dataset.isnull().sum()
#Removing unwanted columns

Telecom_DS1 = Telecom_dataset.drop(['mobile_number','circle_id','loc_og_t2o_mou','std_og_t2o_mou','loc_ic_t2o_mou'],1)
#Checking the data types of all the columns

Telecom_DS1.columns.to_series().groupby(Telecom_DS1.dtypes).groups
#Removing the Date Time columns as not needed

Telecom_DS1 = Telecom_DS1.drop(['last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8',

        'last_date_of_month_9', 'date_of_last_rech_6', 'date_of_last_rech_7',

        'date_of_last_rech_8', 'date_of_last_rech_9',

        'date_of_last_rech_data_6', 'date_of_last_rech_data_7',

        'date_of_last_rech_data_8', 'date_of_last_rech_data_9'],1)
Telecom_DS1.columns.to_series().groupby(Telecom_DS1.dtypes).groups
null_columns=list(Telecom_DS1.columns[round(100*(Telecom_DS1.isnull().sum()/len(Telecom_DS1.index)), 2) > 40])

null_columns
Telecom_DS1.shape
#impiting null values with 0

Telecom_DS1.fillna(0,inplace=True)
null_columns=list(Telecom_DS1.columns[round(100*(Telecom_DS1.isnull().sum()/len(Telecom_DS1.index)), 2) > 40])

null_columns
Telecom_DS1['tot_data_rech_6']=Telecom_DS1['total_rech_data_6'] * Telecom_DS1['av_rech_amt_data_6']

Telecom_DS1['tot_data_rech_7']=Telecom_DS1['total_rech_data_7'] * Telecom_DS1['av_rech_amt_data_7']

Telecom_DS1['tot_data_rech_8']=Telecom_DS1['total_rech_data_8'] * Telecom_DS1['av_rech_amt_data_8']

Telecom_DS1['tot_data_rech_9']=Telecom_DS1['total_rech_data_9'] * Telecom_DS1['av_rech_amt_data_9']
Telecom_DS1['tot_rech_amt_6']=Telecom_DS1['tot_data_rech_6'] + Telecom_DS1['total_rech_amt_6']

Telecom_DS1['tot_rech_amt_7']=Telecom_DS1['tot_data_rech_7'] + Telecom_DS1['total_rech_amt_7']

Telecom_DS1['tot_rech_amt_8']=Telecom_DS1['tot_data_rech_8'] + Telecom_DS1['total_rech_amt_8']

Telecom_DS1['tot_rech_amt_9']=Telecom_DS1['tot_data_rech_9'] + Telecom_DS1['total_rech_amt_9']
Telecom_DS1.shape
#Determining the average revenue of all customers

Telecom_DS1['av_rech_amt']=((Telecom_DS1['tot_rech_amt_6'] + Telecom_DS1['tot_rech_amt_7'])/2)
#Determining the 70th percentile of average recharge amount

Telecom_DS1['av_rech_amt'].quantile(0.7)
#Filtering High Value Customers

Telecom_DS1=Telecom_DS1.loc[(Telecom_DS1['av_rech_amt'] > 478.0)]
Telecom_DS1.shape
def conditions(Telecom_DS1):

    if (Telecom_DS1['total_ic_mou_9'] <=0 ) & (Telecom_DS1['total_og_mou_9'] <= 0) & (Telecom_DS1['vol_2g_mb_9'] <= 0) & (Telecom_DS1['vol_3g_mb_9']<=0):

        return 1

    else:

        return 0

    

Telecom_DS1['Churn'] = Telecom_DS1.apply(conditions, axis=1)

Telecom_DS1['Churn'].value_counts()
# Removing attributes of Churn Phase to get final dataset for modelling

col_churn = [col for col in Telecom_DS1.columns if '_9' not in col]

Telecom_DS2 = Telecom_DS1[col_churn]

Telecom_DS2.shape
Telecom_DS2.info()
Telecom_DS2.describe()
X=Telecom_DS2.drop(['Churn'],1)

X.head()
y=Telecom_DS2['Churn']

y.head()
# Split the dataset into 70% train and 30% test



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
scaler = MinMaxScaler()

col_list=list(X_train.columns)

X_train[col_list] = scaler.fit_transform(X_train[col_list])



X_train.head()
corr=Telecom_DS2.corr()
plt.figure(figsize = (20,10))        # Size of the figure

sns.heatmap(Telecom_DS2.corr(),annot = True)

plt.show()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.75

to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

print(len(to_drop))

to_drop
#Dropping the Highly correlated columns

X_test = X_test.drop(X_test[to_drop], 1)

X_train = X_train.drop(X_train[to_drop], 1)
print(X_train.shape)

print(X_test.shape)
X_train.head()
from sklearn.decomposition import PCA 



pca = PCA(svd_solver='randomized', random_state=42)



X_train_pca = pca.fit_transform(X_train)  

  

pca.components_
pca.explained_variance_ratio_
#Plotting the scree plot

%matplotlib inline

fig = plt.figure(figsize = (16,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

#plt.xticks(range(min(valueX), max(valueX)+1))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.show()
from sklearn.decomposition import IncrementalPCA

pca_final = IncrementalPCA(n_components=15)
X_train_pca = pca_final.fit_transform(X_train)  

X_train.shape
X_train_pca.shape
X_train_pca
pca_final.components_
X_train_pca.shape
y_train.shape
import statsmodels.api as sm

X_train_sm=sm.add_constant(X_train_pca)

logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

logm1=logm1.fit()

logm1.summary()
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
X_train_pca_df=pd.DataFrame(X_train_pca[1:,1:],    # values

#             index=X_train_pca[1:,0],    # 1st column as index

#             columns=X_train_pca[0,1:]  # 1st row as the column names

                           ) 
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_pca_df.columns

vif['VIF'] = [variance_inflation_factor(X_train_pca_df.values, i) for i in range(X_train_pca_df.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Getting the predicted values on the train set

y_train_pred = logm1.predict(X_train_sm)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Churn':y_train.values, 'Churn_Prob':y_train_pred})

#y_train_pred_final['CustID'] = y_train.index

y_train_pred_final.tail(20)
y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
y_train_pred_final['Churn'].value_counts()
y_train_pred_final['predicted'].value_counts()
from sklearn import metrics

# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false postive rate - predicting churn when customer does not have churned

print(FP/ float(TN+FP))
# positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Churn, y_train_pred_final.Churn_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Churn_Prob.map( lambda x: 1 if x > 0.3 else 0)



y_train_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false postive rate - predicting churn when customer does not have churned

print(FP/ float(TN+FP))
# Positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
#Precision: TP / TP + FP

confusion[1,1]/(confusion[0,1]+confusion[1,1])
#Recall: TP / TP + FN

confusion[1,1]/(confusion[1,0]+confusion[1,1])
from sklearn.metrics import precision_recall_curve
y_train_pred_final.Churn, y_train_pred_final.predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
X_test_pca = pca_final.fit_transform(X_test)  
X_test_sm = sm.add_constant(X_test_pca)
y_test_pred = logm1.predict(X_test_sm)
y_test_pred_df = pd.DataFrame(y_test_pred)
y_test_df = pd.DataFrame(y_test)
# Removing index for both dataframes to append them side by side 

y_test_pred_df.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)
y_pred_final = pd.concat([y_test_df, y_test_pred_df],axis=1)
y_pred_final= y_pred_final.rename(columns={ 0 : 'Churn_Prob'})
y_pred_final.tail()
y_pred_final['final_predicted'] = y_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_pred_final.Churn, y_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_pred_final.Churn, y_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Importing decision tree classifier from sklearn library

from sklearn.tree import DecisionTreeClassifier



# Fitting the decision tree with default hyperparameters, apart from

# max_depth which is 5 so that we can plot and read the tree.

dt_default = DecisionTreeClassifier(max_depth=5)

dt_default.fit(X_train, y_train)
# Let's check the evaluation metrics of our default model



# Importing classification report and confusion matrix from sklearn metrics

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Making predictions

y_pred_default = dt_default.predict(X_test)



# Printing classification report

print(classification_report(y_test, y_pred_default))
# Printing confusion matrix and accuracy

print(confusion_matrix(y_test,y_pred_default))

print(accuracy_score(y_test,y_pred_default))
# Importing required packages for visualization

from IPython.display import Image  

from sklearn.externals.six import StringIO  

from sklearn.tree import export_graphviz

import pydotplus, graphviz



# Putting features

features = list(X_train.columns[1:])

features.append(list(y_train))
# plotting tree with max_depth=3

dot_data = StringIO()  

export_graphviz(dt_default, out_file=dot_data,

                feature_names=features, filled=True,rounded=True)



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_depth': range(1, 40)}



# instantiate the model

dtree = DecisionTreeClassifier(criterion = "gini", 

                               random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

tree.fit(X_train, y_train)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with max_depth

plt.figure()

#plt.plot(scores["param_max_depth"], 

#         scores["mean_train_score"], 

#         label="training accuracy")

plt.plot(scores["param_max_depth"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_depth")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# so we will take 5 folds
# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_leaf': range(5, 200, 20)}



# instantiate the model

dtree = DecisionTreeClassifier(criterion = "gini", 

                               random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

tree.fit(X_train, y_train)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with min_samples_leaf

plt.figure()

#plt.plot(scores["param_min_samples_leaf"], 

#         scores["mean_train_score"], 

#         label="training accuracy")

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_leaf")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# Create the parameter grid 

param_grid = {

    'max_depth': range(5, 15, 5),

    'min_samples_leaf': range(50, 150, 50),

    'min_samples_split': range(50, 150, 50),

    'criterion': ["entropy", "gini"]

}



n_folds = 5



# Instantiate the grid search model

dtree = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 

                          cv = n_folds, verbose = 1)



# Fit the grid search to the data

grid_search.fit(X_train,y_train)
# cv results

cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results
# printing the optimal accuracy score and hyperparameters

print("best accuracy", grid_search.best_score_)

print(grid_search.best_estimator_)
# tree with max_depth = 3

clf_gini = DecisionTreeClassifier(criterion = "gini", 

                                  random_state = 100,

                                  max_depth=3, 

                                  min_samples_leaf=50,

                                  min_samples_split=50)

clf_gini.fit(X_train, y_train)



# score

print(clf_gini.score(X_test,y_test))
# plotting tree with max_depth=3

dot_data = StringIO()  

export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
# classification metrics

from sklearn.metrics import classification_report,confusion_matrix

y_pred = clf_gini.predict(X_test)

print(classification_report(y_test, y_pred))
# confusion matrix

print(confusion_matrix(y_test,y_pred))