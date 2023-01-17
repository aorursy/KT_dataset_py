# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd
df = pd.read_csv("../input/creditcard.csv")
df.columns


df['Time'].isnull().sum()
for i in df.columns:

    print(df[i].isnull().sum())
print('No frauds : ',round(df['Class'].value_counts()[0]/len(df)*100,2),'% of the dataset')

print('Frauds : ',round(df['Class'].value_counts()[1]/len(df)*100,2),'% of the dataset')
import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot('Class', data=df)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)')


from sklearn.preprocessing import StandardScaler, RobustScaler
std_scaler  = StandardScaler()

rbst_scaler = RobustScaler()



df['scalled_time'] = rbst_scaler.fit_transform(df['Time'].values.reshape(-1,1))



df['scalled_amount'] = rbst_scaler.fit_transform(df['Amount'].values.reshape(-1,1))

df.drop(['Time','Amount'],axis =1 , inplace = True)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



X = df.drop('Class', axis=1)

y = df['Class']
sss = StratifiedShuffleSplit(n_splits = 5 , test_size = 0.2 , random_state = 42 )
for train_index, test_index in sss.split(X, y):

    print("Train:", train_index, "Test:", test_index)

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
df = df.sample(frac = 1)
fraud_df = df.loc[df['Class']==1]

notfraud_df = df.loc[df['Class']==0][:492]
normal_distributed_df = pd.concat([fraud_df,notfraud_df])

new_df = normal_distributed_df.sample(frac = 1 , random_state = 42)
new_df.corr()
sub_sample_corr = new_df.corr()

sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)

ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)

plt.show()

from scipy.stats import norm

f ,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20, 6))

v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values

sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')

ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)



v12_fraud_dist = new_df['V12'].loc[new_df['Class'] ==1].values

sns.distplot(v12_fraud_dist,ax = ax2 , fit = norm, color = '#FB8861')

ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)



v10_fraud_dist = new_df['V10'].loc[new_df['Class'] ==1].values

sns.distplot(v10_fraud_dist , ax = ax3 , fit = norm,color = '#FB8861')

ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)

v14_fraud = new_df['V14'].values

v14_fraud
v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values

q25 , q75 = np.percentile(v14_fraud,25) , np.percentile(v14_fraud,75)

print('Quartile 25: {} || Quartile 75: {}'.format(q25, q75))

v14_iqr = q75 - q25

print('v14_IQR = {}'.format(v14_iqr))

v14_cut_off = v14_iqr * 1.5

v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off

print('Cut Off: {}'.format(v14_cut_off))

print('V14 Lower: {}'.format(v14_lower))

print('V14 Upper: {}'.format(v14_upper))

outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]

print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V14 outliers:{}'.format(outliers))
v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values

q25 , q75 = np.percentile(v12_fraud,25) , np.percentile(v12_fraud,75)

print('Quartile 25: {} || Quartile 75: {}'.format(q25, q75))

v12_iqr = q75 - q25

print('v12_IQR = {}'.format(v12_iqr))

v12_cut_off = v12_iqr * 1.5

v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off

print('Cut Off: {}'.format(v12_cut_off))

print('V12 Lower: {}'.format(v12_lower))

print('V12 Upper: {}'.format(v12_upper))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]

print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V12 outliers:{}'.format(outliers))
v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values

q25 , q75 = np.percentile(v10_fraud,25) , np.percentile(v10_fraud,75)

print('Quartile 25: {} || Quartile 75: {}'.format(q25, q75))

v10_iqr = q75 - q25

print('v10_IQR = {}'.format(v10_iqr))

v10_cut_off = v10_iqr * 1.5

v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off

print('Cut Off: {}'.format(v10_cut_off))

print('V10 Lower: {}'.format(v10_lower))

print('V10 Upper: {}'.format(v10_upper))

outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]

print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V10 outliers:{}'.format(outliers))
X = new_df.drop('Class', axis = 1 )

y = new_df['Class']
from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)
X_train

from sklearn.linear_model import LogisticRegression 

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from sklearn.model_selection import cross_val_score
classifiers = { "LogisticRegression " : LogisticRegression() ,

                "KNearst " : KNeighborsClassifier() ,

                "Support Vector Classifier " : SVC() ,

                "Decision Tree Classifier " : DecisionTreeClassifier()

    

}
for key , classifier in classifiers.items() :

    

    classifier.fit(X_train , y_train)

    training_score = cross_val_score(classifier,X_train , y_train , cv = 5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of",round(training_score.mean(),2)

          *100,"% accuracy score")
from sklearn.model_selection import GridSearchCV
log_reg_param = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid_log_reg = GridSearchCV(LogisticRegression(),log_reg_param)

grid_log_reg.fit(X_train,y_train)

log_reg = grid_log_reg.best_estimator_

log_reg
knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knear_algo = GridSearchCV(KNeighborsClassifier(),knears_params)

grid_knear_algo.fit(X_train,y_train)

knear_algo  = grid_knear_algo.best_estimator_
knear_algo
# Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(X_train, y_train)



# SVC best estimator

svc = grid_svc.best_estimator_



# DecisionTree Classifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(X_train, y_train)



# tree best estimator

tree_clf = grid_tree.best_estimator_
tree_clf


log_reg_score = cross_val_score(log_reg, X_train, y_train, cv = 5)

print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')



knear_algo_score = cross_val_score(knear_algo,X_train,y_train, cv = 5)

print('KNeighbour Algorithm Score: ', round(knear_algo_score.mean() * 100, 2).astype(str) + '%')



svc_score = cross_val_score(svc,X_train , y_train , cv = 5 )

print('SVM Algorithm Score: ', round(svc_score.mean() * 100, 2).astype(str) + '%')



tree_clf_score =  cross_val_score( tree_clf,X_train , y_train , cv = 5 )

print('Tree classifier Score: ', round(tree_clf_score.mean() * 100, 2).astype(str) + '%')



from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_predict
log_predict  = cross_val_predict(log_reg, X_train , y_train , cv = 5 ,  method = "decision_function")

knear_predict = cross_val_predict(knear_algo , X_train , y_train , cv = 5 )

svc_predict = cross_val_predict(svc ,  X_train , y_train , cv = 5 , method = "decision_function" )

tree_predict = cross_val_predict(tree_clf , X_train , y_train , cv = 5 )
from sklearn.metrics import roc_auc_score

print('Logestic regression test:', roc_auc_score( y_train ,log_predict ))

print('KNears Neighbors: ', roc_auc_score(y_train, knear_predict))

print('Support Vector Classifier: ', roc_auc_score(y_train, svc_predict))

print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_predict))
log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_predict)

knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knear_predict)

svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_predict)

tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_predict)

plt.figure(figsize=(16,8))

plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)

plt.plot(log_fpr,log_tpr,label='Logistic Regression Classifier Score:'.format(y_train,log_predict))

plt.plot(knear_fpr,knear_tpr,label='KNears Neighbors Classifier Score:'.format(y_train,knear_predict))

plt.plot(svc_fpr,svc_tpr,label='SVC Classifier Score:'.format(y_train,svc_predict))

plt.plot(tree_fpr,tree_tpr,label='Tree Classifier Score:'.format(y_train,tree_predict))



plt.plot([0, 1], [0, 1], 'k--')

plt.axis([-0.01, 1, 0, 1])

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),

                arrowprops=dict(facecolor='#6E726D', shrink=0.05),

                )

plt.legend()


from imblearn.over_sampling import SMOTE  

print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))

print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))
from sklearn.model_selection import train_test_split, RandomizedSearchCV

accuracy_lst = []

precision_lst = []

recall_lst = []

f1_lst = []

auc_lst = []
log_reg_sm = LogisticRegression()

rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_param, n_iter=4)

knears_neighbors = KNeighbours()
log_reg_sm.fit(X_test,y_test)

y_pred_log_reg = log_reg_sm.predict(X_test)

knears_neighbors = KNeighborsClassifier() 

knears_neighbors.fit(X_test,y_test)

y_pred_knear = knears_neighbors.predict(X_test)

# Other models fitted with UnderSampling



log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)

kneighbors_cf = confusion_matrix(y_test, y_pred_knear)



fig, ax = plt.subplots(2, 2,figsize=(22,12))

sns.heatmap(log_reg_cf, ax=ax[0][0], annot=True, cmap=plt.cm.copper)

ax[0, 0].set_title("Logistic Regression \n Confusion Matrix", fontsize=14)

ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)



sns.heatmap(kneighbors_cf, ax=ax[0][1], annot=True, cmap=plt.cm.copper)

ax[0][1].set_title("KNearsNeighbors \n Confusion Matrix", fontsize=14)

ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)

ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

log_reg_cf
kneighbors_cf