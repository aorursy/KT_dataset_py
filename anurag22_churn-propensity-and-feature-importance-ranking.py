import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head()
data.shape
data.info()                                                  ## data type of each column, missing values, shape of table..
data.TotalCharges=pd.to_numeric(data.TotalCharges,errors='coerce')
data.describe(include=[np.object])
col_names=list(data.columns)
col_names.remove('customerID')
col_names.remove('tenure')
col_names.remove('MonthlyCharges')
col_names.remove('TotalCharges')
col_names
for i in col_names:
    j=data[i].value_counts()
    print('-----------------------------------')
    print(j)
for m in col_names:
    data[m].hist()
    plt.show()
data.describe(include=[np.number])
data.info()                                     ## Check the Missing Value
data.isnull().sum()                               ## Check the number missing value
## Calculate the median of the column

q=data.TotalCharges.quantile([0.1,0.5,0.9])
type(q)                                                                                 ## one Dimensional labelled Array
q
TC_median=q[.5]
TC_median
#data.loc[null_value].index             ## Indexes of the Missing Values
column_names=list(data.columns)
column_names
column_names[18:20]
plt.scatter(data.MonthlyCharges,data.TotalCharges, alpha=0.1)
plt.xlabel(column_names[18])
plt.ylabel(column_names[19])
plt.scatter(data.tenure,data.TotalCharges, alpha=0.01)
plt.xlabel(column_names[5])
plt.ylabel(column_names[19])
data.TotalCharges =  data.TotalCharges.fillna(TC_median)           
data.info()
data.boxplot(column=['MonthlyCharges','tenure'])
data.boxplot(column='TotalCharges')
sns.kdeplot(data.MonthlyCharges)
print(data[['MonthlyCharges','TotalCharges','tenure']].corr())
print(data.corr())
data_copy=data
data_copy=data_copy.drop(columns=['customerID', 'TotalCharges'])
data_dummy=pd.get_dummies(data_copy,drop_first=True)
len(data_dummy.columns)
data_dummy.head()
X=data_dummy.iloc[:,0:29]
y=data_dummy.iloc[:,29]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
y_pred1 = logreg.predict(X_train)
print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(logreg.score(X_train, y_train)))
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
results.mean()
results.std()
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
logit_roc_auc
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
print('recall score = ',recall_score(y_test,y_pred))
print('precision score = ',precision_score(y_test,y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_train,y_pred1))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.model_selection import GridSearchCV
# Create logistic regression instance
logistic = LogisticRegression()
# Regularization penalty space
penalty = ['l1', 'l2']

# Regularization hyperparameter space
C = np.logspace(0, 4, 10)
# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)
# Create grid search using 5-fold cross validation
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
# Fit grid search
best_model = clf.fit(X_train, y_train)
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
y_pred_GCV = best_model.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(best_model.score(X_test, y_test)))
y_pred_GCV = best_model.predict(X_train)
print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(best_model.score(X_train, y_train)))
from sklearn.ensemble import RandomForestClassifier
# Create random forest classifer object that uses entropy
rfc = RandomForestClassifier(criterion='entropy', random_state=0, n_jobs=-1,n_estimators=200,max_depth=11)

# Train model
rfc_model = rfc.fit(X_train, y_train)
              
# Predict    
y_pred_rfc=rfc_model.predict(X_test)
print('Accuracy of random forest classifier on test set: {:.2f}'.format(rfc_model.score(X_test, y_test)))
print(classification_report(y_test,y_pred_rfc))
# Create a series with feature importance 

rfc_model.feature_importances_
rfc_imp=list(rfc_model.feature_importances_)
rfc_colname=list(X.columns)
rfc_dict={'Column_Names_rfc':rfc_colname,'feature_imp_rfc':rfc_imp}
rfc_feature_imp=pd.DataFrame(rfc_dict)
rfc_feature_rank=rfc_feature_imp.sort_values(by='feature_imp_rfc',ascending = False)
rfc_feature_rank
from sklearn.feature_selection import RFE
model_rfe=LogisticRegression()
rfe=RFE(model_rfe,1)
rfe_fit=rfe.fit(X_train,y_train)
rfe_fit.n_features_
rfe_fit.ranking_
rank=list(rfe_fit.ranking_)
X.columns
col_nm=list(X.columns)
dict_rank={'Column_Name': col_nm,'Ranking':rank}
df_rank=pd.DataFrame(dict_rank)
df_rank.sort_values('Ranking')
y_pred_list=list(y_pred)
y_prob=logreg.predict_proba(X_test)
y_prob_list=list(y_prob)
pd.DataFrame(y_prob_list,columns=['No_Churn','Churn']).sort_values(by='Churn', ascending=False).head(20)