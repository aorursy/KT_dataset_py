import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
hr=pd.read_csv('../input/HR_Analytics.csv') # simply give the path where file is stored
hr.head()
type(hr)
# Renaming certain columns for better readability
hr = hr.rename(columns={'sales' : 'department'})
sns.heatmap(hr.isnull(),yticklabels=False,cbar=False,cmap='viridis')
hr['left'].value_counts()/len(hr['left'])
hr['number_project'].value_counts()
hr.info()
hr.describe()
corr=hr.corr()
sns.heatmap(corr,xticklabels=True,yticklabels=True,cmap='Oranges_r')
corr
summary_of_left=hr.groupby('left').mean()
summary_of_left
sns.pairplot(hr,hue='left')
#employee retention vs salary range
f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(x="salary", hue='left', data=hr).set_title('Employee Salary vs left Distribution')
#employee retention based on departments
flatui = ["#FF6037", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
f, ax = plt.subplots(figsize=(15, 10))
sns.countplot(x='department', hue='left', data=hr,palette=flatui).set_title('Retention across departments');
# Set up the matplotlib figure
f, axes = plt.subplots(ncols=3, figsize=(15, 6))

# Graph Employee Satisfaction
sns.distplot(hr.satisfaction_level, color="#3AA655", ax=axes[0]).set_title('Employee Satisfaction Distribution')
axes[0].set_ylabel('Employee Count')

# Graph Employee Evaluation
sns.distplot(hr.last_evaluation, color="#ED0A3F", ax=axes[1]).set_title('Employee Evaluation Distribution')
axes[1].set_ylabel('Employee Count')

# Graph Employee Average Monthly Hours
sns.distplot(hr.average_montly_hours, color="#F8D568", ax=axes[2]).set_title('Employee Average Monthly Hours Distribution')
axes[2].set_ylabel('Employee Count')
f, ax = plt.subplots(figsize=(15, 8))
sns.violinplot(x="number_project", y="satisfaction_level", hue="left", data=hr, split=True,
               inner="quart")
f, ax = plt.subplots(figsize=(15, 8))
sns.violinplot(x="time_spend_company", y="satisfaction_level", hue="left", data=hr, split=True,
               inner="quart")
# Kernel Density Plot
fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(hr.loc[(hr['left'] == 0),'last_evaluation'] , color='g',shade=True,label='Did not leave')
ax=sns.kdeplot(hr.loc[(hr['left'] == 1),'last_evaluation'] , color='b',shade=True, label='Left the company')
ax.set(xlabel='Employee Evaluation', ylabel='Frequency')
plt.title('Employee Evaluation Distribution - left vs not left')
# Kernel Density Plot
fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(hr.loc[(hr['left'] == 0),'satisfaction_level'] , color='g',shade=True,label='Did not leave')
ax=sns.kdeplot(hr.loc[(hr['left'] == 1),'satisfaction_level'] , color='r',shade=True, label='Left the company')
ax.set(xlabel='Employee Satisfaction', ylabel='Frequency')
plt.title('Employee Satisfaction Distribution - left vs not left')
# Kernel Density Plot
fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(hr.loc[(hr['left'] == 0),'average_montly_hours'] , color='g',shade=True,label='Did not leave')
ax=sns.kdeplot(hr.loc[(hr['left'] == 1),'average_montly_hours'] , color='r',shade=True, label='Left the company')
ax.set(xlabel='Monthly Hours', ylabel='Frequency')
plt.title('Employee Monthly hours Distribution - left vs not left')
f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(x="promotion_last_5years", hue='left', data=hr).set_title('Employee Salary vs left Distribution')
f, ax = plt.subplots(figsize=(15, 6))
sns.boxplot(x="number_project", y="last_evaluation", hue="left", data=hr,palette=flatui)
f, ax = plt.subplots(figsize=(15, 6))
sns.boxplot(x="time_spend_company", y="last_evaluation", hue="left", data=hr,palette=flatui)
# satisfaction and evaluation
sns.lmplot(x='satisfaction_level', y='last_evaluation', data=hr,fit_reg=False,hue='left')
#decision tree classifier to find most useful features

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12,6)

#converting into categorical variables
hr["department"] = hr["department"].astype('category').cat.codes
hr["salary"] = hr["salary"].astype('category').cat.codes


# Create train and test splits
X_train, X_test, y_train, y_test = train_test_split(hr.drop('left',axis=1), 
                                                    hr['left'], test_size=0.30, 
                                                    random_state=101)

dtree = tree.DecisionTreeClassifier(
    #max_depth=3,
    class_weight="balanced",
    min_weight_fraction_leaf=0.01
    )
dtree = dtree.fit(X_train,y_train)

## plot the importances ##
importances = dtree.feature_importances_
feat_names = hr.drop(['left'],axis=1).columns


indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature importances by DecisionTreeClassifier")
plt.bar(range(len(indices)), importances[indices], color='blue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()
#train test split
target_name = 'left'
X = hr.drop('left', axis=1)

y=hr[target_name]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=123, stratify=y)
#Feature Scaling
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_1 = scaler.transform(X_train)
X_test_1 = scaler.transform(X_test)
# Cross Validation function
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn import metrics
#Defining a Cross validation function
#n_folds = 10
def classification_cv(model):
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    scoring = 'accuracy'
    results = model_selection.cross_val_score(model, X_train_1, y_train, cv=kfold, scoring=scoring)
    return(print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std())))
#using logistic regression to predict employee risk of leaving
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression(C=0.4,class_weight = "balanced")
#Cross validating the model using holdout method
#Cross validation Holdout method for learning
Logistic_regression_cv=classification_cv(logis)
Logistic_regression_cv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve


logis.fit(X_train_1, y_train)
print ("\n\n ---Logistic Model---")
logit_roc_auc = roc_auc_score(y_test, logis.predict(X_test_1))
print ("Logistic AUC = %2.2f" % logit_roc_auc)
print(classification_report(y_test, logis.predict(X_test_1)))
print(confusion_matrix(y_test, logis.predict(X_test_1)))
print ("Logistic regression accuracy is %2.2f" % accuracy_score(y_test, logis.predict(X_test_1) ))
from sklearn.ensemble import RandomForestClassifier
# Random Forest Model
rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=6, 
    min_samples_split=10, 
    class_weight="balanced",
    random_state=100
    )
# cross validation
classification_cv(rf)
rf.fit(X_train_1, y_train)
print ("\n\n ---Random Forest  Model---")
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test_1))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test, rf.predict(X_test_1)))
print(confusion_matrix(y_test, rf.predict(X_test_1)))
print ("Random Forest is %2.2f" % accuracy_score(y_test, rf.predict(X_test_1) ))
## Gradient Boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier(n_estimators=400,learning_rate=0.1,random_state=100,max_features=4 )
#fitting the model
gbc.fit(X_train_1, y_train)
print ("\n\n ---GBC---")
gbc_roc_auc = roc_auc_score(y_test, gbc.predict(X_test_1))
print ("GBC AUC = %2.2f" % gbc_roc_auc)
print(classification_report(y_test, gbc.predict(X_test_1)))
print(confusion_matrix(y_test, gbc.predict(X_test_1)))
print ("GBC accuracy is %2.2f" % accuracy_score(y_test, gbc.predict(X_test_1) ))
# Create ROC Graph
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, logis.predict_proba(X_test_1)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test_1)[:,1])
gbc_fpr, gbc_tpr, gbc_thresholds = roc_curve(y_test, gbc.predict_proba(X_test_1)[:,1])



plt.figure(figsize=(15, 10))

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)

# Plot GBC ROC
plt.plot(gbc_fpr, gbc_tpr, label='Gradient Boosting(area = %0.2f)' % gbc_roc_auc)


# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()