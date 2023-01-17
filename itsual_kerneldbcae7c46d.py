#!/usr/bin/env python
# coding: utf-8

# # POC
# 
# ## Steps:-
# 1. Read, understand and prepare the data
# 2. Exploratory Data Analysis
# 3. Outlier Analysis / Transformation / Treatment
# 4. Modelling (Logistic, SVM & Random forest)
# 5. Model accurancy & Metrics
# 5. Final analysis and conclusions

# In[60]:


#Loading Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import xticks
import matplotlib.ticker as mtick
import os

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

#ML Libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.datasets import make_classification


# In[61]:


#reading Dataset
emp = pd.read_excel("C:/Users/aliborious/Downloads/TakenMind-Python-Analytics-Problem-case-study-1-1.xlsx",'Existing employees')
ex =  pd.read_excel("C:/Users/aliborious/Downloads/TakenMind-Python-Analytics-Problem-case-study-1-1.xlsx",'Employees who have left')


# In[62]:


emp.head()


# In[63]:


ex.head()


# In[64]:


# Missing data check
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False).round(2)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print("Missing Data" '\n', missing_data(emp),'\n')
print("Shape" '\n', emp.shape,'\n')
print("Data Type" '\n', emp.dtypes)


# In[65]:


print("Missing Data" '\n', missing_data(ex),'\n')
print("Shape" '\n', ex.shape,'\n')
print("Data Type" '\n', ex.dtypes)


# ## Insights:
# 
# 1. Dataset "Existing employees" has 11428 rows and 10 columns with no missing values
# 2. Dataset "Employees who have left" has 3571 rows and 10 columns  with no missing values

# In[66]:


emp.columns


# In[67]:


ex.columns


# In[68]:


emp.describe(percentiles=[.25,.5,.75,.80,0.85,0.90,0.95]).round(2)


# In[69]:


ex.describe(percentiles=[.25,.5,.75,.80,0.85,0.90,0.95]).round(2)


# In[70]:


emp['dept'].value_counts()


# In[71]:


emp['salary'].value_counts()


# In[72]:


ex['dept'].value_counts()


# In[73]:


ex['salary'].value_counts()


# In[74]:


#Inserting new column status of employee
emp.insert(0, 'status', 1)
ex.insert(0, 'status', 0)


# In[75]:


emp.dtypes


# In[76]:


ex.dtypes


# In[77]:


data = pd.concat([emp,ex])


# In[78]:


data.shape


# In[79]:


print("Number of Unique values in each column:",'\n\n', data.nunique(dropna = True).sort_values(ascending = True))


# In[80]:


data.groupby('dept').mean().round(3)


# In[81]:


data.groupby('salary').mean().round(3)


# In[82]:


data.groupby(['dept', 'status']).size().unstack().plot(kind = 'bar', stacked = False,figsize = (8,6))


# In[83]:


data.groupby(['dept', 'status']).size().groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack().plot(kind='bar',stacked=True,figsize = (8,6))

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.title('Pecentage distribution of employee attrition - Department wise')
plt.show()


# In[84]:


data.groupby(['salary', 'status']).size().unstack().plot(kind = 'bar', stacked = False,figsize = (8,6))


# In[85]:


data.groupby(['salary', 'status']).size().groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack().plot(kind='bar',stacked=True,figsize = (8,6))

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.title('Pecentage distribution of employee attrition - Salary wise')
plt.show()


# In[86]:


data.groupby('status').mean()


# In[87]:


plt.figure(figsize = (20, 15))
sns.boxplot(x = 'salary', y = 'satisfaction_level', data = data, showfliers = False)
plt.xticks(rotation=45)


# In[88]:


plt.figure(figsize = (20, 15))
sns.boxplot(x = 'dept', y = 'satisfaction_level', data = data, showfliers = False)
plt.xticks(rotation=45)


# In[89]:


plt.figure(figsize = (20, 15))
sns.heatmap(data.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[90]:


# Creating a dummy variable for the variable 'dept' and dropping the first one.
dept = pd.get_dummies(data['dept'],prefix='dept',drop_first=True)
#Adding the results to the master dataframe
data = pd.concat([data,dept],axis=1)

# Creating a dummy variable for the variable 'salary' and dropping the first one.
salary = pd.get_dummies(data['salary'],prefix='salary',drop_first=True)
#Adding the results to the master dataframe
data = pd.concat([data,salary],axis=1)


# In[91]:


# We have created dummies for the below variables, so we can drop them
data_f = data.drop(['dept','salary'], 1)
data_f.head(3)


# In[92]:


data_f.info()


# In[93]:


data_f.describe(percentiles=[.25,.5,.75,.80,0.85,0.90,0.95]).round(2)


# In[94]:


#Checking the attrition Rate
att = 100 -(sum(data_f['status'])/len(data_f['status'].index))*100
att


# In[95]:


#Model Building
# Putting feature variable to X
X = data_f.drop(['status','Emp ID'],axis=1)

# Putting response variable to y
y = data_f['status']

y.head()


# In[96]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)


# In[97]:


# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[98]:


logreg = LogisticRegression()
rfe = RFE(logreg, 10)             # running RFE with 10 variables as output
rfe = rfe.fit(X,y)
print(rfe.support_)           # Printing the boolean results
print(rfe.ranking_)           # Printing the ranking)


# In[99]:


# Variables selected by RFE 
col = ['satisfaction_level','last_evaluation','time_spend_company','Work_accident',
       'promotion_last_5years','dept_RandD','dept_hr','dept_management','salary_low','salary_medium']


# In[100]:


logsk = LogisticRegression(C=1e9)
#logsk.fit(X_train[col], y_train)
logsk.fit(X_train, y_train)


# In[101]:


print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logsk.predict(X_test))))


# In[102]:


# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train[col], y_train)


# In[103]:


print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(y_test, rf.predict(X_test[col]))))


# In[104]:


# Predicted probabilities
y_pred = rf.predict_proba(X_test[col])
# Converting y_pred to a dataframe which is an array
y_pred_df = pd.DataFrame(y_pred)
# Converting to column dataframe
y_pred_1 = y_pred_df.iloc[:,[1]]
# Let's see the head
y_pred_1.head()


# In[105]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
y_test_df.head()


# In[106]:


# Putting CustID to index
y_test_df['Emp ID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df,y_pred_1],axis=1)
# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 1 : 'Attrition_Prob'})
# Rearranging the columns
y_pred_final = y_pred_final.reindex_axis(['Emp ID','status','Attrition_Prob'], axis=1)
# Let's see the head of y_pred_final
y_pred_final.head()


# In[107]:


# Exporting to excel
y_pred_final.to_excel("C:/Users/aliborious/Downloads/output.xlsx")


# In[108]:


# Support Vector Machine
svc = SVC()
svc.fit(X_train, y_train)


# In[109]:


print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(y_test, svc.predict(X_test))))


# In[110]:


#Cross Validation
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = RandomForestClassifier()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# In[111]:


#Random forest model accuracy is high and we will go with it
#Precision and recall

print(classification_report(y_test, rf.predict(X_test[col])))


# In[112]:


y_pred = rf.predict(X_test[col])
forest_conf = metrics.confusion_matrix(y_pred, y_test, [1,0])
sns.heatmap(forest_conf, annot=True, fmt='.2f',xticklabels = ["Present","Left"] , yticklabels = ["Present","Left"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Random Forest')
plt.savefig('random_forest')


# In[113]:


print(classification_report(y_test, logsk.predict(X_test)))


# In[114]:


logsk_y_pred = logsk.predict(X_test)
logsk_conf = metrics.confusion_matrix(logsk_y_pred, y_test, [1,0])
sns.heatmap(logsk_conf, annot=True, fmt='.2f',xticklabels = ["Present","Left"] , yticklabels = ["Present","Left"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Logistic Regression')
plt.savefig('logistic_regression')


# In[115]:


print(classification_report(y_test, svc.predict(X_test)))


# In[116]:


svc_y_pred = svc.predict(X_test)
svc_conf = metrics.confusion_matrix(svc_y_pred, y_test, [1,0])
sns.heatmap(svc_conf, annot=True, fmt='.2f',xticklabels = ["Present","Left"] , yticklabels = ["Present","Left"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Support Vector Machine')
plt.savefig('support_vector_machine')


# In[117]:


#ROC Curve
logit_roc_auc = roc_auc_score(y_test, logsk.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logsk.predict_proba(X_test)[:,1])
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test[col]))
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test[col])[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC')
plt.show()


# In[118]:


feature_labels = np.array(['satisfaction_level','last_evaluation','time_spend_company','Work_accident',
       'promotion_last_5years','dept_RandD','dept_hr','dept_management','salary_low','salary_medium'])
importance = rf.feature_importances_
feature_indexes_by_importance = importance.argsort()
for index in feature_indexes_by_importance:
    print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))


# In[ ]: