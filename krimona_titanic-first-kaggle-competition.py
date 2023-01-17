# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_score, recall_score

from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import precision_recall_curve
df_train=pd.read_csv("/kaggle/input/titanic/train.csv")
df_train.head()
df_test=pd.read_csv("/kaggle/input/titanic/test.csv")
df_test.head()
df_train.shape
df_test.shape
df_train.info()
df_test.info()
df_train.describe()
df_test.describe()
df_train.isnull().sum()/len(df_train)*100
df_test.isnull().sum()/len(df_test)*100
df_train.drop(['Cabin'],1,inplace=True)
df_train.head()
df_test.drop(['Cabin'],1,inplace=True)
df_test.head()
df_train['Age'].fillna((df_train['Age'].mean()),inplace=True)
df_train['Age'].isnull().sum()
df_test['Age'].fillna((df_test['Age'].mean()),inplace=True)
df_test['Age'].isnull().sum()
df_train['Embarked'].fillna((df_train['Embarked'].mode()),inplace=True)
df_train['Embarked'].isnull().sum()
df_test['Embarked'].fillna((df_test['Embarked'].mode()),inplace=True)
df_test['Embarked'].isnull().sum()
sns.pairplot(df_train)
plt.show()
sns.barplot(x = 'Sex', y = 'Survived', data = df_train)
plt.show()
sns.boxplot(x=df_train['Parch'])
plt.show()
sns.boxplot(x=df_test['Parch'])
plt.show()
sns.boxplot(x=df_train['Fare'])
plt.show()
sns.boxplot(x=df_test['Fare'])
plt.show()
df_train.groupby(['Parch'])['Ticket'].value_counts().sort_values(ascending=False)
df_test.groupby(['Parch'])['Ticket'].value_counts().sort_values(ascending=False)
df_train.groupby(['Fare'])['Pclass'].value_counts()
df_test.groupby(['Fare'])['Pclass'].value_counts()
df_train.groupby(['SibSp'])['Name'].value_counts(ascending=False)
df_test.groupby(['SibSp'])['Name'].value_counts(ascending=False)
df_train['Survived']
### Checking survived
Survived= (sum(df_train['Survived'])/len(df_train['Survived'].index))*100
Survived
df_train['Pclass']=df_train['Pclass'].map({1:'cls_1', 2:'cls_2',3:'cls_3'})
dum_Pclass = pd.get_dummies(df_train['Pclass'],drop_first = True)
dum_Pclass.head()
df_test['Pclass']=df_test['Pclass'].map({1:'cls_1', 2:'cls_2',3:'cls_3'})
dum_Pclass_test = pd.get_dummies(df_test['Pclass'],drop_first = True)
dum_Pclass_test.head()
df_train['SibSp']=df_train['SibSp'].map({0:'SibSp_0',1:'SibSp_1', 2:'SibSp_2',3:'SibSp_3',4:'SibSp_4',5:'SibSp_5',8:'SibSp_8'})
dum_SibSp = pd.get_dummies(df_train['SibSp'],drop_first = True)
dum_SibSp.head()
df_test['SibSp']=df_test['SibSp'].map({0:'SibSp_0',1:'SibSp_1', 2:'SibSp_2',3:'SibSp_3',4:'SibSp_4',5:'SibSp_5',8:'SibSp_8'})
dum_SibSp_test = pd.get_dummies(df_test['SibSp'],drop_first = True)
dum_SibSp_test.head()
df_train['Parch']=df_train['Parch'].map({0:'Parch_0',1:'Parch_1', 2:'Parch_2',3:'Parch_3',4:'Parch_4',5:'Parch_5',6:'Parch_6'})
dum_Parch = pd.get_dummies(df_train['Parch'],drop_first = True)
dum_Parch.head()
df_train['Parch']=df_train['Parch'].map({0:'Parch_0',1:'Parch_1', 2:'Parch_2',3:'Parch_3',4:'Parch_4',5:'Parch_5',6:'Parch_6',9:'Parch_9'})
dum_Parch_test = pd.get_dummies(df_test['Parch'],drop_first = True)
dum_Parch_test.head()
dum_sex = pd.get_dummies(df_train['Sex'],drop_first = True)
dum_sex.head()
dum_sex_test = pd.get_dummies(df_test['Sex'],drop_first = True)
dum_sex_test.head()
df_train['Age']=pd.cut(df_train['Age'],bins=[0,10,20,40,60,80],labels=['child','teen','adults','mid-age_adults','old_adults'])
df_test['Age']=pd.cut(df_test['Age'],bins=[0,10,20,40,60,80],labels=['child','teen','adults','mid-age_adults','old_adults'])
dum_age = pd.get_dummies(df_train['Age'],drop_first = True)
dum_age.head()
dum_age_test = pd.get_dummies(df_test['Age'],drop_first = True)
dum_age_test.head()
dum_embarked = pd.get_dummies(df_train['Embarked'],drop_first = True)
dum_embarked.head()
dum_embarked_test = pd.get_dummies(df_test['Embarked'],drop_first = True)
dum_embarked_test.head()
df_train = pd.concat([df_train,dum_Pclass,dum_Parch,dum_sex,dum_SibSp,dum_age,dum_embarked], axis = 1)
df_train.head()
df_test = pd.concat([df_test,dum_Pclass_test,dum_Parch_test,dum_sex_test,dum_SibSp_test,dum_age_test,dum_embarked_test], axis = 1)
df_test.head()
df_train= df_train.drop(['Pclass','Parch','Age','Sex','SibSp','Ticket','Embarked'], 1)
df_test= df_test.drop(['Pclass','Parch','Age','Sex','SibSp','Ticket','Embarked'], 1)
X_train = df_train.drop(['Survived','PassengerId','Name'],1)

X_train.head()
y_train=df_train.pop('Survived')
scaler = StandardScaler()

X_train[['Fare']] = scaler.fit_transform(X_train[['Fare']])

# checking the correlation 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(df_train.corr(),annot = True)
plt.show()
# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()
log_reg=LogisticRegression()

rfe = RFE(log_reg, 15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train_rfe=X_train[col]
X_train.columns[~rfe.support_]
X_train_sm = sm.add_constant(X_train_rfe)
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
# Dropping highly correlated variables 
X_train_new=X_train_rfe.drop(['Parch_6'],axis=1)
X_train_sm = sm.add_constant(X_train_new)
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res1 = logm3.fit()
res1.summary()
# Dropping highly correlated variables 
X_train_new1=X_train_new.drop(['SibSp_8'],axis=1)
X_train_sm = sm.add_constant(X_train_new1)
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res2 = logm4.fit()
res2.summary()
# Dropping highly correlated variables 
X_train_new2=X_train_new1.drop(['SibSp_5'],axis=1)
X_train_sm = sm.add_constant(X_train_new2)
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res3 = logm5.fit()
res3.summary()
# Dropping highly correlated variables 
X_train_new3=X_train_new2.drop(['Parch_4'],axis=1)
X_train_sm = sm.add_constant(X_train_new3)
logm6 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res4 = logm6.fit()
res4.summary()
# Dropping highly correlated variables 
X_train_new4=X_train_new3.drop(['Parch_5'],axis=1)
X_train_sm = sm.add_constant(X_train_new4)
logm7 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res5 = logm7.fit()
res5.summary()
vif = pd.DataFrame()
vif['Features'] = X_train_new4.columns
vif['VIF'] = [variance_inflation_factor(X_train_new4.values, i) for i in range(X_train_new4.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
y_train_pred = res5.predict(X_train_sm)
y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Survived':y_train.values, 'Surv_Prob':y_train_pred})
y_train_pred_final['PassengerID'] = y_train.index
y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Surv_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )
print(confusion)
# Predicted     not_survived   survived
# Actual
# not_survived        471      78
# survived            90       252  
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted))
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
# Calculate false postive rate - predicting survived when passenger does not have survived
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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Survived, y_train_pred_final.Surv_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Survived, y_train_pred_final.Surv_Prob)
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Surv_Prob.map(lambda x: 1 if x > i else 0)
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
    cm1 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Surv_Prob.map( lambda x: 1 if x > 0.42 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.final_predicted )
print(confusion2)
# Predicted     not_survived   survived
# Actual
# not_survived        439      110
# survived            72       270  
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
#Looking at the confusion matrix again
confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )
confusion[1,1]/(confusion[0,1]+confusion[1,1])
confusion[1,1]/(confusion[1,0]+confusion[1,1])
precision_score(y_train_pred_final.Survived, y_train_pred_final.predicted)
recall_score(y_train_pred_final.Survived, y_train_pred_final.predicted)
y_train_pred_final.Survived, y_train_pred_final.predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Survived, y_train_pred_final.predicted)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()
