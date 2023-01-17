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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import plotly.express as px
Train_set=pd.read_csv("/kaggle/input/titanic/train.csv",encoding = "ISO-8859-1")
Train_set.head()
Train_set.info()
Train_set.describe(include='all')
missing=Train_set.isnull()
missing.head()
for column in missing.columns.values.tolist():
    print(column)
    print(missing[column].value_counts())
    print("")
objects=('Age','Cabin','Embarked')
y_pos=np.arange(len(objects))
miss=[117,687,2]
plt.barh(y_pos, miss, align='center', color='g',alpha=0.7)
plt.yticks(y_pos, objects)
plt.xlabel('missing data')
plt.title('columns')
Train_set.head()
#number of male and females
sns.countplot(x='Sex',data=Train_set)
plt.title("Number of male and female")
#number of male and female who servived
sns.countplot(x='Sex',hue='Survived',data=Train_set)
plt.title("Number of male and female who servived and who not ")
Train_set.head()
sns.countplot(x='Pclass',data=Train_set)
plt.title("Number of people in each ticket class ")
sns.countplot(x='Pclass',hue='Survived',data=Train_set)
plt.title("Serviers of each class ")
#dist of age
sns.distplot(Train_set['Age'],bins=25,kde=False)
plt.title("Age distribution ",pad=5)
plt.figure(figsize=(10,5))
fig = px.box(data_frame=Train_set,x='Pclass',y='Age')
print("Age group of each ticket class ")
fig.show()
def missing_age_fill(cols):
    age=cols[0]
    ticket_class=cols[1]
    
    if pd.isnull(age):
        if ticket_class==1:
            return 37
        elif ticket_class==2:
            return 29
        else:
            return 24
    else:
        return age
#apply on age
Train_set['Age'] =  Train_set[['Age',"Pclass"]].apply(missing_age_fill,axis=1)
Train_set.isnull().sum()
lst=['Sex']
embarked_err_remo = Train_set.groupby(lst).count()
embarked_err_remo.drop(['Survived','Name','Age','SibSp','Parch','Ticket','Fare','Cabin'],axis=1,inplace=True)
embarked_err_remo
sns.countplot(x='Embarked',hue='Sex',data=Train_set)
def missing_embarked_fill(cols):
    embarked=cols[0]
    
    if pd.isnull(embarked):
        return 'S'
    else:
        return embarked
Train_set['Embarked'] =  Train_set[['Embarked']].apply(missing_embarked_fill,axis=1)
#again check for missing data
Train_set.isnull().sum()
Train_set.drop(['Cabin'],axis=1,inplace=True)
Train_set.isnull().sum()
Train_set.head()
sns.countplot(x='Fare',hue='Survived',data=Train_set)
Train_set.drop(['PassengerId','Name','Ticket','Fare'],axis=1,inplace=True)
Train_set.head()
#some ages are in float so convert them to there floor value
Train_set['Age']=Train_set['Age'].apply(np.floor)
Train_set.iloc[78,:]
dummy1 = pd.get_dummies(Train_set['Sex'],drop_first=True)
Train_set = pd.concat([Train_set,dummy1],axis=1)
Train_set.head()
dummy2 = pd.get_dummies(Train_set['Embarked'],drop_first=True)
Train_set = pd.concat([Train_set,dummy2],axis=1)
Train_set.head()
#drop the columns sex and embarked
Train_set.drop(['Sex','Embarked'],axis=1,inplace=True)
Train_set.head()
sns.heatmap(Train_set.corr(),annot=True)
y=Train_set.pop('Survived')
x=Train_set
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x[['Pclass','Age','SibSp','Parch']]=scaler.fit_transform(x[['Pclass','Age','SibSp','Parch']])
x.head()
y.head()
from sklearn.linear_model import LinearRegression
log_reg = LinearRegression()
from sklearn.feature_selection import RFE
rfe = RFE(log_reg,5)
rfe=rfe.fit(x,y)
rfe.support_
list(zip(x.columns, rfe.support_, rfe.ranking_))
col=x.columns[rfe.support_]
import statsmodels.api as sm
x_1=sm.add_constant(x[col])    
log1 = sm.GLM(y,x_1,family=sm.families.Binomial())
res = log1.fit()
res.summary()
x_1.drop(['Parch'],axis=1,inplace=True)
x_2=sm.add_constant(x_1)

log2= sm.GLM(y,x_2,family=sm.families.Binomial())
res2 = log2.fit()
res2.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['features'] = x_2.columns
vif['VIF'] = [variance_inflation_factor(x_2.values,i) for i in range(x_2.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF" , ascending=False)
vif
x_3=sm.add_constant(x_2)
log3= sm.GLM(y,x_3,family=sm.families.Binomial())
res3 = log3.fit()
res3.summary()
y_pred = res3.predict(x_3)
y_pred[:10]
y_pred = y_pred.values.reshape(-1)
y_pred[:10]
y_pred_final = pd.DataFrame({'Servived':y.values,'Servivved_prob':y_pred})
y_pred_final.head()
y_pred_final['predicted'] = y_pred_final.Servivved_prob.map(lambda x:1 if x>0.5 else 0)
y_pred_final.head()
from sklearn import metrics
confusion = metrics.confusion_matrix(y_pred_final.Servived,y_pred_final.predicted)
print(confusion)
print(metrics.accuracy_score(y_pred_final.Servived,y_pred_final.predicted))
TP = confusion[1,1] # true positive
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
sensitivity = (TP / float(TP+FN))
sensitivity
specificity = (TN / float(TN+FP))
specificity
# Calculate false postive rate -
print(FP/ float(TN+FP))
# positive predictive value
print (TP / float(TP+FP))
fpr,tpr, threshold = metrics.roc_curve(y_pred_final.Servived,y_pred_final.Servivved_prob)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_pred_final[i]= y_pred_final.Servivved_prob.map(lambda x: 1 if x > i else 0)
y_pred_final.head()

# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix
# TP = confusion[1,1] # true positive
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_pred_final.Servived, y_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)

# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show
y_pred_final['final_predicted'] = y_pred_final.Servivved_prob.map( lambda x: 1 if x > 0.6 else 0)
y_pred_final.head()
# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Servived, y_pred_final.final_predicted)

confusion2 = metrics.confusion_matrix(y_pred_final.Servived, y_pred_final.final_predicted )
confusion2
TP = confusion2[1,1] # true positive
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives

# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
# Calculate false postive rate 
print(FP/ float(TN+FP))
# Positive predictive value
print (TP / float(TP+FP))
# Negative predictive value
print (TN / float(TN+ FN))

confusion = metrics.confusion_matrix(y_pred_final.Servived, y_pred_final.predicted )
confusion

#precision
confusion[1,1]/(confusion[0,1]+confusion[1,1])
#recall
confusion[1,1]/(confusion[1,0]+confusion[1,1])
from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_pred_final.Servived, y_pred_final.Servivved_prob)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()
test = pd.read_csv('/kaggle/input/titanic/test.csv')
survivors = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
test.describe(include='all')
test.info()
survivors.head()
survivors.info()
#combine the both dataset
Test_set = pd.merge(test,survivors,how='inner',on='PassengerId')
Test_set.head()
Test_set.info()
test_miss=Test_set.isnull()
test_miss
for column in test_miss.columns.values.tolist():
    print(column)
    print(test_miss[column].value_counts())
    print("")
objects=('Age','Cabin','Fare')
y_pos=np.arange(len(objects))
miss=[86,91,1]
plt.barh(y_pos, miss, align='center', color='g',alpha=0.7)
plt.yticks(y_pos, objects)
plt.xlabel('missing data')
plt.title('columns')
#filling the missing  age
Test_set['Age'] =  Test_set[['Age',"Pclass"]].apply(missing_age_fill,axis=1)
Test_set.isnull().sum()
Test_set.head()
Test_set.drop(['PassengerId','Name','Parch','Ticket','Fare','Cabin','Embarked'],axis=1,inplace=True)
Test_set.head()
#dummies
dummy3 = pd.get_dummies(Test_set['Sex'],drop_first=True)
Test_set = pd.concat([Test_set,dummy3],axis=1)
Test_set.head()
#drop Sex column
Test_set.drop(['Sex'],axis=1,inplace=True)
Test_set.head()
y_test=Test_set.pop('Survived')
x_test=Test_set
#minmaxscalling
x_test[['Pclass','Age']]=scaler.fit_transform(x_test[['Pclass','Age']])
x_test.head()
x_test_sm = sm.add_constant(x_test)
y_test_pred = res3.predict(x_test_sm)
y_test_pred[:10]
 # Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head
y_pred_1.head()
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final
 # Renaming the column
y_pred_final= y_pred_final.rename(columns={ 0 : 'Servived_Prob'})
y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Servived_Prob.map(lambda x: 1 if x > 0.6 else 0)
y_pred_final.head()
# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Survived, y_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_pred_final.Survived, y_pred_final.final_predicted )
confusion2
TP = confusion2[1,1] # true positive
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives

test_sensitivity = TP / float(TP+FN)
test_sensitivity
test_specificity = TN / float(TN+FP)
test_specificity
