# import libraries



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

# Reading the train and test dataset



titanic_train=pd.read_csv("../input/train-and-test-set/train.csv",encoding='utf-8',engine='python')



titanic_test=pd.read_csv("../input/train-and-test-set/test.csv",encoding='utf-8',engine='python')
titanic_train.head()
titanic_test.head()
titanic_train.shape
titanic_test.shape
round(sum(titanic_train['Survived']/len(titanic_train['Survived'])),2)*100
# Since column 'Cabin' and 'Pclass' are related we can drop 'Cabin' as 'Pclass' is numerical and shows correlation with 'Cabin'
titanic_train=titanic_train.drop('Cabin',axis=1)

titanic_test=titanic_test.drop('Cabin',axis=1)
titanic_train.info()
# Data Transformation
titanic_train['Family_Size']=titanic_train['SibSp'] + titanic_train['Parch']



titanic_test['Family_Size']=titanic_test['SibSp'] + titanic_test['Parch']





# removing the extra columns 'SibSp' and 'Parch'



titanic_train=titanic_train.drop(['SibSp','Parch'],axis=1)

titanic_test=titanic_test.drop(['SibSp','Parch'],axis=1)
# Removing unnecessary columns like 'Name' and  'Ticket'



titanic_train=titanic_train.drop(['Name','Ticket'],axis=1)

titanic_test=titanic_test.drop(['Name','Ticket'],axis=1)
dummy_sex=pd.get_dummies(titanic_train['Sex'],prefix='Sex',drop_first=True)



titanic_train=pd.concat([titanic_train,dummy_sex],axis=1)
dummy_sex=pd.get_dummies(titanic_test['Sex'],prefix='Sex',drop_first=True)



titanic_test=pd.concat([titanic_test,dummy_sex],axis=1)
dummy_embark=pd.get_dummies(titanic_train['Embarked'],prefix='Embarked',drop_first=True)



titanic_train=pd.concat([titanic_train,dummy_embark],axis=1)
dummy_embark=pd.get_dummies(titanic_test['Embarked'],prefix='Embarked',drop_first=True)



titanic_test=pd.concat([titanic_test,dummy_embark],axis=1)
titanic_train=titanic_train.drop(['Sex','Embarked'],axis=1)



titanic_test=titanic_test.drop(['Sex','Embarked'],axis=1)
titanic_train.isnull().sum()
titanic_train.Age.describe()
# Computing missing values



# We will fill the missing values with median value as we have outlier in the 'Age' column

# and hence median is our best shot here
titanic_train=titanic_train.fillna(titanic_train.Age.median())
titanic_test.isnull().sum()
# we will remove the one row of null value in 'Fare' column and fill the 'Age' column with median of the 'Age' column
titanic_test.Fare.isnull().index
titanic_test.iloc[152,3]=titanic_test.Fare.median()
titanic_test=titanic_test.fillna(titanic_test.Age.median())
import sklearn.preprocessing

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore")



scaler=StandardScaler()
titanic_train.describe()
titanic_train.head()
# Removing PassengerId and Survived column from the train dataset



X=titanic_train.drop(['PassengerId','Survived'],axis=1)

y=titanic_train['Survived']
X.head()
X[['Pclass','Age','Fare','Family_Size']]=scaler.fit_transform(X[['Pclass','Age','Fare','Family_Size']])



X.head()
y=pd.DataFrame(y)



y.head()
# Let's look at the correlation of the train dataset



plt.figure(figsize=(12,8))

sns.heatmap(X.corr(),annot=True)

plt.show()
# Building the model



import statsmodels.api as sm
logm1=sm.GLM(y,(sm.add_constant(X)),family=sm.families.Binomial())

logm1.fit().summary()
# using RFE



from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE



logreg=LogisticRegression()

rfe=RFE(logreg,5)

rfe=rfe.fit(X,y)
col=X.columns[rfe.support_]
col
X_new=sm.add_constant(X[col])
logm2=sm.GLM(y,X_new,family=sm.families.Binomial())

res=logm2.fit()

res.summary()
# Predicting the model



y_train_pred=res.predict(X_new)



y_train_pred=y_train_pred.values.reshape(-1)

y_train_final=pd.DataFrame({'Survived': y.values.reshape(-1),'Survive_probablity':y_train_pred})





y_train_final['Passenger_id']=titanic_train['PassengerId']





y_train_final['Predicted']=y_train_final['Survive_probablity'].map(lambda x: 1 if x>0.4 else 0)



y_train_final.head()
# Calculating the VIFs
X_vif=X_new.drop('const',1)
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()

vif['Features'] = X_vif[col].columns

vif['VIF'] = [variance_inflation_factor(X_vif[col].values, i) for i in range(X_vif[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Creating confusion matrix



from sklearn import metrics



confusion=metrics.confusion_matrix(y_train_final.Survived,y_train_final.Predicted)

print(confusion)
print(metrics.accuracy_score(y_train_final.Survived,y_train_final.Predicted))
# Predicting





y_train_pred=res.predict(X_new)



y_train_pred=y_train_pred.values.reshape(-1)



y_train_pred



plt.figure(figsize=(10,8))

sns.set_style('whitegrid')

fpr, tpr,threshold=metrics.roc_curve(y_train_final.Survived,y_train_final.Survive_probablity,drop_intermediate=False)

auc_score=metrics.roc_auc_score(y_train_final.Survived,y_train_final.Survive_probablity)

plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' % auc_score)

plt.plot([0,1],[0,1],'r--')

plt.xlim(0.0,1.0)

plt.ylim(0.0,1.05)

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.legend(loc="lower right")

plt.title('Receiver operating characteristic')

# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(0,10)]

for i in numbers:

    y_train_final[i]= y_train_final.Survive_probablity.map(lambda x: 1 if x > i else 0)

y_train_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensitivity','specificity'])

from sklearn.metrics import confusion_matrix



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_final.Survived, y_train_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensitivity,specificity]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.



cutoff_df.plot.line(x='prob', y=['accuracy','sensitivity','specificity'])

# From the curve above, 0.36 is the optimum point to take as a cutoff probability.





y_train_final['Final_prediction']=y_train_final['Survive_probablity'].map(lambda x:1 if x > 0.36 else 0)



y_train_final.head()
print(metrics.accuracy_score(y_train_final.Survived,y_train_final.Final_prediction))
confusion2=metrics.confusion_matrix(y_train_final.Survived,y_train_final.Final_prediction)

confusion2
TP=confusion2[1,1]  # True positive

TN=confusion2[0,0]  # True negative

FP=confusion2[0,1]  # False positive

FN=confusion2[1,0]  # False negative
# Sensitivity



TP/float(TP+FN)
# Specificity



TN / float(TN+FP)
# Calculate false postive rate - predicting surviving when the passenger was not survived



print(FP/ float(TN+FP))
# Positive predictive value 



print (TP / float(TP+FP))
# Negative predictive value



print (TN / float(TN+ FN))
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_final.Survived,y_train_final.Final_prediction)
recall_score(y_train_final.Survived,y_train_final.Final_prediction)
# Making prediction on test



titanic_test.shape
# Removing 'PassengerId' column and scalling the data



X_test=titanic_test.drop('PassengerId',1)



X_test[['Pclass','Age','Fare','Family_Size']]=scaler.transform(X_test[['Pclass','Age','Fare','Family_Size']])





X_test.shape
X_test_sm=sm.add_constant(X_test)
X_test_sm=X_test_sm.drop(['Fare','Embarked_Q'],1)
y_test_pred=res.predict(X_test_sm)
y_test_pred=pd.DataFrame(y_test_pred)

y_test_pred.head()
y_test_pred=y_test_pred.rename(columns={0 : 'Surviving_Probability'})
y_test_pred['Passenger_id']=titanic_test['PassengerId']
y_test_pred.head()
y_test_pred['Survived']=y_test_pred['Surviving_Probability'].map(lambda x: 1 if x>0.36 else 0)
y_test_pred.head()
y_test_pred_final=y_test_pred[['Passenger_id','Survived']]
y_test_pred_final=y_test_pred_final.rename(columns={'Passenger_id':'PassengerId'})
y_test_pred_final=y_test_pred_final.sort_values(by='PassengerId',axis=0,ascending=True)
y_test_pred_final.shape
#y_test_pred_final.to_csv("Titanic_submission.csv",index=False,header=True)
titanic_test.head()
y_test_pred_final.head()
final_test_titanic=y_test_pred_final.merge(titanic_test,on='PassengerId',how='inner')
plt.figure(figsize=(20,9))



sns.boxplot(x='Family_Size',y='Age',hue='Survived',data=final_test_titanic)
plt.figure(figsize=(20,9))



sns.boxplot(x='Pclass',y='Age',hue='Survived',data=final_test_titanic)
sns.barplot(x='Survived',y='Sex_male',data=final_test_titanic)
sns.barplot(x='Pclass',y='Age',hue='Survived',data=final_test_titanic)