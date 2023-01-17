# Kaggle Reference Link: https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
test.head()
print(train.shape,test.shape)
train.info()
test.info()
train.shape[0]       
(train.isnull().sum()/train.shape[0])*100
(test.isnull().sum()/test.shape[0])*100
train.describe()
test.describe()
train_PassengerId = train['PassengerId']
test_PassengerId = test['PassengerId']
train_drop=train.drop(['Cabin','Name','PassengerId','Ticket'],axis=1,inplace=True)
test_drop=test.drop(['Cabin','Name','PassengerId','Ticket'],axis=1,inplace=True)
train.head()
test.head()
train['Survived'].value_counts()
sns.pairplot(train,x_vars=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'],y_vars='Survived',size=2,aspect=1)
plt.show()
train.isnull().sum()
train['Age'].fillna(train['Age'].median(),inplace=True)
train.isnull().sum()
train.dropna(inplace=True)
test.isnull().sum()
test['Age'].fillna(test['Age'].median(),inplace=True)
test['Fare'].fillna(test['Fare'].median(),inplace=True)
# test.dropna(inplace=True)
test.isnull().sum()
bxplot=train.select_dtypes(include=['float64','int64'])
bxplot.drop('Survived',axis=1,inplace=True)
bxcols=bxplot.columns
bxcols
def bxplott(df): 
    for i in bxcols:
            sns.boxplot(data=df,x=df[i])
            plt.show()
bxplott(train)
Q3=train['Pclass'].quantile(0.85)
Q1=train['Pclass'].quantile(0.15)
IQR=Q3-Q1
train=train[(train['Pclass'] >= Q1 - 1.5*IQR) & (train['Pclass']<= Q3 + 1.5*IQR) ]

Q3=train['Age'].quantile(0.85)
Q1=train['Age'].quantile(0.15)
IQR=Q3-Q1
train=train[(train['Age'] >= Q1 - 1.5*IQR) & (train['Age']<= Q3 + 1.5*IQR) ]

Q3=train['SibSp'].quantile(0.85)
Q1=train['SibSp'].quantile(0.15)
IQR=Q3-Q1
train=train[(train['SibSp'] >= Q1 - 1.5*IQR) & (train['SibSp']<= Q3 + 1.5*IQR) ]

Q3=train['Parch'].quantile(0.85)
Q1=train['Parch'].quantile(0.15)
IQR=Q3-Q1
train=train[(train['Parch'] >= Q1 - 1.5*IQR) & (train['Parch']<= Q3 + 1.5*IQR) ]

Q3=train['Fare'].quantile(0.85)
Q1=train['Fare'].quantile(0.15)
IQR=Q3-Q1
train=train[(train['Fare'] >= Q1 - 1.5*IQR) & (train['Fare']<= Q3 + 1.5*IQR) ]
train.head()
def bxplott(df): 
      for i in bxcols:
            sns.boxplot(data=df,x=df[i])
            plt.show()
bxplott(train)
train['Sex']=train['Sex'].map({'male':0,'female':1})
test['Sex']=test['Sex'].map({'male':0,'female':1})
embarked_train=pd.get_dummies(train['Embarked'],prefix='Embarked',drop_first=True)
train.drop(["Embarked"],axis=1,inplace=True)
train=pd.concat([train,embarked_train],axis=1)
embarked_test=pd.get_dummies(test['Embarked'],prefix='Embarked',drop_first=True)
test.drop(["Embarked"],axis=1,inplace=True)
test=pd.concat([test,embarked_test],axis=1)
train.head()
test.head()
test.isnull().sum()
train['SibSp'].value_counts()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scalecols=['Pclass','Age','Fare','SibSp','Parch']
train[scalecols]=scaler.fit_transform(train[scalecols])
train.head()
scalecols_test=['Pclass','Age','Fare','SibSp','Parch']
test[scalecols_test]=scaler.transform(test[scalecols_test])
test.head()
test.isnull().sum()
y=train['Survived']
X=train.drop(['Survived'],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=100)
X_train.head()
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print("Cross_Val_Score:",cross_val_score(logreg,X_train,y_train,cv = 5,scoring = 'accuracy').mean())
converted=(sum(train['Survived'])/len(train['Survived'].index)*100)
print(converted)
cols= X_train.columns
from sklearn.linear_model import LogisticRegression
from statsmodels import api as sm
logreg=LogisticRegression()
X_train_1 = sm.add_constant(X_train[cols])
model_1 = sm.GLM(y_train,X_train_1,family=sm.families.Binomial()).fit()
print(model_1.summary())
y_train_pred_1=model_1.predict(X_train_1)
y_train_pred_1.head()
y_train_pred_1=y_train_pred_1.values.reshape(-1)
y_train_pred_1[:10]
y_train_pred_1=pd.DataFrame({'Survived':y_train.values,'Survival_Prob':y_train_pred_1})
y_train_pred_1.head()
y_train_pred_1['Index']=y_train.index
y_train_pred_1.head()
y_train_pred_1['Predicted_Value']=y_train_pred_1['Survival_Prob'].map(lambda x:1 if x>0.5 else 0)
y_train_pred_1.head()
from sklearn import metrics
confusion=metrics.confusion_matrix(y_train_pred_1['Survived'],y_train_pred_1['Predicted_Value'])
print(confusion)
print(metrics.accuracy_score(y_train_pred_1['Survived'],y_train_pred_1['Predicted_Value']))
from statsmodels.stats.outliers_influence import variance_inflation_factor
VIF=pd.DataFrame()
VIF['Features']=X_train[cols].columns
VIF['VIF']=[variance_inflation_factor(X_train[cols].values,i) for i in range(X_train[cols].shape[1])]
VIF['VIF']=round(VIF['VIF'],2)
VIF=VIF.sort_values(by='VIF',ascending=False)
VIF.head()
cols=cols.drop(['Embarked_Q'])
X_train_2=sm.add_constant(X_train[cols])
model_2=sm.GLM(y_train,X_train_2,family=sm.families.Binomial()).fit()
print(model_2.summary())
y_train_pred_2=model_2.predict(X_train_2)
y_train_pred_2.head()
y_train_pred_2=pd.DataFrame({'ID':y_train.index,'Survived':y_train,'Survival_Prob':y_train_pred_2})
y_train_pred_2['Predicted_value']=y_train_pred_2['Survival_Prob'].map(lambda x:1 if x>0.5 else 0)
y_train_pred_2.head()
from sklearn import metrics
confusion=metrics.confusion_matrix(y_train_pred_2['Survived'],y_train_pred_2['Predicted_value'])
confusion
accuracy=metrics.accuracy_score(y_train_pred_2['Survived'],y_train_pred_2['Predicted_value'])
accuracy
vif=pd.DataFrame()
vif['Features']=X_train[cols].columns
vif['VIF']=[variance_inflation_factor(X_train[cols].values,i) for i in range(X_train[cols].shape[1])]
VIF['VIF']=round(vif['VIF'],2)
vif.head()
cols=cols.drop(['Embarked_S'])
X_train_3=sm.add_constant(X_train[cols])
model_3=sm.GLM(y_train,X_train_3,family=sm.families.Binomial()).fit()
print(model_3.summary())
y_train_pred_3=model_3.predict(X_train_3)
y_train_pred_3.head()
y_train_pred_3=pd.DataFrame({'ID':y_train.index,'Survived':y_train,'Survival_Prob':y_train_pred_3})
y_train_pred_3['Predicted_value']=y_train_pred_3['Survival_Prob'].map(lambda x:1 if x>0.5 else 0)
y_train_pred_3.head()
confusion=metrics.confusion_matrix(y_train_pred_3['Survived'],y_train_pred_3['Predicted_value'])
confusion
accuracy=metrics.accuracy_score(y_train_pred_3['Survived'],y_train_pred_3['Predicted_value'])
accuracy
vif=pd.DataFrame()
vif['Features']=X_train[cols].columns
vif['VIF']=[variance_inflation_factor(X_train[cols].values,i) for i in range(X_train[cols].shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
vif.head()
cols=cols.drop('Parch')
X_train_4=sm.add_constant(X_train[cols])
model_4=sm.GLM(y_train,X_train_4,family=sm.families.Binomial()).fit()
print(model_4.summary())
y_train_pred_4=model_4.predict(X_train_4)
y_train_pred_4.head()
y_train_pred_4=pd.DataFrame({'ID':y_train.index,'Survived':y_train,'Survival_Prob':y_train_pred_4})
y_train_pred_4['Predicted_value']=y_train_pred_4.Survival_Prob.map(lambda x:1 if x>0.5 else 0)
y_train_pred_4.head()
confusion=metrics.confusion_matrix(y_train_pred_4['Survived'],y_train_pred_4['Predicted_value'])
confusion
accuracy=metrics.accuracy_score(y_train_pred_4['Survived'],y_train_pred_4['Predicted_value'])
accuracy
vif=pd.DataFrame()
vif['Features']=X_train[cols].columns
vif['VIF']=[variance_inflation_factor(X_train[cols].values,i) for i in range(X_train[cols].shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
vif.head()
cols=cols.drop('SibSp')
X_train_5=sm.add_constant(X_train[cols])
model_5=sm.GLM(y_train,X_train_5,family=sm.families.Binomial()).fit()
print(model_5.summary())
y_train_pred_5=model_5.predict(X_train_5)
y_train_pred_5.head()
y_train_pred_5=pd.DataFrame({'ID':y_train.index,'Survived':y_train,'Survival_Prob':y_train_pred_5})
y_train_pred_5['Predicted_value']=y_train_pred_5['Survival_Prob'].map(lambda x:1 if x>0.5 else 0)
y_train_pred_5.head()
confusion=metrics.confusion_matrix(y_train_pred_5['Survived'],y_train_pred_5['Predicted_value'])
confusion
accuracy=metrics.accuracy_score(y_train_pred_5['Survived'],y_train_pred_5['Predicted_value'])
accuracy
vif=pd.DataFrame()
vif['Features']=X_train[cols].columns
vif['VIF']=[variance_inflation_factor(X_train[cols].values,i) for i in range(X_train[cols].shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif.sort_values(by='VIF',ascending=False)
vif
cols=cols.drop('Fare')
### Model 6:
X_train_6=sm.add_constant(X_train[cols])
model_6=sm.GLM(y_train,X_train_6,family=sm.families.Binomial()).fit()
print(model_6.summary())
y_train_pred_6=model_6.predict(X_train_6)
y_train_pred_6.head()
y_train_pred_6=pd.DataFrame({'ID':y_train.index,'Survived':y_train,'Survival_Prob':y_train_pred_6})
y_train_pred_6['Predicted_value']=y_train_pred_6['Survival_Prob'].map(lambda x:1 if x>0.5 else 0)
y_train_pred_6.head()
confusion=metrics.confusion_matrix(y_train_pred_6['Survived'],y_train_pred_6['Predicted_value'])
confusion
accuracy=metrics.accuracy_score(y_train_pred_6['Survived'],y_train_pred_6['Predicted_value'])
accuracy
vif=pd.DataFrame()
vif['Features']=X_train[cols].columns
vif['VIF']=[variance_inflation_factor(X_train[cols].values,i) for i in range(X_train[cols].shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif.sort_values(by='VIF',ascending=False)
vif
TP=confusion[1,1]
TN=confusion[0,0]
FP=confusion[0,1]
FN=confusion[1,0]
print('TP:{0}, TN:{1}, FP:{2}, FN:{3}'.format(TP,TN,FP,FN))
# Sensitivity = TPR = Precision
Sensitivity= round(TP/(TP+FN),2)
# Specificity:
Specificity=round(TN/(TN+FP),2)
# FPR
FPR=round(FP/(FP+TN),2)
# Precision:
Precision=round(TP/(TP+FP),2)
# Recall:
Recall=round(TP/(TP+FN),2)
print('Sensitivity:',Sensitivity,'Specificity:',Specificity,'FPR:',FPR,'Precicion:',Precision,'Recall:',Recall)
def draw_roc(actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs,drop_intermediate = False )
    auc_score = metrics.roc_auc_score(actual, probs )
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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_6['Survived'], y_train_pred_6['Survival_Prob'], drop_intermediate = False )
draw_roc(y_train_pred_6['Survived'], y_train_pred_6['Survival_Prob'])
metrics.auc(fpr,tpr)
numbers=[float(i/10) for i in range(10)]
for i in numbers:
    y_train_pred_6[i]=y_train_pred_6['Survival_Prob'].map(lambda x:1 if x>i else 0)
y_train_pred_6.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame(columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_6['Survived'], y_train_pred_6[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
y_train_pred_6['FinalPredicted']=y_train_pred_6['Survival_Prob'].map(lambda x:1 if x> 0.3 else 0)
y_train_pred_6.head()
accuracy=metrics.accuracy_score(y_train_pred_6['Survived'],y_train_pred_6['FinalPredicted'])
accuracy
confusion=metrics.confusion_matrix(y_train_pred_6['Survived'],y_train_pred_6['FinalPredicted'])
confusion
TP=confusion[1,1]
TN=confusion[0,0]
FP=confusion[0,1]
FN=confusion[1,0]
sensitivity=round(TP/(TP+FN),2)
specificity=round(TN/(TN+FP),2)
precision=round(TP/(TP+FP),2)
print("Sensitivity:",sensitivity ,"Specificity:",specificity ,"Precision:",precision)
p, r, thresholds = metrics.precision_recall_curve(y_train_pred_6['Survived'], y_train_pred_6['Survival_Prob'])
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()
X_test=X_test[cols]
X_test_1=sm.add_constant(X_test)
y_test_pred=model_6.predict(X_test_1)
y_test_pred_1=pd.DataFrame(y_test_pred)
# Let's see the head
y_test_pred_1.head()
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Putting CustID to index
y_test_df['CustID'] = y_test_df.index
y_test_df=pd.concat([y_test_df,y_test_pred_1],axis=1)
y_test_df.rename(columns={0:'Survival_Prob'},inplace=True)
y_test_df['Predicted_Survival']=y_test_df['Survival_Prob'].map(lambda x:1 if x>0.29 else 0)
y_test_df.head(2)
confusion=metrics.confusion_matrix(y_test_df['Survived'],y_test_df['Predicted_Survival'])
confusion
accuracy=metrics.accuracy_score(y_test_df['Survived'],y_test_df['Predicted_Survival'])
accuracy
TP=confusion[1,1]
TN=confusion[0,0]
FP=confusion[0,1]
FN=confusion[1,0]
print('TP:{0}, TN:{1}, FP:{2}, FN:{3}'.format(TP,TN,FP,FN))
# Sensitivity:
sensitivity=round(TP/(TP+FN),2)
# Specificity:
Specificity=round(TN/(TN+FP),2)
# FPR
FPR=round(FP/(FP+TN),2)
# Precision:
Precision=round(TP/(TP+FP),2)
# Recall:
Recall=round(TP/(TP+FN),2)
print('Sensitivity:',sensitivity,'Specificity:',Specificity,'FPR:',FPR,'Precision:',Precision,'Recall:',Recall )

test.shape
test_cols = test[cols]
test_cols.head()
test_sm = sm.add_constant(test_cols)
test_sm
y_test_pred = model_6.predict(test_sm)
y_test_pred[:10]
y_test_pred.shape
# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)
y_pred_1.shape
# Converting y_test to dataframe
y_test_df = pd.DataFrame(test_PassengerId)
y_test_df.shape
# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
print(y_pred_1.shape,y_test_df.shape)
# Appending y_test_df and y_pred_1
submission = pd.concat([y_test_df, y_pred_1],axis=1)
submission.rename(columns={0:'Survival_Prob'},inplace=True)
submission['Survived']=submission['Survival_Prob'].map(lambda x:1 if x>0.29 else 0)
submission.drop(columns=['Survival_Prob'],inplace=True)
submission.head()
submission.to_csv("submission_logistic.csv", index=False)
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
params = {'n_estimators': np.arange(20,101,10), 'max_depth':np.arange(2,16,2),'criterion':['gini', 'entropy'],'min_samples_split':[2,3,4,5,6,7]}

rf_classifier = RandomForestClassifier()
rf_classifier_rs = GridSearchCV(rf_classifier, param_grid=params,n_jobs=-1 )
rf_classifier_rs.fit(X, y)
rf_classifier_rs.best_estimator_
params = {'n_estimators': [20], 'max_depth':[6],'criterion':['gini'],'min_samples_split':[3]}
rf_classifier_rs = GridSearchCV(rf_classifier, param_grid=params,n_jobs=-1 )
rf_classifier_rs.fit(X, y)
y_pred = rf_classifier_rs.predict(test)
df = pd.DataFrame({'PassengerId': test_PassengerId,"Survived": y_pred})
df.to_csv('submission_rf.csv', index=False)
import xgboost as xgb
xgb=xgb.XGBClassifier()
params={'learning_rate':[0.01,0.001], 'max_depth':[8,9], 'gamma':[0,1], 'max_delta_step':[0,1], 'min_child_weight':[1,2], 'n_estimators':[110,115,120,125,130], 'seed':[0,1]}
xgb_classifier_xgb = GridSearchCV(xgb, param_grid=params,n_jobs=-1 )
xgb_classifier_xgb.fit(X, y)
xgb_classifier_xgb.best_estimator_
params = {'learning_rate':[0.01], 'max_depth':[8], 'gamma':[0], 'max_delta_step':[0], 'min_child_weight':[1], 'n_estimators':[110], 'seed':[0]}
xgb_classifier_xgb = GridSearchCV(xgb, param_grid=params,n_jobs=-1)
xgb_classifier_xgb.fit(X, y)
y_pred = xgb_classifier_xgb.predict(test)
submission_xgb= pd.DataFrame({'PassengerId': test_PassengerId,"Survived": y_pred})
submission_xgb.to_csv('submission_xgb.csv', index=False)