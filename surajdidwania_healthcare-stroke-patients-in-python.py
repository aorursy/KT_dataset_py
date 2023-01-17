import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,auc,roc_auc_score,precision_score,recall_score
train_data = pd.read_csv('../input/train_2v.csv')

test_data = pd.read_csv('../input/test_2v.csv')
train_data.shape
test_data.head()
print ('Train Data Shape: {}'.format(train_data.shape))



print ('Test Data Shape: {}'.format(test_data.shape))
train_data.describe()
train_data.isnull().sum()/len(train_data)*100
test_data.isnull().sum()/len(test_data)*100
joined_data = pd.concat([train_data,test_data])
print ('Joined Data Shape: {}'.format(joined_data.shape))
joined_data.isnull().sum()/len(joined_data)*100
train_data["bmi"]=train_data["bmi"].fillna(train_data["bmi"].mean())
train_data.head()
label = LabelEncoder()

train_data['gender'] = label.fit_transform(train_data['gender'])

train_data['ever_married'] = label.fit_transform(train_data['ever_married'])

train_data['work_type']= label.fit_transform(train_data['work_type'])

train_data['Residence_type']= label.fit_transform(train_data['Residence_type'])
train_data_without_smoke = train_data[train_data['smoking_status'].isnull()]

train_data_with_smoke = train_data[train_data['smoking_status'].notnull()]
train_data_without_smoke.drop(columns='smoking_status',axis=1,inplace=True)
train_data_without_smoke.head()
train_data_with_smoke.head()
train_data_with_smoke['smoking_status']= label.fit_transform(train_data_with_smoke['smoking_status'])
train_data_with_smoke.head()

train_data_with_smoke.shape
train_data_with_smoke.corr('pearson')
train_data_with_smoke['stroke'].value_counts()
train_data_without_smoke['stroke'].value_counts()
ros = RandomOverSampler(random_state=0)

smote = SMOTE()
X_resampled, y_resampled = ros.fit_resample(train_data_with_smoke.loc[:,train_data_with_smoke.columns!='stroke'], 

                                            train_data_with_smoke['stroke'])
train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'].columns
print ('ROS Input Data Shape for Smoke Data: {}'.format(X_resampled.shape))

print ('ROS Output Data Shape for Smoke Data: {}'.format(y_resampled.shape))
X_resampled_1, y_resampled_1 = ros.fit_resample(train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'], 

                                            train_data_without_smoke['stroke'])
print ('ROS Input Data Shape for Non Smoke Data: {}'.format(X_resampled_1.shape))

print ('ROS Output Data Shape for Non Smoke Data: {}'.format(y_resampled_1.shape))
X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.2)

print(X_train.shape)

print(X_test.shape)
X_train_1,X_test_1,y_train_1,y_test_1 = train_test_split(X_resampled_1,y_resampled_1,test_size=0.2)

print(X_train_1.shape)

print(X_test_1.shape)
dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)



pred = dtree.predict(X_test)

print(classification_report(y_test,pred))

print (accuracy_score(y_test,pred))

print (confusion_matrix(y_test,pred))



precision = precision_score(y_test,pred)

recall = recall_score(y_test,pred)

print( 'precision = ', precision, '\n', 'recall = ', recall)



y_pred_proba = dtree.predict_proba(X_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)

auc = roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()



impFeatures = pd.DataFrame(dtree.feature_importances_ ,index=train_data_with_smoke.loc[:,train_data_with_smoke.columns!='stroke'].columns,columns=['Importance']).sort_values(by='Importance',ascending=False)

print (impFeatures)
dtree_nosmoke = DecisionTreeClassifier()

dtree_nosmoke.fit(X_train_1,y_train_1)



pred = dtree_nosmoke.predict(X_test_1)

print(classification_report(y_test_1,pred))

print ('Accuracy: {}'.format(accuracy_score(y_test_1,pred)))

print ('COnfusion Matrix: \n {}'.format(confusion_matrix(y_test_1,pred)))



precision = precision_score(y_test_1,pred)

recall = recall_score(y_test_1,pred)

print( 'precision = ', precision, '\n', 'recall = ', recall)



y_pred_proba = dtree_nosmoke.predict_proba(X_test_1)[::,1]

fpr, tpr, _ = roc_curve(y_test_1,  y_pred_proba)

auc = roc_auc_score(y_test_1, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()



impFeatures = pd.DataFrame(dtree_nosmoke.feature_importances_ ,index=train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'].columns,columns=['Importance']).sort_values(by='Importance',ascending=False)

print (impFeatures)
log = LogisticRegression(penalty='l2', C=0.1)

log.fit(X_train,y_train)



pred = log.predict(X_test)

print(classification_report(y_test,pred))

print (accuracy_score(y_test,pred))

print (confusion_matrix(y_test,pred))



precision = precision_score(y_test,pred)

recall = recall_score(y_test,pred)

print( 'precision = ', precision, '\n', 'recall = ', recall)



y_pred_proba = log.predict_proba(X_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)

auc = roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()

impFeatures = pd.DataFrame(log.coef_[0] ,index=train_data_with_smoke.loc[:,train_data_with_smoke.columns!='stroke'].columns,columns=['Importance']).sort_values(by='Importance',ascending=False)

print (impFeatures)
logg = LogisticRegression(penalty='l2', C=0.1)

logg.fit(X_train_1,y_train_1)



pred = logg.predict(X_test_1)

print(classification_report(y_test_1,pred))

print (accuracy_score(y_test_1,pred))

print (confusion_matrix(y_test_1,pred))



precision = precision_score(y_test_1,pred)

recall = recall_score(y_test_1,pred)

print( 'precision = ', precision, '\n', 'recall = ', recall)



y_pred_proba = logg.predict_proba(X_test_1)[::,1]

fpr, tpr, _ = roc_curve(y_test_1,  y_pred_proba)

auc = roc_auc_score(y_test_1, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()



impFeatures = pd.DataFrame(logg.coef_[0] ,index=train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'].columns,columns=['Importance']).sort_values(by='Importance',ascending=False)

print (impFeatures)
ran = RandomForestClassifier(n_estimators=50,random_state=0)

ran.fit(X_train_1,y_train_1)



pred = ran.predict(X_test_1)

print(classification_report(y_test_1,pred))

print (accuracy_score(y_test_1,pred))

print (confusion_matrix(y_test_1,pred))



precision = precision_score(y_test_1,pred)

recall = recall_score(y_test_1,pred)

print( 'precision = ', precision, '\n', 'recall = ', recall)



y_pred_proba = ran.predict_proba(X_test_1)[::,1]

fpr, tpr, _ = roc_curve(y_test_1,  y_pred_proba)

auc = roc_auc_score(y_test_1, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()





impFeatures = pd.DataFrame((ran.feature_importances_) ,index=train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'].columns,columns=['Importance']).sort_values(by='Importance',ascending=False)

print (impFeatures)
feat_importances = pd.Series(ran.feature_importances_, index=train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'].columns)

feat_importances.plot(kind='barh')
test_data["bmi"]=test_data["bmi"].fillna(test_data["bmi"].mean())
test_data.drop(axis=1,columns=['smoking_status'],inplace=True)
label = LabelEncoder()

test_data['gender'] = label.fit_transform(test_data['gender'])

test_data['ever_married'] = label.fit_transform(test_data['ever_married'])

test_data['work_type']= label.fit_transform(test_data['work_type'])

test_data['Residence_type']= label.fit_transform(test_data['Residence_type'])

pred = ran.predict(test_data)
prediction = pd.DataFrame(pred,columns=['Pred'])
prediction['Pred'].value_counts()