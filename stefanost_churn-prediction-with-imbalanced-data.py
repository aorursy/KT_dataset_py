import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df.info()
df[df['TotalCharges']==" "]
df['TotalCharges']=df['TotalCharges'].replace(" ",0).astype('float32')
#percentage of classes
ch=df[df['Churn']=='Yes']
no_ch=df[df['Churn']=='No']
print('churn percentage-->',(ch.shape[0]/df.shape[0])*100)
print('no churn percentage-->',(no_ch.shape[0]/df.shape[0])*100)

df['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%');
data=df.copy()
def pie(features):
    for feature in features:
        plt.figure(figsize=(10,10))
        plt.subplot(1,2,1)
        data[data['Churn']=='Yes'][feature].value_counts().plot(kind='pie', autopct='%1.1f%%');
        plt.title('Churn');
        plt.subplot(1,2,2)
        data[data['Churn']=='No'][feature].value_counts().plot(kind='pie', autopct='%1.1f%%');
        plt.title('No Churn');

features=['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod']
pie(features)
def kde(feature):
    plt.figure(figsize=(14,4))
    plt.title('Distribution of {}'. format(feature))
    sns.kdeplot(data[data['Churn']=='Yes'][feature], label='Churn');
    sns.kdeplot(data[data['Churn']=='No'][feature], label='No Churn');

kde('tenure')
kde('MonthlyCharges')
kde('TotalCharges')
def box(feature):
    plt.figure(figsize=(4,4))
    sns.boxplot(x='Churn', y=feature, data=data);
    
box('tenure')
box('MonthlyCharges')    
box('TotalCharges')
#step-by-step dummy encoding, 
#encoding one column at a time and deleting redundant columns

data.drop(columns=data.columns[0],inplace=True)

data['Male']=pd.get_dummies(data.iloc[:,0], drop_first=True)
data.drop(columns=data.columns[0],inplace=True)

data['Partner_yes']=pd.get_dummies(data.iloc[:,1],drop_first=True)
data.drop(columns=data.columns[1], inplace=True)

data['Dependent_yes']=pd.get_dummies(data.iloc[:,1],drop_first=True)
data.drop(columns=data.columns[1], inplace=True)

data['Phone_service_yes']=pd.get_dummies(data.iloc[:,2],drop_first=True)
data.drop(columns=data.columns[2], inplace=True)

data['multiple_lines_yes']=pd.get_dummies(data.iloc[:,2]).iloc[:,-1]
data.drop(columns=data.columns[2], inplace=True)

internet=pd.get_dummies(data.iloc[:,2],prefix='Internet')
data=pd.concat([data,internet],axis=1).drop(columns=['InternetService'])

data['online security_yes']=pd.get_dummies(data.iloc[:,2]).iloc[:,2]
data.drop(columns='OnlineSecurity',inplace=True)

data['online backup_yes']=pd.get_dummies(data.iloc[:,2]).iloc[:,2]
data.drop(columns='OnlineBackup',inplace=True)

data['device protection_yes']=pd.get_dummies(data.iloc[:,2]).iloc[:,2]
data.drop(columns='DeviceProtection',inplace=True)

data['tech support_yes']=pd.get_dummies(data.iloc[:,2]).iloc[:,2]
data.drop(columns='TechSupport',inplace=True)

data['streamingTV_yes']=pd.get_dummies(data.iloc[:,2]).iloc[:,2]
data.drop(columns='StreamingTV',inplace=True)

data['streaming movies_yes']=pd.get_dummies(data.iloc[:,2]).iloc[:,2]
data.drop(columns='StreamingMovies',inplace=True)

contract=pd.get_dummies(data.iloc[:,2],prefix='contract')
data=pd.concat([data,contract],axis=1).drop(columns=['Contract'])

data['paperless biling_yes']=pd.get_dummies(data.iloc[:,2]).iloc[:,1]
data.drop(columns='PaperlessBilling',inplace=True)

paymethod=pd.get_dummies(data.iloc[:,2],prefix='paymethod')
data=pd.concat([data,paymethod],axis=1).drop(columns=['PaymentMethod'])
data.head(2)
#separate data and labels
y=data['Churn']
data.drop(columns='Churn', inplace=True)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#feature selection via random forest
forest=RandomForestClassifier(n_estimators=600, max_depth=5, random_state=7)
forest.fit(data,y)
imp=forest.feature_importances_

#store feature importances in new DataFrame
feature_importances=pd.DataFrame()
feature_importances['feature']=pd.Series(data.columns)
feature_importances['importance']=imp
feature_importances.head()
plt.figure(figsize=(10,10))
sns.barplot(x='importance', y='feature', 
            data=feature_importances.sort_values(by='importance',ascending=False));
#keep most important columns and create final training dataset
cols=feature_importances.sort_values(by='importance',ascending=False).iloc[:12,0].values
x=data[cols].values
data[cols].head(2)
#encode labels, and train_test_split
enc=LabelEncoder()
y=enc.fit_transform(y)

x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,stratify=y, random_state=77)
#scaling data, only numerical columns
from sklearn.preprocessing import StandardScaler
num_cols=[1,3,4]#numerical columns(tenure,total charges, monthly charges)
sc=StandardScaler()
x_tr[:,num_cols]=sc.fit_transform(x_tr[:,num_cols])
x_ts[:,num_cols]=sc.transform(x_ts[:,num_cols])
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
x_tr_pca=pca.fit_transform(x_tr)

x_viz=pd.concat(objs=[pd.DataFrame(x_tr_pca),pd.Series(y_tr)],axis=1).values

plt.figure(figsize=(10,10))
ax=plt.axes()
xv=x_viz[:,0]
yv=x_viz[:,1]
zv=x_viz[:,2]
cv=x_viz[:,3]
ax.scatter(xv, yv, c=cv, cmap='winter')
plt.show();
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score

#DataFrame to store performance metrics for later comparison between models
results=pd.DataFrame([], columns=['model', 'parameters','accuracy','precision','recall','F1-score'])
nb=GaussianNB()
nb.fit(x_tr,y_tr)
print('accuracy:',accuracy_score(y_ts, nb.predict(x_ts)))
sns.heatmap(confusion_matrix(y_ts, nb.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
knn=KNeighborsClassifier()
knn.fit(x_tr,y_tr)
print('accuracy:',accuracy_score(y_ts, knn.predict(x_ts)))
sns.heatmap(confusion_matrix(y_ts, knn.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
svm=SVC()
svm.fit(x_tr,y_tr)
print('accuracy:',accuracy_score(y_ts, svm.predict(x_ts)))
sns.heatmap(confusion_matrix(y_ts, svm.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
sgd=SGDClassifier()
sgd.fit(x_tr, y_tr)
print('accuracy:',accuracy_score(y_ts, sgd.predict(x_ts)))
sns.heatmap(confusion_matrix(y_ts, sgd.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
lr=LogisticRegression()
lr.fit(x_tr,y_tr)
print('accuracy"',accuracy_score(y_ts, lr.predict(x_ts)))
sns.heatmap(confusion_matrix(y_ts, lr.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
tree=DecisionTreeClassifier(max_depth=4, random_state=3)
tree.fit(x_tr,y_tr)
print('accuracy:',accuracy_score(y_ts, tree.predict(x_ts)))
sns.heatmap(confusion_matrix(y_ts, tree.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
rf=RandomForestClassifier(n_estimators=100,max_depth=8,random_state=17)
rf.fit(x_tr,y_tr)
print('accuracy:',accuracy_score(y_ts, rf.predict(x_ts)))
sns.heatmap(confusion_matrix(y_ts, rf.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
#dummy classifier
from sklearn.dummy import DummyClassifier

dum=DummyClassifier(strategy='most_frequent')
dum.fit(x_tr,y_tr)
pred=dum.predict(x_ts)
print('dummy class:',format(np.unique(pred)))
print('dummy accuracy:',accuracy_score(y_ts,pred) )
sns.heatmap(confusion_matrix(y_ts, dum.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
#another way to show this, array with all zeros
accuracy_score(y_ts,np.zeros(x_ts.shape[0]))
svm=SVC(class_weight='balanced')
svm.fit(x_tr,y_tr)
print('accuracy:',accuracy_score(y_ts, svm.predict(x_ts)))
sns.heatmap(confusion_matrix(y_ts, svm.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
sgd=SGDClassifier(class_weight='balanced')
sgd.fit(x_tr, y_tr)
print('accuracy:',accuracy_score(y_ts, sgd.predict(x_ts)))
sns.heatmap(confusion_matrix(y_ts,sgd.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
lr=LogisticRegression(class_weight='balanced')
lr.fit(x_tr,y_tr)
print('accuracy:',accuracy_score(y_ts, lr.predict(x_ts)))
sns.heatmap(confusion_matrix(y_ts, lr.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
tree=DecisionTreeClassifier(max_depth=4, random_state=3,class_weight='balanced')
tree.fit(x_tr,y_tr)
print('accuracy:',accuracy_score(y_ts, tree.predict(x_ts)))
sns.heatmap(confusion_matrix(y_ts, tree.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
rf=RandomForestClassifier(n_estimators=100,max_depth=8,
                          random_state=17,class_weight='balanced')
rf.fit(x_tr,y_tr)
print('accuracy:',accuracy_score(y_ts, rf.predict(x_ts)))
sns.heatmap(confusion_matrix(y_ts, rf.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
from sklearn.utils import resample
from sklearn.dummy import DummyClassifier

x_up, y_up=resample(x_tr[y_tr==1],y_tr[y_tr==1],replace=True,
                        n_samples=x_tr[y_tr==0].shape[0],random_state=42)
print(x_tr[y_tr==1].shape)
print(x_up.shape)

x_bal=np.vstack((x_tr[y_tr==0], x_up))
y_bal=np.hstack((y_tr[y_tr==0],y_up))


dum2=DummyClassifier(strategy='most_frequent')
dum2.fit(x_bal,y_bal)
print('dummy accuracy on balanced dataset:',
      accuracy_score(y_bal,dum2.predict(x_bal)))
gb=GaussianNB()
gb.fit(x_bal,y_bal)
y_pred=gb.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred)
recall=recall_score(y_ts,y_pred)
f1=f1_score(y_ts,y_pred)
results=results.append(pd.DataFrame([['Gaussian NB', 'default', accuracy,precision,recall,f1]],columns=list(results.columns)))
print('accuracy:',accuracy_score(y_ts, y_pred))
sns.heatmap(confusion_matrix(y_ts, gb.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
knn=KNeighborsClassifier()
knn.fit(x_bal,y_bal)
y_pred=knn.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred)
recall=recall_score(y_ts,y_pred)
f1=f1_score(y_ts,y_pred)
results=results.append(pd.DataFrame([['KNN', 'default', accuracy,precision,recall,f1]],columns=list(results.columns)))
print('accuracy:',accuracy_score(y_ts, y_pred))
sns.heatmap(confusion_matrix(y_ts, knn.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
svm=SVC()
rng=[0.01, 0.1, 1.0, 10.0, 100.0]
params={'C':rng, 'gamma':rng}
gs=GridSearchCV(estimator=svm,param_grid=params)


gs.fit(x_bal,y_bal)
best_params=gs.best_params_
best_est=gs.best_estimator_

print('best params',best_params)

best_est.fit(x_bal,y_bal)

y_pred=best_est.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred)
recall=recall_score(y_ts,y_pred)
f1=f1_score(y_ts,y_pred)
results=results.append(pd.DataFrame([['SVM', best_params, accuracy,precision,recall,f1]],columns=list(results.columns)))
print('accuracy:',accuracy_score(y_ts, y_pred))

sns.heatmap(confusion_matrix(y_ts, y_pred),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');

svm=SVC()
svm.fit(x_bal,y_bal)


y_pred=svm.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred)
recall=recall_score(y_ts,y_pred)
f1=f1_score(y_ts,y_pred)
results=results.append(pd.DataFrame([['SVM', 'default', accuracy,precision,recall,f1]],columns=list(results.columns)))
print('accuracy:',accuracy_score(y_ts, y_pred))

sns.heatmap(confusion_matrix(y_ts, svm.predict(x_ts)),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
sgd=SGDClassifier(random_state=3)
params={'loss':['log', 'modified_huber', 'squared_hinge'],
       'penalty': ['l1','l2'],
       'alpha':[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
gs=GridSearchCV(estimator=sgd, param_grid=params)
gs.fit(x_bal, y_bal)
best=gs.best_estimator_
best_params=gs.best_params_

best.fit(x_bal,y_bal)

y_pred=best.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred)
recall=recall_score(y_ts,y_pred)
f1=f1_score(y_ts,y_pred)
results=results.append(pd.DataFrame([['SGC', best_params, accuracy,precision,recall,f1]],columns=list(results.columns)))


print('best parameters:',best_params)

print('best estimator accuracy:',accuracy_score(y_ts, y_pred))
sns.heatmap(confusion_matrix(y_ts, y_pred),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
lr=LogisticRegression(random_state=3)
params={'penalty':['l2'],
       'C':[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
       'solver': [ 'sag','saga', 'lbfgs']}
gs=GridSearchCV(estimator=lr,param_grid=params)
gs.fit(x_bal,y_bal)
best=gs.best_estimator_
best_params=gs.best_params_
print('best params:', best_params)
y_pred=best.predict(x_ts)

best.fit(x_bal,y_bal)

y_pred=best.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred)
recall=recall_score(y_ts,y_pred)
f1=f1_score(y_ts,y_pred)
results=results.append(pd.DataFrame([['Logistic Regression', best_params, accuracy,precision,recall,f1]],columns=list(results.columns)))
print('accuracy"',accuracy_score(y_ts, y_pred))
sns.heatmap(confusion_matrix(y_ts, y_pred),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
tree=DecisionTreeClassifier(max_depth=4, random_state=3)
tree.fit(x_bal,y_bal)

y_pred=tree.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred)
recall=recall_score(y_ts,y_pred)
f1=f1_score(y_ts,y_pred)
results=results.append(pd.DataFrame([['Decision Tree', 'max_depth=4, rand_state=3', accuracy,precision,recall,f1]],columns=list(results.columns)))
print('accuracy:',accuracy)
sns.heatmap(confusion_matrix(y_ts, y_pred),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
rf=RandomForestClassifier(n_estimators=1000,max_depth=10,random_state=17)
rf.fit(x_bal,y_bal)


y_pred=rf.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred)
recall=recall_score(y_ts,y_pred)
f1=f1_score(y_ts,y_pred)
results=results.append(pd.DataFrame([['Random Forest', 'max_depth=10, rand_state=17', accuracy,precision,recall,f1]],columns=list(results.columns)))
print('accuracy:',accuracy_score(y_ts, y_pred))

sns.heatmap(confusion_matrix(y_ts, y_pred),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
tree=DecisionTreeClassifier(criterion='entropy',random_state=1, max_depth=1)
ada=AdaBoostClassifier(base_estimator=tree,n_estimators=1000,learning_rate=0.1, random_state=5)
ada.fit(x_bal,y_bal)

y_pred=ada.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred)
recall=recall_score(y_ts,y_pred)
f1=f1_score(y_ts,y_pred)
results=results.append(pd.DataFrame([['AdaBoost Tree', 'criterion=entropy, max_depth=1, rate=0.1, estimators=500',
                                      accuracy,precision,recall,f1]],columns=list(results.columns)))
print('accuracy:',accuracy)
sns.heatmap(confusion_matrix(y_ts, y_pred),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
gradb=GradientBoostingClassifier(random_state=42)
gradb.fit(x_bal,y_bal)

y_pred=gradb.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred)
recall=recall_score(y_ts,y_pred)
f1=f1_score(y_ts,y_pred)
results=results.append(pd.DataFrame([['GradientBoosted Tree', 'default, random_state=42', accuracy,precision,recall,f1]],columns=list(results.columns)))
print('accuracy:',accuracy)
sns.heatmap(confusion_matrix(y_ts, y_pred),annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results=results.reset_index().drop(columns='index')
results
sns.catplot(y='model', x='accuracy', kind='bar', data=results.sort_values(by='accuracy',ascending=False), color='grey');
plt.title('Model Accuracy');
print(' ')
sns.catplot(y='model', x='F1-score', kind='bar', data=results.sort_values(by='F1-score',ascending=False), color='black');
plt.title('model F1-score' );