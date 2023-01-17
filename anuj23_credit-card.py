# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))











# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("../input/creditcard.csv")
dataset.head()
print(dataset.shape)

dataset.describe()
dataset.head()
ax=sns.countplot(x="Class",data=dataset,color="blue")
X=dataset.iloc[:,0:30].values
#print(X.tail())
y=dataset.iloc[:,[30]].values
#print(y.head())
#If there is any missing value in data

dataset.isnull().sum()
dataset.corr()
ax=dataset.hist(figsize=(20,20))


from sklearn.preprocessing import StandardScaler
x_scale=StandardScaler()
X[:,[0,29]]=x_scale.fit_transform(X[:,[0,29]])
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=50,random_state=0)
classifier.fit(X_train,y_train.ravel())
y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
c_metric=confusion_matrix(y_test,y_pred)
print(accuracy)
print(c_metric)
##start under sample

frauds=len(dataset[dataset['Class']==1])
print(frauds)
non_fraud_indices=dataset[dataset['Class']==0].index
print(non_fraud_indices)
random_indices=np.random.choice(non_fraud_indices,frauds,replace='False')
print(len(random_indices))
fraud_indices=dataset[dataset["Class"]==1].index
print(fraud_indices)
under_sample_indices=np.concatenate([fraud_indices,random_indices])
print(under_sample_indices)
under_sample=dataset.loc[under_sample_indices]
print(under_sample)
sns.countplot(x='Class',data=under_sample)

#end of under sampling
x_under=under_sample.iloc[:,0:30].values
y_under=under_sample.iloc[:,[30]].values
x_under.shape
y_under
from sklearn.preprocessing import StandardScaler
under_scale=StandardScaler()
x_under[:,[0,29]]=under_scale.fit_transform(x_under[:,[0,29]])
from sklearn.cross_validation import train_test_split
x_under_train,x_under_test,y_under_train,y_under_test=train_test_split(x_under,y_under,test_size=0.3
                                                                      ,random_state=0)
from sklearn.ensemble import RandomForestClassifier
under_classifier=RandomForestClassifier(n_estimators=25,criterion='gini',max_depth=3
                                       ,min_samples_split=4,min_weight_fraction_leaf=0.05
                                       ,min_samples_leaf=2,oob_score=True)
under_classifier.fit(x_under_train,y_under_train.ravel())
y_pred_under=under_classifier.predict(x_under_test)
feat=under_classifier.feature_importances_
print(feat)
oob=under_classifier.oob_score_
print(oob)
path=under_classifier.decision_path
print(path)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_under_test,y_pred_under)
print(classification_report(y_under_test,y_pred_under))
c_metric=confusion_matrix(y_under_test,y_pred_under)
print(accuracy)
print(c_metric)
from sklearn.model_selection import cross_val_score
cross=cross_val_score(estimator=under_classifier,X=x_under_train,y=y_under_train,cv=10)
print(cross.mean())
print(cross.std())
from sklearn.svm import SVC
svc_under_classifier=SVC(C=1,kernel='rbf',gamma='auto')
svc_under_classifier.fit(x_under_train,y_under_train.ravel())
svc_y_pred_under=svc_under_classifier.predict(x_under_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_under_test,svc_y_pred_under)
print(classification_report(y_under_test,svc_y_pred_under))
c_metric=confusion_matrix(y_under_test,svc_y_pred_under)
print(accuracy)
print(c_metric)
from sklearn.model_selection import cross_val_score
cross=cross_val_score(estimator=svc_under_classifier,X=x_under_train,y=y_under_train,cv=10)
print(cross.mean())
print(cross.std())
from sklearn.linear_model import LogisticRegression
regression_class=LogisticRegression()
regression_class.fit(x_under_train,y_under_train.ravel())
log_y_pred_under=regression_class.predict(x_under_test)

coeff_df=pd.DataFrame(dataset.columns.delete(30))
coeff_df.columns=['Feature']
coeff_df['correlation']=pd.Series(regression_class.coef_[0])
coeff_df=coeff_df.sort_values('correlation',ascending=False)
coeff_df
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_under_test,log_y_pred_under)
print(classification_report(y_under_test,log_y_pred_under))
c_metric=confusion_matrix(y_under_test,log_y_pred_under)
print(accuracy)
print(c_metric)
from sklearn.model_selection import cross_val_score
cross=cross_val_score(estimator=regression_class,X=x_under_train,y=y_under_train,cv=10)
print(cross.mean())
print(cross.std())
from sklearn.neighbors import KNeighborsClassifier
k_neigh_class=KNeighborsClassifier(n_neighbors=3,algorithm='kd_tree',leaf_size=25)
k_neigh_class.fit(x_under_train,y_under_train)
k_y_pred=k_neigh_class.predict(x_under_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_under_test,k_y_pred)
print(classification_report(y_under_test,k_y_pred))
c_metric=confusion_matrix(y_under_test,k_y_pred)
print(accuracy)
print(c_metric)
from sklearn.model_selection import cross_val_score
cross=cross_val_score(estimator=k_neigh_class,X=x_under_train,y=y_under_train,cv=10)
print(cross.mean())
print(cross.std())
data=dataset.sample(frac=0.1,random_state=1)
#print(dataset.shape)
print(data.shape)
#Calculate no of fraud and normal transaction in sample data

fraud=data[data["Class"]==1]
normal=data[data["Class"]==0]

outlier_fraction=len(fraud)/float(len(normal))
print(outlier_fraction)
print("Fraud Transaction:{}".format(len(fraud)))
print("Normal Transaction:{}".format(len(normal)))

X_a=data.iloc[:,0:30].values
y_a=data.iloc[:,[30]].values
print(X_a.shape)
print(y_a.shape)
from sklearn.ensemble import IsolationForest
i_classifier=IsolationForest(n_estimators=100,max_samples=len(y_a),contamination=outlier_fraction,
                             random_state=1)
i_classifier.fit(X_a)
i_y_pred=i_classifier.predict(X_a)

#reshape the prediction values to 0 for valid and 1 for fraud
i_y_pred[i_y_pred==1]=0
i_y_pred[i_y_pred==-1]=1
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_a,i_y_pred))
cm=confusion_matrix(y_a,i_y_pred)
ac=accuracy_score(y_a,i_y_pred)
print(cm)
print(ac)

n_errors=(i_y_pred!=y_a).sum()
print('{}'.format(n_errors))
parameter={'n_estimators':[50,100,120],'max_features':[0.5,1.0,1.5],'bootstrap':[True,False]}
from sklearn.model_selection import GridSearchCV
gd_cv=GridSearchCV(estimator=i_classifier,param_grid=parameter)
gd_cv.fit(X_a,y_a)

from sklearn.neighbors import LocalOutlierFactor
lo_classifier=LocalOutlierFactor(n_neighbors=40,contamination=outlier_fraction,novelty=True)
lo_classifier.fit(X_a)
lo_y_pred=lo_classifier.predict(X_a)
lo_y_pred[lo_y_pred==1]=0
lo_y_pred[lo_y_pred==-1]=1
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_a,lo_y_pred))
cm=confusion_matrix(y_a,lo_y_pred)
ac=accuracy_score(y_a,lo_y_pred)
print(cm)
print(ac)

n_errors=(lo_y_pred!=y_a).sum()
print('{}'.format(n_errors))
from sklearn.ensemble import GradientBoostingClassifier
gb_classifier=GradientBoostingClassifier()
gb_classifier.fit(X_train,y_train.ravel())


gb_y_pred=gb_classifier.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
ac_score=accuracy_score(y_test,gb_y_pred)
con_matrix=confusion_matrix(y_test,gb_y_pred)

print(classification_report(y_test,gb_y_pred))

print(ac_score)
print(con_matrix)

Normal.Amount.describe()
dataset.describe()
Fraud=data1[data1["Class"]==1]
Valid=data1[data1["Class"]==0]
outlier_fraction=len(Fraud)/float(len(Valid))
print(outlier_fraction)
print("Fraud: {}".format(len(Fraud)))
print("Normal: {}".format(len(Valid)))
correlation_matrix=data1.corr()
fig=plt.figsize=(12,9)
sns.heatmap(correlation_matrix,vmax=0.8,square=True)
print(correlation_matrix)
#plt.show()
classifiers={'Isolation Forest':IsolationForest(n_estimators=100,max_samples=len(X), contamination=outlier_fraction,random_state=state,
                                               verbose=0),
            'Local Outlier Factor':LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
             'Support Vector Machine':OneClassSVM(kernel='rbf',degree=3,gamma=0.1,nu=0.05,max_iter=-1,
                                                 random_state=state)
            }








