import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dataset = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

print('Dataset: ',dataset.head(10))

print('*******************************************************')

print('Dataset Shape: ',dataset.shape)

print('*******************************************************')

print(dataset.columns)

print('*******************************************************')

# More arrangement

#for col in dataset.columns:

#  print('Columns: ',col)

#
dataset.drop(['id','Unnamed: 32'],axis=1,inplace=True)

print(dataset.head(10))
print(dataset['diagnosis'].value_counts())

print('*******************************************************')

import seaborn as sns

y=dataset.diagnosis

ax=sns.countplot(y,label='Count')

B,M=y.value_counts()

print('Number of Benign',B)

print('Number of Malignant',M)
Positive=dataset[dataset['diagnosis'].isin(['M'])]

Negative=dataset[dataset['diagnosis'].isin(['B'])]

print('Positive',Positive)

print('*******************************************************')

print('Negative',Negative)
dataset.var()
dataset.corr()
import seaborn as sns

plt.figure(figsize=(25, 12))

sns.heatmap(dataset.corr(), annot=True)

plt.show()
dataset.corr()['radius_mean'].plot(kind='bar')
sns.pairplot(dataset,hue='diagnosis')
X=dataset.iloc[:,1:].values

y=dataset.iloc[:,:1].values

print('X: ',X[:10,:])

print('*******************************************************')

print('X Shape: ',X.shape)

print('*******************************************************')

print('Y: ',y[:10,:])

print('*******************************************************')

print('Y Shape: ',y.shape)

print('*******************************************************')
dataset.isnull().sum()
import missingno as msno

msno.bar(dataset)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

X=imputer.fit_transform(X)

print('X: ',X[:10,:])
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

y=le.fit_transform(y)

print(y)
from sklearn.feature_selection import SelectPercentile

from sklearn.feature_selection import chi2

print('Original X Shape: ',X.shape)

FeatureSelection=SelectPercentile(score_func=chi2,percentile=85)

X = FeatureSelection.fit_transform(X, y)

print('X Shape: ',X.shape)

print('Feature Selected: ',FeatureSelection.get_support())
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X=sc.fit_transform(X)

print(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

print('X_Train')

print(X_train)

print('****************************')

print('X_test')

print(X_test)

print(X_test.shape)

print('****************************')

print('Y_train')

print(y_train)

print('****************************')

print('Y_test')

print(y_test)
from sklearn.linear_model import LogisticRegression

LR=LogisticRegression(random_state=0)

LR.fit(X_train,y_train)

print('Logistic Regression Train Score : ',LR.score(X_train,y_train))

print('Logistic Regression Test Score : ',LR.score(X_test,y_test))
y_pred_LR=LR.predict(X_test)

print('Y Test: ',y_test)

print('Y Predict :',y_pred_LR)

print(np.concatenate((y_pred_LR.reshape(len(y_pred_LR),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import accuracy_score

accuracy_s=accuracy_score(y_test,y_pred_LR)

print(accuracy_s)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred_LR)

print(cm)

sns.heatmap(cm,center=True)

plt.show
from sklearn.metrics import classification_report

cr=classification_report(y_test,y_pred_LR)

print(cr)
from sklearn.metrics import roc_curve,auc

LR_tpr,LR_fpr,threshold=roc_curve(y_test,y_pred_LR)

LR_auc = auc(LR_tpr, LR_fpr)

print('LR_tpr: ',LR_tpr)

print('LR_fpr: ',LR_fpr)

print('threshold: ',threshold)



#Draw ROC Curve && AUC [Area Under The Curve]

plt.figure(figsize=(9, 8))

plt.plot(LR_tpr, LR_fpr, marker='o', label='Logistic Regression (auc = %0.3f)' % LR_auc)

plt.ylabel('True Positive Rate -->')

plt.xlabel('False Positive Rate -->')



plt.legend()



plt.show()
from sklearn.svm import SVC

s_v_m=SVC(random_state=0)

s_v_m.fit(X_train,y_train)

print('SVM Train Score',s_v_m.score(X_train,y_train))

print('SVM Test Score',s_v_m.score(X_test,y_test))
y_pred_svm=s_v_m.predict(X_test)

print('Y Test: ',X_test)

print('Y Predict: ,',y_pred_svm)

print(np.concatenate((y_pred_svm.reshape(len(y_pred_svm),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import accuracy_score

accuracy_s=accuracy_score(y_test,y_pred_svm)

print(accuracy_s)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred_svm)

print(cm)

plt.figure(figsize=(9,5))

sns.heatmap(cm,center=True)

plt.show()
from sklearn.metrics import classification_report

cr=classification_report(y_test,y_pred_svm)

print(cr)
from sklearn.metrics import roc_curve,auc

svm_tpr,svm_fpr,threshold=roc_curve(y_test,y_pred_svm)

svm_auc=auc(svm_tpr,svm_fpr)

print('svm_tpr',svm_tpr)

print('svm_fpr',svm_fpr)

print('threshold',threshold)





#Draw ROC Curve && AUC [Area Under The Curve]

plt.figure(figsize=(9, 5))

plt.plot(svm_tpr, svm_fpr, linestyle=':', label='SVM (auc = %0.3f)' % svm_auc)



plt.xlabel('False Positive Rate -->')

plt.ylabel('True Positive Rate -->')



plt.legend()



plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=13,metric = 'minkowski', p = 2)

knn.fit(X_train,y_train)

print('KNN Train Score :',knn.score(X_train,y_train))

print('KNN Test Score :',knn.score(X_test,y_test))
y_pred_knn=knn.predict(X_test)

print('Y Test:',y_test)

print('Y Predict:',y_pred_knn)

print(np.concatenate((y_pred_knn.reshape(len(y_pred_knn),1),y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import accuracy_score

accuracy_s=accuracy_score(y_test,y_pred_knn)

print('Accuracy Score:',accuracy_s)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred_knn)

print(cm)

plt.figure(figsize=(9,5))

sns.heatmap(cm,center=True)

plt.show()
from sklearn.metrics import classification_report

cr=classification_report(y_test,y_pred_knn)

print(cr)
from sklearn.metrics import roc_curve,auc

knn_tpr,knn_fpr,threshold=roc_curve(y_test,y_pred_knn)

knn_auc=auc(knn_tpr,knn_fpr)

print('knn_tpr',knn_tpr)

print('knn_fpr',knn_fpr)

print('threhold',threshold)



plt.figure(figsize=(9,5))

plt.plot(knn_tpr,knn_fpr,linestyle='-', label='KNN (auc = %0.3f)' % knn_auc)



plt.xlabel('False Positive Rate -->')

plt.ylabel('True Positive Rate -->')



plt.legend()



plt.show()
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth=3)

dt.fit(X_train,y_train)

print('Decision Tree Train Score :',dt.score(X_train,y_train))

print('Decision Tree Test Score :',dt.score(X_test,y_test))

print('DecisionTreeClassifierModel feature importances are :\n ' , dt.feature_importances_)

print('----------------------------------------------------')

y_pred_dt=dt.predict(X_test)

print('Y Test: ',y_test)

print('Y Pred',y_pred_dt)

print(np.concatenate((y_pred_dt.reshape(len(y_pred_dt),1),y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import accuracy_score

accuracy_s=accuracy_score(y_test,y_pred_dt)

print('Accuracy Score',accuracy_s)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred_dt)

print(cm)

plt.figure(figsize=(9,5))

sns.heatmap(cm,center=True)

plt.show()
from sklearn.metrics import classification_report

cr=classification_report(y_test,y_pred_dt)

print(cr)
from sklearn.metrics import roc_curve,auc

dt_tpr,dt_fpr,threshold=roc_curve(y_test,y_pred_dt)

dt_auc=auc(dt_tpr,dt_fpr)

print('dt_tpr Value  : ', dt_tpr)

print('dt_fpr Value  : ', dt_fpr)

print('thresholds Value  : ', threshold)



#Draw ROC Curve && AUC [Area Under The Curve]



plt.figure(figsize=(5, 5), dpi=100)

plt.plot(dt_tpr, dt_fpr, linestyle='--', label='DecisionTree (auc = %0.3f)' % dt_auc)



plt.xlabel('True Positive Rate -->')

plt.ylabel('False Positive Rate -->')



plt.legend()

plt.show()
from sklearn.ensemble import RandomForestClassifier 

rf=RandomForestClassifier(criterion = 'entropy',random_state=0,max_depth=2)

rf.fit(X_train,y_train)

print('Random Force Train Score :',rf.score(X_train,y_train))

print('Random Force Test Score :',rf.score(X_test,y_test))

print('Random Force Classifier Model feature importances are :\n ' , rf.feature_importances_)

print('----------------------------------------------------')
y_pred_rf=rf.predict(X_test)

print('Y Test: ',y_test)

print('Y Pred',y_pred_rf)

print(np.concatenate((y_pred_rf.reshape(len(y_pred_rf),1),y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import accuracy_score

accuracy_s=accuracy_score(y_test,y_pred_rf)

print('Accuracy Score',accuracy_s)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred_rf)

print(cm)

plt.figure(figsize=(9,5))

sns.heatmap(cm,center=True)

plt.show()
from sklearn.metrics import classification_report

cr=classification_report(y_test,y_pred_rf)

print(cr)
from sklearn.metrics import roc_curve,auc

rf_tpr,rf_fpr,threshold=roc_curve(y_test,y_pred_rf)

rf_auc=auc(rf_tpr,rf_fpr)

print('rf_tpr Value  : ', rf_tpr)

print('rf_fpr Value  : ', rf_fpr)

print('thresholds Value  : ', threshold)



#Draw ROC Curve && AUC [Area Under The Curve]



plt.figure(figsize=(5, 5), dpi=100)

plt.plot(rf_tpr, rf_fpr, linestyle='-', label='Random Force (auc = %0.3f)' % rf_auc)



plt.xlabel('True Positive Rate -->')

plt.ylabel('False Positive Rate -->')



plt.legend()

plt.show()
from sklearn.naive_bayes import BernoulliNB

NB=BernoulliNB()

NB.fit(X_train,y_train)

print('Naive Bayse Train Score',NB.score(X_train,y_train))

print('Naive Bayse Test Score',NB.score(X_test,y_test))



#Calculating Prediction

y_pred_NB = NB.predict(X_test)

y_pred_prob = NB.predict_proba(X_test)

y_pred_prob2=y_pred_prob.astype(int)

print('Y Test' ,y_test)

print('Predicted Value for BernoulliNBModel is : ' , y_pred_NB)

print('Prediction Probabilities Value for BernoulliNBModel is : \n' , y_pred_prob2)
from sklearn.metrics import accuracy_score

accuracy_s=accuracy_score(y_test,y_pred_NB)

print(accuracy_s)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred_NB)

print(cm)

plt.figure(figsize=(9,5))

sns.heatmap(cm,center=True)

plt.show()
from sklearn.metrics import classification_report

cr=classification_report(y_test,y_pred_NB)

print(cr)
from sklearn.metrics import roc_curve,auc

nb_tpr,nb_fpr,threshold=roc_curve(y_test,y_pred_NB)

nb_auc=auc(nb_tpr,nb_fpr)

print('nb_tpr Value  : ', nb_tpr)

print('nb_fpr Value  : ', nb_fpr)

print('thresholds Value  : ', threshold)



#Draw ROC Curve && AUC [Area Under The Curve]



plt.figure(figsize=(5, 5), dpi=100)

plt.plot(rf_tpr, rf_fpr, linestyle='-', label='Naive Basye (auc = %0.3f)' % nb_auc)



plt.xlabel('True Positive Rate -->')

plt.ylabel('False Positive Rate -->')



plt.legend()

plt.show()
from sklearn.metrics import roc_curve, auc



LR_tpr,LR_fpr,threshold=roc_curve(y_test,y_pred_LR)

LR_auc = auc(LR_tpr, LR_fpr)



svm_fpr, svm_tpr, threshold = roc_curve(y_test, y_pred_svm)

svm_auc = auc(svm_fpr, svm_tpr)



knn_fpr, knn_tpr, threshold = roc_curve(y_test, y_pred_knn)

knn_auc = auc(knn_fpr, knn_tpr)



dt_fpr, dt_tpr, threshold = roc_curve(y_test, y_pred_dt)

dt_auc = auc(dt_fpr, dt_tpr)



rf_fpr, rf_tpr, threshold = roc_curve(y_test, y_pred_rf)

rf_auc = auc(rf_fpr, rf_tpr)





nb_fpr, nb_tpr, threshold = roc_curve(y_test, y_pred_NB)

nb_auc = auc(nb_fpr, nb_tpr)



plt.figure(figsize=(9, 5))

plt.plot(LR_tpr, LR_fpr, marker='o', label='Logistic Regression (auc = %0.3f)' % LR_auc)

plt.plot(svm_fpr, svm_tpr, linestyle='--', label='SVM (auc = %0.3f)' % svm_auc)

plt.plot(knn_fpr, knn_tpr, linestyle=':', label='KNN (auc = %0.3f)' % knn_auc)

plt.plot(dt_fpr, dt_tpr, linestyle='-', label='DT (auc = %0.3f)' % dt_auc)

plt.plot(rf_fpr, rf_tpr, linestyle='-', label='RF (auc = %0.3f)' % rf_auc)

plt.plot(nb_fpr, nb_tpr, linestyle='--', label='NB (auc = %0.3f)' % nb_auc)







plt.xlabel('False Positive Rate -->')

plt.ylabel('True Positive Rate -->')



plt.legend()