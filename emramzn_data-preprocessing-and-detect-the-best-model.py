
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import warnings
from sklearn import metrics
warnings.filterwarnings("ignore", category=FutureWarning)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df=pd.read_csv("/kaggle/input/bank-marketing/bank-additional-full.csv", sep=";")
df.head()
df.info()

sns.countplot(x='education', data= df)
sns.despine()
print(df['previous'].value_counts())
print(" Job title : ", df['job'].unique())

# Education state
fig, ax=plt.subplots()
fig.set_size_inches(10,8)
sns.countplot(x='education',data=df)
ax.set_xlabel('Education', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title("Education State", fontsize=15)
sns.despine()
# marital
print("\nMarital")
fig, ax=plt.subplots()
fig.set_size_inches(10,8)
sns.countplot(x='marital', data=df)
ax.set_xlabel("marital", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("Marital State", fontsize=15)
sns.despine()

# Encoding 

from sklearn.preprocessing import  LabelEncoder

Encoder=LabelEncoder()

df['job']=Encoder.fit_transform(df['job'])
#df['marital']=Encoder.fit_transform(df['marital'])
df['education']=Encoder.fit_transform(df['education'])
df['default']=Encoder.fit_transform(df['default'])
df['housing']=Encoder.fit_transform(df['housing'])
df['loan']=Encoder.fit_transform(df['loan'])
df['month']=Encoder.fit_transform(df['month'])
df['contact']=Encoder.fit_transform(df['contact'])
df['day_of_week']=Encoder.fit_transform(df["day_of_week"])
df['poutcome']=Encoder.fit_transform(df['poutcome'])

# transform to binary the target attribute
df['y']=Encoder.fit_transform(df['y'])

# setting value of marital

df['marital'].replace(['married', 'single', 'divorced','unknown'],[1,2,3,4], inplace=True)
df.loc[df['age'] <26, 'age'] = 1
df.loc[(df['age'] >25) & (df['age']< 49 ),'age']=2
df.loc[(df['age']>48)&(df['age']<71), 'age']=3
df.loc[(df['age'] >70)&(df['age']<98), 'age']=4

fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1, figsize = (13, 5))

sns.boxplot(x=df['education'],ax=ax2)
ax2.set_xlabel("education", fontsize=15)
sns.despine(ax=ax2)
ax1.tick_params(labelsize=10)

sns.distplot(df['duration'], ax = ax1)
sns.despine(ax = ax1)
print("Max : {} and Min duration  : {} " .format(max(df['duration']), min(df['duration'])))

# Setting value of duration
df.loc[df['duration']<120 , 'duration'] = 1 
df.loc[(df['duration'] > 119)&( df['duration'] <= 200) , 'duration'] =2
df.loc[(df['duration'] >200)&( df['duration'] <=350), 'duration']=3
df.loc[(df['duration'] >350)&( df['duration']<=550), 'duration']=4
df.loc[df['duration'] > 550, 'duration']=5

# Target Attribute
y=df.iloc[:,df.columns=='y']
x=df.iloc[:,df.columns!='y']

df.head()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=0)

# proccess of Standardize
from sklearn.preprocessing import StandardScaler
S_Scaler=StandardScaler()
x_train=S_Scaler.fit_transform(x_train)
x_test=S_Scaler.fit_transform(x_test)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#grid={'n_neighbors': np.arange(1,20,1)}
KnnClass=KNeighborsClassifier(n_neighbors=9)
#KnnCV=GridSearchCV(KnnClas, grid, cv=10)
KnnClass.fit(x_train,np.ravel(y_train,order='C'))
#print("Best Parameters : {}\nBest Score {} ".format(KnnCV.best_params_, KnnCV.best_score_) )
#%% Xgboosting model 

Xgboost=XGBClassifier(learning_rate=0.1,max_depth=4, n_estimators=100,verbosity=1)
Xgboost.fit(x_train,np.ravel(y_train,order='C'))
y_pred=Xgboost.predict(x_test)
print("Test accuracy with XGBoos: ",accuracy_score(y_test,y_pred))

grid={'C':[0.0001,0.001,0.01,1] ,'gamma':['auto','scale'],'kernel':['rbf','linear','sigmoid'] , 'max_iter':[10,100]}
SVCModel=SVC(probability=True)
SVCGCV=GridSearchCV(SVCModel,grid,cv=10)
SVCGCV.fit(x_train,np.ravel(y_train, order='C'))
print("Best Params {} and best score {}".format(SVCGCV.best_params_, SVCGCV.best_score_))
#print(SVCModel.score(x_test,np.ravel(y_test, order='C')))

RForest=RandomForestClassifier(n_estimators=20)
RForest.fit(x_train,np.ravel(y_train,order='C'))
RFPred=RForest.predict(x_test)
print("Random Forest Accuracy : " ,accuracy_score(y_test,RFPred))
print("Cross Valudate Score : ", cross_val_score(RForest,x_test,y_test.values.ravel()))
print("Confisuon matrix :", confusion_matrix(y_test,RFPred.ravel()))
LRegression=LogisticRegression()
LRegression.fit(x_train,y_train.values.ravel())
y_predLR=LRegression.predict(x_test)
print("Logistic Regression Accuracy :", accuracy_score(y_test,y_predLR.ravel()))
print("Cross val score :", cross_val_score(LRegression,x_train,y_train.values.ravel(),cv=10,n_jobs=1,scoring='accuracy').mean())

BayesModel=GaussianNB()
BayesModel.fit(x_train,y_train.values.ravel())
bayesPred=BayesModel.predict(x_test)
print("Bayes Model Accuracy :", accuracy_score(y_test,bayesPred.ravel()))
print("Cross val score :", cross_val_score(BayesModel,x_train,y_train.values.ravel(),cv=10,n_jobs=2,scoring='accuracy').mean())

fig, ax_Array = plt.subplots(nrows = 1,  figsize = (8,6))

# bayes roc
probs = BayesModel.predict_proba(x_test)
preds = probs[:,1]
fprbayes, tprxbayes, thresholdbayes = metrics.roc_curve(y_test, preds)
roc_aucbayes = metrics.auc(fprbayes, tprxbayes)

# LR roc

probs=LRegression.predict_proba(x_test)
predicts=probs[:,1]
fprLR,tprLR, thresholdLR=metrics.roc_curve(y_test,predicts)
roc_aucLR=metrics.auc(fprLR,tprLR)

# KNN roc
probs=KnnClass.predict_proba(x_test)
predKnn=probs[:,1]
fprKnn, tprKnn, thresholdKnn=metrics.roc_curve(y_test,predKnn)
roc_aucknn=metrics.auc(fprKnn,tprKnn)


# Random Forest 
probs=RForest.predict_proba(x_test)
pred_RForest=probs[:,1]
fprRF,tprRF,thresholfRF= metrics.roc_curve(y_test,pred_RForest)
roc_aucRF=metrics.auc(fprRF,tprRF)

 # SVM model roc
 
prob=SVCGCV.predict_proba(x_test)
pred_Svm=prob[:,1]
fprSvm,tprsvm,tresholdSvm=metrics.roc_curve(y_test,pred_Svm)
roc_aucSvm=metrics.auc(fprSvm,tprsvm) 




ax_Array.plot(fprSvm,tprsvm, 'b', label='SMV Auc %0.2f' %roc_aucSvm, color="green")
ax_Array.plot(fprRF,tprRF,'b', label="RF Auc %0.2f"%roc_aucRF, color="blue")
ax_Array.plot(fprKnn,tprKnn,'b', label="Knn Auc %0.2f" %roc_aucknn, color="red")
ax_Array.plot(fprLR,tprLR, 'b', label='LR Auc = %0.2f' % roc_aucLR, color='pink')
ax_Array.plot(fprbayes,tprxbayes,'b', label='Bayes Auc %0.2f' % roc_aucbayes, color="black")
ax_Array.set_title('Receiver Operating Characteristic LR ',fontsize=10)
ax_Array.set_ylabel('True Positive Rate',fontsize=20)
ax_Array.set_xlabel('False Positive Rate',fontsize=15)
ax_Array.legend(loc = 'lower right', prop={'size': 10})



plt.subplots_adjust(wspace=1)




