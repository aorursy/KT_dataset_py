# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

dataset = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv', header=0)
dataset.head()
X=dataset.drop('Outcome',axis=1)

y=dataset.iloc[:,8]
X.describe()
X.dtypes
X.iloc[:,5]=X.iloc[:,5].astype('int64')

X.iloc[:,6]=X.iloc[:,6].astype('int64')

X.isnull().sum()
import seaborn as sea

heatmap=sea.heatmap(dataset.corr(),annot=True)

plt.show()


#checking outliers

import seaborn as sns

fig,ax = plt.subplots(figsize=(12,12))

sns.boxplot(data =X , ax = ax )

print(X['Age'].quantile(.10)) 

print(X['Age'].quantile(.90)) 

print(X['Pregnancies'].quantile(.90))

print(X['Pregnancies'].quantile(.10))

print(X['Glucose'].quantile(.90))

print(X['Glucose'].quantile(.10))

print(X['BloodPressure'].quantile(.90))

print(X['BloodPressure'].quantile(.10))

print(X['SkinThickness'].quantile(.90))

print(X['SkinThickness'].quantile(.10))

print(X['Insulin'].quantile(.90))

print(X['Insulin'].quantile(.10))

print(X['BMI'].quantile(.90))

print(X['BMI'].quantile(.10))



      

      

      
X['Age']=np.where(X['Age']<22,22,X['Age'])

X['Age']=np.where(X['Age']>51,51,X['Age'])

X['Pregnancies']=np.where(X['Pregnancies']>9,9,X['Pregnancies'])

X['Pregnancies']=np.where(X['Pregnancies']<0,0,X['Pregnancies'])

X['Glucose']=np.where(X['Glucose']>167,167,X['Glucose'])

X['Glucose']=np.where(X['Glucose']<85,85,X['Glucose'])

X['BloodPressure']=np.where(X['BloodPressure']>88,88,X['BloodPressure'])

X['BloodPressure']=np.where(X['BloodPressure']<54,54,X['BloodPressure'])

X['SkinThickness']=np.where(X['SkinThickness']>40,40,X['SkinThickness'])

X['SkinThickness']=np.where(X['SkinThickness']<0,0,X['SkinThickness'])

X['Insulin']=np.where(X['Insulin']>210,210,X['Insulin'])

X['Insulin']=np.where(X['Insulin']<0,0,X['Insulin'])

X['BMI']=np.where(X['BMI']>41,41,X['BMI'])

X['BMI']=np.where(X['BMI']<23,23,X['BMI'])

fig,ax = plt.subplots(figsize=(12,12))

sns.boxplot(data =X , ax = ax )



#scaling the data

from sklearn.preprocessing import StandardScaler

stand=StandardScaler()

X_scl=stand.fit_transform(X)



#checking multicollinearity 

from statsmodels.stats.outliers_influence import variance_inflation_factor

X_new=pd.DataFrame(X_scl)

print(type(X_new))

VIF=pd.DataFrame()

VIF['Var Name']=X.columns

VIF['vif values']=[variance_inflation_factor(X_new.values,i) for i in range(X_new.shape[1])]

print(VIF)
#selecting best features

from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test=  train_test_split(X,y,test_size=0.33,random_state=1)

from sklearn.feature_selection import SelectKBest , f_classif

fs=SelectKBest(score_func=f_classif,k=6)

fs.fit(X_train,y_train)

X_new_train=fs.transform(X_train)

X_new_test=fs.transform(X_test)

for i in range(len(fs.scores_)):

    print(i," ", fs.scores_[i])

plt.bar([i for i in range(len(fs.scores_))],fs.scores_)

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, roc_auc_score,plot_roc_curve, confusion_matrix

log=LogisticRegression()

log.fit(X_new_train,y_train)

print("log  with train ",log.score(X_new_train,y_train))

print("log  with test ",log.score(X_new_test,y_test))



from sklearn.svm import SVC

svc= SVC(kernel='rbf')

svc.fit(X_new_train,y_train)

print("svc with train ",svc.score(X_new_train,y_train))

print("svc with test ",svc.score(X_new_test,y_test))





y_pred_l=log.predict(X_new_test)

lr_probs_l=log.predict_proba(X_new_test)



auc_l = roc_auc_score(y_test, lr_probs_l[:,1])





fpr_l, tpr_l, thr_l = roc_curve(y_test, lr_probs_l[:,1])

plt.subplots(1, figsize=(10,10))

plt.title('Receiver Operating Characteristic - DecisionTree')

plt.plot(fpr_l, tpr_l,"-b",label="log, auc="+str(auc_l))

plt.plot([0, 1], ls="--")

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend(loc='best')

plt.show()

                                

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, roc_auc_score,plot_roc_curve, confusion_matrix

log=LogisticRegression()

log.fit(X_new_train,y_train)

print("log  with train ",log.score(X_new_train,y_train))

print("log  with test ",log.score(X_new_test,y_test))



from sklearn.svm import SVC

svc= SVC(kernel='rbf')

svc.fit(X_new_train,y_train)

print("svc with train ",svc.score(X_new_train,y_train))

print("svc with test ",svc.score(X_new_test,y_test))





y_pred_l=log.predict(X_new_test)



y_pred_s=svc.predict(X_new_test)





auc_l = roc_auc_score(y_test,y_pred_l)

auc_s = roc_auc_score(y_test,y_pred_s)

fpr_l, tpr_l, thr_l = roc_curve(y_test, y_pred_l)

fpr_s, tpr_s, thr_s = roc_curve(y_test, y_pred_s)

plt.subplots(1, figsize=(10,10))

plt.title('Receiver Operating Characteristic - DecisionTree')

plt.plot(fpr_l, tpr_l,"-b",label="log, auc="+str(auc_l))

plt.plot(fpr_s, tpr_s,"-r",label="svc, auc="+str(auc_s))

plt.plot([0, 1], ls="--")

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend(loc='best')

plt.show()

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(log, X_new_test,y_test)

plt.show()
tn, fp, fn, tp = confusion_matrix(y_test,y_pred_l).ravel()

print(confusion_matrix(y_test,y_pred_l))

print(tn, fp, fn, tp)

accuracy = (tp+tn)/(tp+tn+fp+fn)

print("test accuracy ", accuracy)
