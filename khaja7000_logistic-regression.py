import pandas as pd

import numpy as np 
Data=pd.DataFrame({'BMI':[75,126,63,154,187,106,110],'AGE':[38,52,40,45,55,30,50],'OUT':[0,1,0,1,1,0,0]})
Data
B1=np.cov(Data['BMI'],Data['OUT'],ddof=1)/np.var(Data['BMI'],ddof=1)

B1=B1[1:,0]

B1
B2=np.cov(Data['AGE'],Data['OUT'],ddof=1)/np.var(Data['AGE'],ddof=1)

B2=B2[1:,0]

B2
B0=np.mean(Data['OUT'])-(B1*np.mean(Data['BMI']))-(B2*np.mean(Data['AGE']))

B0
x=np.arange(-9,14)

x
y=1/(1+np.exp(-x))

y
import matplotlib.pyplot as plt
plt.plot(x,y)
k=B0+B1*(Data['BMI'])+B2*(Data['AGE'])
y=1/(1+np.exp(-k))
y
Data=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
Data.head()
x=Data.drop('Outcome',axis=1)

y=Data[['Outcome']]
import seaborn as sns
sns.heatmap(Data.corr(),annot=True)
Data.Outcome.value_counts()
G=Data.groupby(['Outcome'])
g0=G.get_group(0)

g0.shape
g1=G.get_group(1)

g1.shape
Data.columns
x1=g0['Outcome']

x2=g1['Outcome']
from scipy.stats import ttest_ind
ttest_ind(x1,x2)
x=Data[['Pregnancies','Glucose','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y=Data['Outcome']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=2)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
from sklearn import metrics
cm=metrics.confusion_matrix(Y_test,y_pred) # standerd  Y_test,y_pred
cm
#cm=metrics.confusion_matrix(y_pred,Y_test)
TPR=31/(45+31)

TPR
FRP=19/(136+19)

FRP
#sensitivity

TPR=(cm[1,1]/cm[1,].sum())*100
TPR
TNR=(cm[0,0]/cm[0,].sum())*100

TNR
from sklearn.metrics import roc_curve,auc
fpr,tpr,_=roc_curve(Y_test,y_pred)

roc_auc=auc(fpr,tpr)

print(roc_auc)

plt.figure()

plt.plot(fpr,tpr)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.0])

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.title('Receiver Operating Charecterstic')

plt.show()
Data1=pd.DataFrame({'BMI':[7.5,12.6,63,15.4,18.7,10.6,11.0],'GENDER':['male','female','male','female','female','male','male'],'AGE':[38,52,40,45,55,30,50],'OUT':[0,1,0,1,1,0,0]})
Data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(Data1['GENDER'])

Data1['GENDER']=le.transform(Data1['GENDER'])
Data1
Data.shape
x.shape
from sklearn.model_selection import KFold
kf=KFold (n_splits=5, shuffle=True, random_state=2)

acc=[]

au=[]

for train,test in kf.split(x,y):

    M=LogisticRegression()

    X_train,X_test=x.iloc[train,:],x.iloc[test,:]

    Y_train,Y_test=y[train],y[test]

    M.fit(X_train,Y_train)

    Y_pred=M.predict(X_test)

    acc.append(metrics.accuracy_score(Y_test,Y_pred))

    fpr,tpr,_=roc_curve(Y_test,Y_pred)

    au.append(auc(fpr,tpr))

print('cross VALIDATE AUC mean score %.2f%%'%np.mean(au))

print('cross VALIDATE AUC Variance score %.6f%%'%np.var(au,ddof=1))