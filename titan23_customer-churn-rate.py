import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df=pd.read_csv("../input/ChurnData.csv")
df.info()

df.head()
df.shape
X_data=df[["tenure","age","address","income","ed","equip","callcard","wireless"]]
X_data.head()
X_data.corr()
plt.hist(X_data.corr())
Y_data=df["churn"]
Y_data.head()
XA=np.asanyarray(X_data)

YA=np.asanyarray(Y_data)
XA.dtype
XA[:3]
from sklearn.preprocessing import StandardScaler

XA=StandardScaler().fit(XA).transform(XA)
print(XA)
m=XA.mean() # calculating mean

std=XA.std()#calculating standard deviation

print("mean of XA {} \nstd of XA {}".format(round(m),std))

X_train=XA[:170]

X_test=XA[170:]



Y_train=YA[:170]

Y_test=YA[170:]
print(X_train)
from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(solver='liblinear')

mdl=clf.fit(X_train,Y_train)
Yp=mdl.predict(X_test)

YA=Y_test

print(Yp)
sns.boxplot(Y_test,Yp)
Yp_prob=mdl.predict_proba(X_test)

print(Yp_prob)
P_Y1X=Yp_prob[:,0] # give the array of truth

P_Y0X=Yp_prob[:,1] # give the array of false
table=pd.DataFrame({"P(Y=1|X)":P_Y1X,"P(Y=0|X)":P_Y0X})

print(table)
from sklearn.metrics import accuracy_score , jaccard_similarity_score,confusion_matrix,roc_curve

jss=jaccard_similarity_score(YA,Yp)

acc=accuracy_score(YA,Yp)

print("jaccard is = {} \naccuracy_score  = {}".format(jss,acc))
cm=confusion_matrix(YA,Yp)

print("Confusion matrix \n",cm)
from sklearn.metrics import classification_report

print(classification_report(YA,Yp))
plt.matshow(cm)

plt.title("Confusion matrix")

plt.xlabel("Predicted")

plt.ylabel("Real")
my_con=pd.crosstab(YA,Yp)

print(my_con)
import seaborn as sns

sns.heatmap(my_con)
table1=pd.DataFrame({"Ya":YA,"Yp":Yp})

print(table1)
import seaborn as sns

conmat=pd.crosstab(tables.Ya,tables.Yp,margins=True)

sns.heatmap(conmat,annot=True)

plt.show()
from sklearn.metrics import roc_curve

Yppr=Yp_prob[:,1]

fpr,tpr,thr=roc_curve(YA,Yppr)
print(fpr)
print(tpr)
print(thr)
roc=roc_curve(YA,Yp)

print(roc)
plt.plot(fpr,tpr,label="ROC CURVE")

plt.grid()