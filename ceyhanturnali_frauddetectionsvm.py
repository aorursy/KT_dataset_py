import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


cc=pd.read_csv('../input/creditcardfraud/creditcard.csv')
cc.head(10)
fraudCount = cc[cc['Class'] == 1] 
plt.figure(figsize=(15,10))
plt.scatter(fraudCount['Time'], fraudCount['Amount']) 
plt.xlabel('Time')
plt.ylabel('Amount')
plt.show()
FraudNum=len(cc[cc.Class==1])
print(FraudNum)
UnFraudNum=len(cc[cc.Class==0])
print(UnFraudNum)
fraudData=cc[cc.Class==1]
equalsizeUnFraud=cc[cc.Class==0]
equalsizeUnFraud=equalsizeUnFraud.sample(500)


fraudData.head(5)
equalsizeUnFraud.count()
equalsizeUnFraud.head(5)
new=equalsizeUnFraud.append(fraudData)
new=new.sample(frac=1)
new.head(5)

X=new.drop(['Class'],axis=1)
X=np.asarray(X)
X[0:5]
from sklearn import preprocessing
X=preprocessing.StandardScaler().fit(X).transform(X)

y=new['Class']
y=np.asarray(y)
y[0:5]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
from sklearn import svm
model=svm.SVC(kernel='linear',probability=True)
model.fit(X_train,y_train)
yhat=model.predict(X_test)
yhat[0:5]
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,yhat,labels=[1,0])
np.set_printoptions(precision=2)
print(classification_report(y_test,yhat))
confusion = pd.DataFrame(cm, index=['is Fraud', 'is Normal'],columns=['predicted fraud','predicted normal'])
confusion