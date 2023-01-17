import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
voices=pd.read_csv("../input/voicegender/voice.csv")
voices.head()
voices.info()
y=voices[['label']].values
y[0:5]
from sklearn import preprocessing
label = preprocessing.LabelEncoder()
label.fit(['male','female'])
y = label.transform(y) 
y[0:5]
x=voices.drop(['label'],axis=1)
x[0:5]
x=preprocessing.StandardScaler().fit(x).transform(x)
x[0:5]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=4)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR=LogisticRegression(C=0.01,solver='liblinear').fit(X_train,Y_train)
LR
Y_hat=LR.predict(X_test)
Y_hat
from sklearn.metrics import confusion_matrix,classification_report
cnf_matrix = confusion_matrix(Y_test, Y_hat, labels=[1,0])
np.set_printoptions(precision=2)
print(classification_report(Y_test,Y_hat))
from sklearn.metrics import log_loss
Y_hat_prob=LR.predict_proba(X_test)
log_loss(Y_test,Y_hat_prob)