import pandas as pd
data=pd.read_csv('../input/ahdd1/csvTrainImages 60k x 784.csv')
data.head()
data.shape
y=pd.read_csv('../input/ahdd1/csvTrainLabel 60k x 1.csv')
y
import matplotlib.pyplot as plt
import numpy as np
a=data.iloc[4,:].values
a=a.reshape(28,28)
plt.imshow(a)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=4)
rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
rf.score(x_train,y_train)
x_train.shape
x_test.shape
test_pred=rf.predict(x_test)
test_pred[0:5]
y_test[0:5]
accuracy_score(y_test,test_pred)
test=pd.read_csv('../input/ahdd1/csvTestImages 10k x 784.csv')
test.head()
test_labels=pd.read_csv('../input/ahdd1/csvTestLabel 10k x 1.csv')
test_labels
pred=rf.predict(test)
pred
s=test_labels.values
count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count=count+1
count
9828/10000
from sklearn.metrics import accuracy_score
accuracy_score(test_labels,pred)
