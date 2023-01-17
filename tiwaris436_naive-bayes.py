

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv("../input/transfusion.data")



data.head()
data.shape
data.isnull().sum()
list(data)
import numpy as np

#defining a normalisation function 

def normalize (x): 

    return ( (x-np.min(x))/ (max(x) - min(x)))

                                            

                                              

# applying normalize ( ) to all columns 

data1 = data.apply(normalize) 
X =  data1.drop(['whether he/she donated blood in March 2007'],axis=1)

y = data1['whether he/she donated blood in March 2007']
from sklearn.model_selection import train_test_split

train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.3,random_state=42)
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model

model.fit(train_X,train_y)
pred_y=model.predict(test_X)
from sklearn import metrics

metrics.accuracy_score(test_y,pred_y)
metrics.confusion_matrix(test_y, pred_y)
metrics.recall_score(test_y, pred_y)
data1['whether he/she donated blood in March 2007'].value_counts()
from imblearn.over_sampling import SMOTE
smt = SMOTE()

X_train, y_train = smt.fit_sample(train_X, train_y)


#np.bincount(y_train)
model.fit(X_train,y_train)
y_pred=model.predict(test_X)
metrics.accuracy_score(test_y, y_pred)
metrics.recall_score(test_y, y_pred)
from imblearn.over_sampling import ADASYN 

sm = ADASYN()

X1, y1 = sm.fit_sample(train_X, train_y)

#np.bincount(y1)
model.fit(X1,y1)
y_pred1=model.predict(test_X)
metrics.accuracy_score(test_y, y_pred1)
metrics.recall_score(test_y, y_pred1)