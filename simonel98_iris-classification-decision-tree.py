import numpy as np

import pandas as pd
df=pd.read_csv('../input/iris/Iris.csv')
df.head()
#checking if missing values are there

df.isnull().all()
#independent and dependent variable

X=df.iloc[:,:-1]

X
Y=df.iloc[:,5]

Y
#Encoded dependent variable

from sklearn.preprocessing import LabelEncoder

Label_Y=LabelEncoder()

Y=Label_Y.fit_transform(Y)
Y
##splitting train and test

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
#applying algorithm

from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)

classifier.fit(X_train,Y_train)
#predicting

Y_pred=classifier.predict(X_test)
Y_pred
#checking Accuracy

from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(Y_test,Y_pred)

print(cm)

accuracy=accuracy_score(Y_test,Y_pred)
accuracy