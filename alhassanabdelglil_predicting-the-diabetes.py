import numpy as np
import pandas as pd
dataset = pd.read_csv('../input/diabetes.csv')
dataset.info()
dataset.describe()
dataset.isnull().sum()
dataset.head()
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,[-1]]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2,random_state = 0)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 6, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm
y=pd.DataFrame(y_pred)
y.to_csv('out.csv',index=False,header=False)
