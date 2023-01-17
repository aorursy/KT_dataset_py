from sklearn import datasets
datadict=datasets.load_breast_cancer()
X=datadict['data']

Y=datadict['target']
import pandas as pd

X=pd.DataFrame(X,columns=datadict['feature_names'])

X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)
from sklearn import linear_model
model=linear_model.LogisticRegression()
model.fit(X_train,y_train)


prediction=model.predict(X_test)



import numpy as np

accuracy=np.mean(y_test==prediction)
accuracy