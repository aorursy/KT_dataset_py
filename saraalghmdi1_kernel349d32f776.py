import numpy as np
import pandas as pd
import sklearn

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression
datadict=datasets.load_breast_cancer()
datadict.keys()
x=datadict['data']

y=datadict['target']
pd.DataFrame(x, columns=datadict['feature_names']).head()
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.30, random_state=0)
print('Training data: ', x_train.shape, y_train.shape)

print('Training data: ', x_test.shape, y_test.shape)
model=linear_model.LogisticRegression()

model.fit(x_train, y_train)
prediction=model.predict(x_test)
accuracy=np.mean(y_test==prediction)
accuracy