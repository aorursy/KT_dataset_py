import pandas as pd
from sklearn import datasets
datadict = datasets.load_breast_cancer()
datadict.keys()
#print(datadict['DESCR'])
X = datadict['data']
y = datadict['target']
pd.DataFrame (X, columns=datadict['feature_names']).head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.3, random_state=42)
print('Training data: ', X_train.shape, y_train.shape)
print('Testing data: ', X_test.shape, y_test.shape)
from sklearn import linear_model
model = linear_model.LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
import numpy as np
accuracy = np.mean(y_test == predictions)
accuracy