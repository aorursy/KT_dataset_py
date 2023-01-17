#Importing the necessary packages  

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from graphviz import *
#Reading the data
data = pd.read_csv('../input/train.csv')
#Splitting into X and y as per the required for Scikit learn packages
X, y = data.iloc[:,:-1], data.iloc[:,-1]

#Splitting the dataset into training and testing
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)

#Using the XGBoost Classifier. I have used just a few combinations here and there without GridSearch or RandomSearch because the dataset was pretty small
xg_cl = xgb.XGBClassifier(objective='multi:softmax', n_estimators=200,seed=123,learning_rate=0.15,max_depth=5,colsample_bytree=1,subsample=1)
#fitting the model
xg_cl.fit(X_train,y_train)
preds = xg_cl.predict(X_test)
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))
#XGBoost in the package itself gives us the feature importance to understand how each features compares
xgb.plot_importance(xg_cl)
plt.show()
#trying on the test data
testdata =pd.read_csv('../input/test.csv')
#Since there is an id column we have to drop it so it is not used for the final predcition
testdata=testdata.drop('id',axis=1)
testdata.head()
#test prediction
test_prediction=xg_cl.predict(testdata)
#putting the prediction in a column 
testdata['price_range']= test_prediction
testdata.head()
