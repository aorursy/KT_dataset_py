import numpy as np 
import pandas as pd 

from sklearn import datasets 
datadict= datasets.load_breast_cancer()
datadict.keys()
x= datadict['data']
y= datadict['target']
pd.DataFrame(x, columns=datadict['feature_names']).head()
from sklearn.model_selection import train_test_split
x_train, x_test , y_train , y_test = train_test_split(x,y , test_size=0.3 )

from sklearn import linear_model
model= linear_model.LogisticRegression()
model.fit(x_train, y_train)
predictions= model.predict(x_test)
accuracy= np.mean(y_test == predictions )
print ('The data is' ,round ((accuracy*100),2) ,'% accurate')