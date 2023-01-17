

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import datasets

datadict = datasets.load_breast_cancer()
datadict.keys()
x = datadict['data']

y = datadict['target'] #  0  or 1
pd.DataFrame(x,columns=datadict['feature_names']).head()
from sklearn.model_selection import train_test_split

from sklearn import linear_model

#70% of our data to training and 30% for ttesting

x_train, x_test, y_train, y_test = train_test_split(x,y,

                                                   test_size = 0.3,

                                                   random_state = 42)
print ("Training Data is :", x_train.shape, y_train.shape )

print("Testing Data is :",x_test.shape, y_test.shape)
model = linear_model.LogisticRegression()

model.fit(x_train, y_train)
prediction = model.predict(x_test)
accuracy = np.mean(y_test == prediction)
accuracy