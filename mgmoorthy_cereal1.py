# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Visualisation

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,r2_score
df=pd.read_csv('/kaggle/input/80-cereals-nutrition-data-on-80-cereal-products/cereal.csv')
df.head(20)
df.shape
df.columns
df['mfr'].value_counts()
y=df['rating']
df.dtypes
cat=df.select_dtypes(include="object")

num=df.select_dtypes(include="number")
cat=pd.get_dummies(cat,dtype='int')

print(cat.head(20))

x=pd.concat([cat,num],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

print(x_train.shape, x_test.shape,y_train.shape,y_test.shape)
model=LinearRegression(normalize=True).fit(x_train,y_train)
predictions=model.predict(x_test)

print(predictions)
print("Mean Squared Error:",mean_squared_error(y_test,predictions))
print("R2 value:",r2_score(y_test,predictions))
model.intercept_
model.coef_
from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train, y_train)
from sklearn.metrics import confusion_matrix
cutoff = 0.7                              # decide on a cutoff limit

predictions_classes = np.zeros_like(predictions)    # initialise a matrix full with zeros

predictions_classes[predictions > cutoff] = 1       # add a 1 if the cutoff was breached
y_test_classes = np.zeros_like(predictions)

y_test_classes[y_test > cutoff] = 1



x_test_classes = np.zeros_like(predictions)

x_test_classes[y_test > cutoff] = 1
confusion_matrix(x_test_classes,y_test_classes)