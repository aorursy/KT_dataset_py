!pip install regressors
import numpy as np

import pandas as pd 



from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from patsy import dmatrices



import os

print(os.listdir("../input"))
df = pd.read_csv('../input/cardio_train.csv',sep=';')

df = df.dropna()

df = df.drop(columns=['id'])

df.describe()
# create dummy variables, and their interactions

y, X = dmatrices('cardio ~ age*cholesterol*weight*ap_hi', df, return_type="dataframe")

# flatten y into a 1-D array so scikit-learn can understand it

y = np.ravel(y)

X.head()
logisticRegr = LogisticRegression()

logisticRegr.fit(X, y)
x1 = {'age': [55*365], 'cholesterol': [2], 'weight': [80], 'ap_hi': 145, 'cardio': 0}

XTest1 = pd.DataFrame(data=x1)

y1, XTest1 = dmatrices('cardio ~ age*cholesterol*weight*ap_hi', XTest1, return_type="dataframe")

y = np.ravel(y1)



ypred1 = logisticRegr.predict(XTest1)

if(ypred1[0]): 

    print("This patient has a strong possibility of cardiovascular disease.")

else:

    print("This patient does not show signs of cardiovascular disease.")
x2 = {'age': [35*365], 'cholesterol': [1], 'weight': [70], 'ap_hi': 128, 'cardio': 0}

XTest2 = pd.DataFrame(data=x2)

y2, XTest2 = dmatrices('cardio ~ age*cholesterol*weight*ap_hi', XTest2, return_type="dataframe")

y = np.ravel(y2)



ypred2 = logisticRegr.predict(XTest2)

if(ypred2[0]): 

    print("This patient has a strong possibility of cardiovascular disease.")

else:

    print("This patient does not show signs of cardiovascular disease.")