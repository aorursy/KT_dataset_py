
import pandas as pd


df = pd.read_csv('../input/iris/Iris.csv')
df.head()
df.describe() 
# gives statistics of the dataset
df.info()  
# columns information
X=df.drop(['Species'], axis=1)
X.head() # default : gives top 5 rows
y=df["Species"]
y.head()
# split the dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=.25, random_state=15)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import LogisticRegression

model= LogisticRegression()

model.fit(X_train, y_train)  # training the model
y_pred= model.predict(X_test)   
# y_pred (predicted values) for a given X_test values
print(X_test)
print(y_test)
print(y_pred)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


import pandas as pd
df =pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df
model.score(X_test, y_test) 
# it gives accuracy score for our model
#accuracy score is good for our dataset