
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df=pd.read_csv('../input/marriage-status/marital status.csv')
df.head(5)
df.isnull().sum()
##Handling Nan using Median

median=df['status'].median()
df.status=df.status.fillna(median)
df.isnull().sum()
##Count Value (0 and 1)

df['status'].value_counts()
##Split the X and y 

x=df[['age']]
y=df['status']
##Take Data Size

from sklearn.model_selection import train_test_split

##Take Data Size

X_train, X_test, y_train, y_test = train_test_split(
                      x, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression

Lg=LogisticRegression()
Lg.fit(X_train,y_train)
Lg.predict(X_test)
X_test
df
Accuracy=Lg.score(X_test,y_test)
##Accuracy Print

print("The Accuracy is ", Accuracy)
