

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
x=df.iloc[:,:8].values
y=df.iloc[:,8].values
df.isnull().sum()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.impute import SimpleImputer
impute=SimpleImputer(missing_values=0, strategy='most_frequent')
x_train=impute.fit_transform(x_train)
x_test=impute.fit_transform(x_test)


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=42)
classifier.fit(x_train,y_train)
y_predict= classifier.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score
con=confusion_matrix(y_test,y_predict)
acc_score=accuracy_score(y_test,y_predict)
print(con)
print(acc_score)
