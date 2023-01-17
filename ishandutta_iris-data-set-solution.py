import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
# Checking files in directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Column Headers
headers=['sepal length','sepal width','petal length','petal width','class']

# Read the data
df=pd.read_csv('/kaggle/input/iris-dataset/iris.data',names=headers)
df.head()
df.columns
df.describe()
# Label encoding the target variable

encode=LabelEncoder()
df['class']=encode.fit_transform(df['class'])
df.head()
# Selecting target and features

X=df.drop(['class'],axis=1)
y=df['class']
# Train Test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
# Making the model

lr=LogisticRegression()
lr.fit(X_train,y_train)
yhat=lr.predict(X_test)
# Evaluating Model

print('Predicted Values on Test Data',encode.inverse_transform(yhat))

print("Accuracy Score : ",accuracy_score(yhat,y_test))
