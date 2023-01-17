import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
iris_data = pd.read_csv('/kaggle/input/iris-dataset/Iris.csv')
print(iris_data.shape)
print(iris_data.head())
iris_data.drop('Id',inplace=True,axis=1)
print(iris_data.head())
print(iris_data['Species'].value_counts())
iris_data.isnull().sum()
iris_data_shuffled = iris_data.sample(frac=1).reset_index(drop=True)
print(iris_data_shuffled)
#drop the Species column. Note that we are not doing it inplace so the original dataframe will not be modified
train_x = iris_data_shuffled.drop('Species',axis=1)
print(train_x.head())
train_y = iris_data_shuffled['Species']
print(train_y.head())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(train_y)
print(labels)
X = train_x.values
y = labels
print(X.shape)
print(y.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)
print('no. of training samples:',len(X_train))
print('no. of testing samples:',len(X_test))
lr_model = LogisticRegression(multi_class='multinomial')
lr_model.fit(X_train,y_train)
y_pred = lr_model.predict(X_test)
print(y_pred)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print('test accuracy:',accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
