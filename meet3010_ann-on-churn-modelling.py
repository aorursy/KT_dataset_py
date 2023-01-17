import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
data = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")
data.head()
# check the shape of the data
data.shape
# independent variable
X=data.drop(['RowNumber','CustomerId','Surname','Exited'],axis=1)
#dependent variable
y=data['Exited']

# checking the shape of the data
X.shape, y.shape
X.head()
# import the encoder
from sklearn.preprocessing import LabelEncoder
# inputing the labelencoder into a variable 'label1'
label1 = LabelEncoder()
# transforming the geography column of the dataset
X['Geography']=label1.fit_transform(X['Geography'])
X.head()
# transforming the Gender column of the dataset
X['Gender']=label1.fit_transform(X['Gender'])
X.head()
X=pd.get_dummies(X,drop_first=True,columns=['Geography'])
X.head()
# dividing the data into training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=99,stratify=y)

# checking the shape of the training and testing data
X_train.shape,X_test.shape,y_train.shape,y_test.shape

X_train.head()
# importing the standard scaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
# scaling the training data
X_train=scaler.fit_transform(X_train)
# scaling the testing data
X_test=scaler.fit_transform(X_test)

# check
X_train
# import model
model=Sequential()
# import layers
model.add(Dense(X.shape[1],activation='relu',input_dim=X.shape[1]))
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
y_train=y_train.values
y_train
y_test=y_test.values
y_test
%%time
# fitting the model
history=model.fit(X_train,y_train,batch_size=10,epochs=20)
# prediction
y_pred=model.predict_classes(X_test)
y_pred
# check the accuracy
x_loss,x_acc=model.evaluate(X_test,y_test)
print(x_loss,x_acc)
# import the libraries
from sklearn.metrics import confusion_matrix,accuracy_score
# accuracy score
confusion_matrix(y_test,y_pred)
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.show()
