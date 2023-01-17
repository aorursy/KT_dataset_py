# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/fake-data/fake_reg.csv")
df.head()
sns.pairplot(df)
from sklearn.model_selection import train_test_split
X=df[["feature1","feature2"]].values
y=df["price"].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
X_train.shape
y_train.shape
X_train
X_test
from sklearn.preprocessing import MinMaxScaler
help(MinMaxScaler)
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
X_train
X_test
X_train.min()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
help(Dense)
#One way of declaring model

model=Sequential([Dense(4,activation="relu"),
                 Dense(2,activation="relu"),
                 Dense(1) 
                 ])
#second way of decalring model

model=Sequential()

model.add(Dense(4,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(4,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="rmsprop",loss="mse")
model.fit(x=X_train,y=y_train,epochs=250)
loss_df=pd.DataFrame(model.history.history)
loss_df.plot()

#Time to evaluate model
model.evaluate(X_test,y_test,verbose=0)
test_predictions=model.predict(X_test)
test_predictions
test_predictions=pd.Series(test_predictions.reshape(300,))
test_predictions
pred_df=pd.DataFrame(y_test,columns=["Test Ture Y"])
pred_df=pd.concat([pred_df,test_predictions],axis=1)
pred_df
pred_df.columns=["Test True Y","Predictions"]
pred_df
sns.scatterplot(x="Test True Y",y="Predictions",data=pred_df)
from sklearn.metrics import mean_absolute_error,mean_squared_error
mean_absolute_error(pred_df["Test True Y"],pred_df["Predictions"])
df.describe()
#let's try any value

new_gem=[[998,100]]
new_gem=scaler.transform(new_gem)
model.predict(new_gem)
from tensorflow.keras.models import load_model
#Saving the model to use later on!
model.save("my_fakedata_model.h5")
#Loading the saved model
later_model=load_model("my_fakedata_model.h5")
later_model.predict(new_gem)

