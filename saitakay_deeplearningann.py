# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/bank-customer-churn-modeling/Churn_Modelling.csv")
df
df.drop("CustomerId",axis="columns",inplace=True)
df.dtypes
df[df.Exited==0]
exited_no=df[df.Exited==0].Age
exited_yes=df[df.Exited==1].Age


plt.xlabel("Exited")
plt.ylabel("Number Of Customers")
plt.title("Bank Exited Prediction Visualiztion")

plt.hist([exited_yes,exited_no],rwidth=0.95, color=['green','red'],label=['Exited=Yes','Exited=No'])
plt.legend()

exited_no=df[df.Exited==0].Tenure
exited_yes=df[df.Exited==1].Tenure


plt.xlabel("Exited Tenure")
plt.ylabel("Number Of Customers")
plt.title("Bank Exited Prediction Visualiztion")

plt.hist([exited_yes,exited_no],rwidth=0.95, color=['green','red'],label=['Exited=Yes','Exited=No'])
plt.legend()

def print_unique_values(df):
    for column in df:
        if df[column].dtypes=="object":
            print(f"{column}:{df[column].unique()}")
print_unique_values(df)
df["Gender"].replace({"Female":0,"Male":1},inplace=True)
df
df2=pd.get_dummies(data=df,columns=["Geography","Surname"])
df2
df2.dtypes

cols_to_scale=["Tenure","EstimatedSalary","Balance","CreditScore"]
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
df2[cols_to_scale]=scaler.fit_transform(df[cols_to_scale])
df2
X=df2.drop(["Exited"],axis="columns")
y=df2["Exited"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)

X_train.shape
X_test.shape
y_train.shape
y_test.shape
X_train[:10]
len(X_train.columns)
import tensorflow
from tensorflow import keras

model=keras.Sequential([
    keras.layers.Dense(2945,input_shape=(2945,),activation="relu"),
    keras.layers.Dense(100,activation="relu"),
    keras.layers.Dense(1,activation="sigmoid")
])
model.compile(optimizer = "adam" ,
              loss = "binary_crossentropy" ,
              metrics = ["accuracy"]
    
)
model.fit(X_train,y_train,epochs=20)
df2



