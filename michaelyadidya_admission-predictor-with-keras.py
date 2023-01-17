# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from keras.models import Sequential 
from keras.layers import Dense
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input/Admission_Predict_Ver1.1.csv"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df.head()
df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})
df.info()
fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()
print("Not Having Research:",len(df[df.Research == 0]))
print("Having Research:",len(df[df.Research == 1]))
y = np.array([df["TOEFL Score"].min(),df["TOEFL Score"].mean(),df["TOEFL Score"].max()])
x = ["Worst","Average","Best"]
plt.bar(x,y)
plt.title("TOEFL Scores")
plt.xlabel("Level")
plt.ylabel("TOEFL Score")
plt.show()
df["GRE Score"].plot(kind = 'hist',bins = 200,figsize = (6,6))
plt.title("GRE Scores")
plt.xlabel("GRE Score")
plt.ylabel("Frequency")
plt.show()
plt.scatter(df["University Rating"],df.CGPA)
plt.title("CGPA Scores for University")
plt.xlabel("University Rating")
plt.ylabel("CGPA")
plt.show()
df = df.drop(['Serial No.'],1)
df.describe()
x_data = df.drop(['Chance of Admit'],axis = 1)

y = df['Chance of Admit'].values
x_data = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
x_train,x_test,y_train,y_test = train_test_split(x_data,y,test_size = 0.20,random_state = 42)
model = Sequential()
model.add(Dense(12,input_dim = 7, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(8,activation = 'relu'))
model.add(Dense(1,activation = 'linear'))
model.summary()
model.compile(loss = 'mse', optimizer= 'adam', metrics = ['mse', 'mae'])
history = model.fit(x_train, y_train, epochs = 65, batch_size = 50, verbose = 1, validation_split = 0.2)
y_pred = model.predict(x_test)
plt.plot(y_test)
plt.plot(y_pred)
plt.title('Prediction')

