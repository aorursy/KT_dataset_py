# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv") # data is now a dataframe of the indian diabetes data 



data.shape #Show the number of rows and columns

data #print data (first and last 5 rows are shown)
InputDiab=data.iloc[0:,0:8] #chose all rows, but choose the columns until the index 7 (smaller than 8).

OutputDiab=data.iloc[0:,8:9]
print(InputDiab)

print(OutputDiab)
#import seaborn as sns

import matplotlib.pyplot as plt
corrmat=data.corr()
corrmat
CountDiab=len(data.loc[data['Outcome']==True])

CountNoDiab=len(data.loc[data['Outcome']==False])

print("Number of Women with Diabtes: ",CountDiab) 



print("Number of Woman without diabetes: ", CountNoDiab) 

print("Number of missing  values for 'DiabetesPedigreeFunction': ", len(data.loc[data['DiabetesPedigreeFunction']==0])) #genetic risc

print("Number of missing  values for 'BMI': ", len(data.loc[data['BMI']==0]))

print("Number of missing  values for 'Glucose': ", len(data.loc[data['Glucose']==0]))

print("Number of missing  values for 'Blood Pressure': ", len(data.loc[data['BloodPressure']==0])) #diastbloodpressure

print("Number of missing  values for 'Insulin': ", len(data.loc[data['Insulin']==0]))

print("Number of missing  values for 'Skin Thickness': ", len(data.loc[data['SkinThickness']==0]))

print("Number of missing  values for 'Age': ", len(data.loc[data['Age']==0]))
InputDiabWithoutPregnancies=InputDiab.iloc[0:,1:8]

print(InputDiabWithoutPregnancies)

InputDiabWithoutPregnancies=InputDiabWithoutPregnancies.mask(InputDiabWithoutPregnancies==0).fillna(InputDiabWithoutPregnancies.mean())

print(InputDiabWithoutPregnancies)
print(InputDiab['Pregnancies'])



InputDiab=pd.concat([InputDiab['Pregnancies'],InputDiabWithoutPregnancies],axis=1)

InputDiab.head(10)
from keras import layers               #comprises sequential networks,  but also more complex network

from keras.layers import Dense 

from keras import models

from keras.models import Sequential



MyNetwork=Sequential()

MyNetwork
num_inner=100 #mittlere Schicht 2 Neuronen

MyNetwork.add(Dense(num_inner,input_dim=8,activation='relu')) #Inputgröße 2 dim 

MyNetwork.add(Dense(32,activation='relu'))#Ausgabeschicht

MyNetwork.add(Dense(1,activation='sigmoid'))#Ausgabeschicht

MyNetwork.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
InputModel=InputDiab.iloc[0:500,0:] # Die ersten 330 Zeilen zum trainieren

#EingabeArray=Eingabe.values

#EingabeArray=EingabeArray.reshape(3,-1)



OutputModel=OutputDiab.iloc[0:500,0:]

#AusgabeArray=Ausgabe.values

#print(EingabeArray) 

print(InputModel)

print(OutputModel)



MyNetwork.fit(InputModel,OutputModel,epochs=10000,verbose=0)
InputPred=InputDiab.iloc[500:,0:]

print(InputPred)

MyNetwork.evaluate(InputPred,OutputDiab.iloc[500:,0:])

#MyNetwork.predict_classes(InputPred)