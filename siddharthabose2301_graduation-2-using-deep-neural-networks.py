# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
data = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")
data.head(10)
data.drop("Serial No.",axis = 1,inplace = True)
data.describe()
data.isnull().sum()
data.columns
corr_mat = data.corr()
sns.heatmap(corr_mat)
sns.pairplot(data)
#Plotting the numerical data

sns.jointplot('GRE Score','Chance of Admit ',data,kind = 'reg')
sns.jointplot('TOEFL Score','Chance of Admit ',data,kind = 'reg')
sns.jointplot('CGPA','Chance of Admit ',data,kind = 'reg')
#Plotting the categorical data

sns.boxplot('University Rating','Chance of Admit ',data=data)
sns.boxplot('SOP','Chance of Admit ',data=data)
sns.boxplot('LOR ','Chance of Admit ',data=data)
sns.boxplot('Research','Chance of Admit ',data=data)
#Normalizing the data
data['GRE Score'] = data['GRE Score']/340
data['TOEFL Score'] = data['TOEFL Score']/120
data['CGPA'] = data['CGPA']/10
data['SOP'] = data['SOP']/5
data['University Rating'] = data['University Rating']/5
data['LOR '] = data['LOR ']/5


target_data = data[data.columns[data.columns != 'Chance of Admit ']].tail(100)
target_data.head()
import keras
from keras.models import Sequential
from keras.layers import Dense
pred = data[data.columns[data.columns != 'Chance of Admit ']]
tar = data['Chance of Admit ']
n_cols = pred.shape[1]
def regg():
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(data.shape[1]-1,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
y_test = data['Chance of Admit '].tail(100)
from sklearn.metrics import mean_squared_error
m = regg()
m.fit(pred,tar, validation_split=0.3, epochs=100, verbose=1)
yhat = m.predict(target_data)
yhat = m.predict(target_data)
print(mean_squared_error(y_test,yhat))
