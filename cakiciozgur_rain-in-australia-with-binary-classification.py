import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.preprocessing import LabelEncoder 

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/weatherAUS.csv")
data.head()
data.info()
def missing_values(data): 
    miss_value_counts = data.isnull().sum()
    missing_value_percantage= 100 * data.isnull().sum()/len(data)
    missing_value_table = pd.concat([miss_value_counts, missing_value_percantage], axis=1)
    new_missing_value_table = missing_value_table.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return new_missing_value_table
missing_values(data)
data.drop(["Location","Date","Evaporation","Sunshine","Cloud9am","Cloud3pm"],axis=1,inplace=True)
data.fillna(data.mean())
data.RainToday=[1 if each=="Yes" else 0 for each in data.RainToday]
data.RainTomorrow=[1 if each=="Yes" else 0 for each in data.RainTomorrow]
data.RainToday=data.RainToday.astype("int")
data.RainTomorrow=data.RainTomorrow.astype("int")
data.dropna(inplace=True)
data.isnull().sum()
y=data.RainTomorrow.values
#data.RainTomorrow.unique()
x_data=data.drop(["RainTomorrow"],axis=1)
labelencoder_x = LabelEncoder()
x_data["WindGustDir"] = labelencoder_x.fit_transform(x_data["WindGustDir"]).reshape(-1,1)
x_data["WindDir9am"] = labelencoder_x.fit_transform(x_data["WindDir9am"]).reshape(-1,1)
x_data["WindDir3pm"] = labelencoder_x.fit_transform(x_data["WindDir3pm"]).reshape(-1,1)
#x_data
#Normalization
x=((x_data-np.min(x_data))/(np.max(x_data))-(np.min(x_data)))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(x_train,y_train)

print("Accuracy :",lr.score(x_test,y_test))