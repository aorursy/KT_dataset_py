# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Imports data for january 2019
flights = pd.read_csv("/kaggle/input/flight-delay-prediction/Jan_2019_ontime.csv")


airportdata = pd.read_excel("/kaggle/input/airportcodedata/AirportCodeData.xlsx")


#weather = pd.read_csv("/kaggle/input/weather/US_WeatherEvents_2016-2019.csv")


df = flights[["DAY_OF_MONTH","OP_CARRIER_AIRLINE_ID","ORIGIN","DEST","DEP_TIME","ARR_TIME","ARR_DEL15"]]

#df2 = weather[["Type","Severity","StartTime(UTC)","EndTime(UTC)","TimeZone","AirportCode"]]
#df3 = weather[["StartTime(UTC)"]]

#df3_array = df3.to_numpy()
#y = str(df3_array[0])[2:-2].split("/")
#w = y[2].split(" ")
#v = y[0:2] + w
#v[3].replace(':','')
#print(v)
#month = [x[0] for x in df3_array]
#print(month)


df = df.dropna()

df = df.sample(frac=1).reset_index(drop=True)
df = df.loc[:20000,:]
df = pd.get_dummies(df)

df.tail(5)
x_train, x_test, y_train, y_test = train_test_split(df.drop(labels="ARR_DEL15",axis=1),df[["ARR_DEL15"]],test_size=0.3)
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train.values.ravel())
index = y_test.index.to_list()
index_str = [str(x) for x in index]
#print(index_str)
#model.score(x_test,y_test.values.ravel())
predictions = model.predict(x_test)
#predict_array = predictions.to_numpy()
for x in range(200):
    print(predictions[x])

#x_test.head(10)
#y_test.head(10)
model.score(x_test,y_test)
len_test = len(y_test)
len_predict = len(predictions)

y_test_array = y_test.to_numpy()

#Correct prediction
correct_count = 0
false_negative = 0
false_positive = 0
for x in range(len_test):
    if y_test_array[x] == predictions[x]:
        correct_count+=1      
#False Positive, predict delay when on time
    if y_test_array[x] == 0 and predictions[x] == 1:
        false_negative+=1
#False Negative, predict no delay when not on time
    if y_test_array[x] == 1 and predictions[x] ==0:
        false_positive+=1

labels = ['Correct','False Negatives','False Positives']
sizes = [correct_count/len_test, false_negative/len_test, false_positive/len_test]
plt.pie(sizes,labels=labels,autopct='%1.1f%%')
plt.show()
#Percent on time
num_on_time = 0
for x in range(len(y_test_array)):
    if y_test_array[x] == 0:
        num_on_time+=1
print(num_on_time/(len(y_test_array)))
