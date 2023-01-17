# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

train=pd.read_excel("../input/Data_Train.xlsx")
test=pd.read_excel("../input/Test_set.xlsx")
train.head()
test.head()
train.info()
test.info()
train=train.dropna()
train['Journey_Day'] = pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.day

train['Journey_Day'] = pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.month
test['Journey_Day'] = pd.to_datetime(test.Date_of_Journey, format='%d/%m/%Y').dt.day

test['Journey_Day'] = pd.to_datetime(test.Date_of_Journey, format='%d/%m/%Y').dt.month
train.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)



test.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)
duration = list(train['Duration'])



for i in range(len(duration)) :

    if len(duration[i].split()) != 2:

        if 'h' in duration[i] :

            duration[i] = duration[i].strip() + ' 0m'

        elif 'm' in duration[i] :

            duration[i] = '0h {}'.format(duration[i].strip())



dur_hours = []

dur_minutes = []  



for i in range(len(duration)) :

    dur_hours.append(int(duration[i].split()[0][:-1]))

    dur_minutes.append(int(duration[i].split()[1][:-1]))

    

train['Duration_hours'] = dur_hours

train['Duration_minutes'] =dur_minutes



train.drop(labels = 'Duration', axis = 1, inplace = True)

durationT = list(test['Duration'])



for i in range(len(durationT)) :

    if len(durationT[i].split()) != 2:

        if 'h' in durationT[i] :

            durationT[i] = durationT[i].strip() + ' 0m'

        elif 'm' in durationT[i] :

            durationT[i] = '0h {}'.format(durationT[i].strip())

            

dur_hours = []

dur_minutes = []  



for i in range(len(durationT)) :

    dur_hours.append(int(durationT[i].split()[0][:-1]))

    dur_minutes.append(int(durationT[i].split()[1][:-1]))

  

    

test['Duration_hours'] = dur_hours

test['Duration_minutes'] = dur_minutes



test.drop(labels = 'Duration', axis = 1, inplace = True)
train['Depart_Time_Hour'] = pd.to_datetime(train.Dep_Time).dt.hour

train['Depart_Time_Minutes'] = pd.to_datetime(train.Dep_Time).dt.minute



train.drop(labels = 'Dep_Time', axis = 1, inplace = True)





train['Arr_Time_Hour'] = pd.to_datetime(train.Arrival_Time).dt.hour

train['Arr_Time_Minutes'] = pd.to_datetime(train.Arrival_Time).dt.minute



train.drop(labels = 'Arrival_Time', axis = 1, inplace = True)

test['Depart_Time_Hour'] = pd.to_datetime(test.Dep_Time).dt.hour

test['Depart_Time_Minutes'] = pd.to_datetime(test.Dep_Time).dt.minute





test.drop(labels = 'Dep_Time', axis = 1, inplace = True)



test['Arr_Time_Hour'] = pd.to_datetime(test.Arrival_Time).dt.hour

test['Arr_Time_Minutes'] = pd.to_datetime(test.Arrival_Time).dt.minute



test.drop(labels = 'Arrival_Time', axis = 1, inplace = True)
train.head()
test.head()
Y_train = train.iloc[:,6].values 

X_train = train.iloc[:,train.columns != 'Price'].values

X_test = test.iloc[:,:].values
from sklearn.preprocessing import LabelEncoder



le1 = LabelEncoder()

le2 = LabelEncoder()



# Training Set    



X_train[:,0] = le1.fit_transform(X_train[:,0])



X_train[:,1] = le1.fit_transform(X_train[:,1])



X_train[:,2] = le1.fit_transform(X_train[:,2])



X_train[:,3] = le1.fit_transform(X_train[:,3])



X_train[:,4] = le1.fit_transform(X_train[:,4])



X_train[:,5] = le1.fit_transform(X_train[:,5])

X_test[:,0] = le2.fit_transform(X_test[:,0])



X_test[:,1] = le2.fit_transform(X_test[:,1])



X_test[:,2] = le2.fit_transform(X_test[:,2])



X_test[:,3] = le2.fit_transform(X_test[:,3])



X_test[:,4] = le2.fit_transform(X_test[:,4])



X_test[:,5] = le2.fit_transform(X_test[:,5])
print(pd.DataFrame(X_train).head())
from sklearn.preprocessing import StandardScaler



sc_X = StandardScaler()



X_train = sc_X.fit_transform(X_train)



X_test = sc_X.transform(X_test)



#sc_y = StandardScaler()



Y_train = Y_train.reshape((len(Y_train), 1)) 



Y_train = sc_X.fit_transform(Y_train)



Y_train = Y_train.ravel()
pd.DataFrame(X_train).head()
pd.DataFrame(Y_train).head()
from sklearn.svm import SVR



svr = SVR(kernel = "rbf")



svr.fit(X_train,Y_train)



Y_pred = sc_X.inverse_transform(svr.predict(X_test))





df=pd.DataFrame(Y_pred, columns = ['Price'])
df.to_excel("Sample_submission.xlsx", index=False)
df.head()