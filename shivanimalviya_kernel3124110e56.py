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
data = pd.read_csv("../input/rainanomaly.csv")
data
day = list(data['day'])

month = list(data['month'])

year = list(data['year'])

day[35060]
del data['hour']

data
del data['DEWP']

data
del data['station']

data
del data['No']

data
day=list(data['day'])

month=list(data['month'])

year=list(data['year'])

rain=list(data['RAIN'])
date=[]

year_string=[]

month_string=[]

for i in range(len(day)):

    si= (year[i])

    year_string.append(si)

    sj= (month[i])

    month_string.append(sj)

    s =  (day[i])

    date.append(s)

date
new_rain=[]

date1=[]

year_string1=[]

month_string1=[]

summ=rain[0]

for i in range(len(day)-1):

    if date[i]== date[i+1]:

        summ+=rain[i+1]

    else:

        new_rain.append(summ)

        summ=rain[i+1]

        date1.append(date[i])

        month_string1.append(month_string[i])

        year_string1.append(year_string[i])
new_data = pd.DataFrame({'Date' :date1,'Month':month_string1,'Year':year_string1,'Rain':new_rain})

new_data
import matplotlib.pyplot as plt

%matplotlib inline
plt.scatter(new_data['Date'],new_data['Rain'])
plt.scatter(new_data['Month']*100 + new_data['Date'],new_data['Rain'])
X= new_data['Month']

y= new_data['Rain']
summ=0

final_year=[]

final_month=[]

final_rain=[]

for i in range(len(X)-1):

    if X[i]== X[i+1]:

        summ+=new_rain[i+1]

    else:

        final_rain.append(summ)

        summ=new_rain[i+1]

        final_month.append(month_string1[i])

        final_year.append(year_string1[i])
new_data2 = pd.DataFrame({'Month':final_month,'Year':final_year,'Rain':final_rain})

new_data2
X= new_data2['Month']

y= new_data2['Rain']
X
from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.01 )

# X_train, X_val, y_train ,y_val= train_test_split(X_train, y_train,test_size=0.3, random_state=1)

X_train.fillna(0)

y_train.fillna(0)
X_train = np.array(X_train)

X_train = np.nan_to_num(X_train).reshape(-1,1)

X_test = np.array(X_test)

X_test = np.nan_to_num(X_test).reshape(-1,1)

np.sort(X_train, axis=0)

X_train
y_train = np.array(y_train)

y_train = np.nan_to_num(y_train)

y_test = np.array(y_test)

y_test = np.nan_to_num(y_test)
lm=linear_model.LinearRegression()
model=lm.fit(X_train, y_train)
predictions=lm.predict(X_test)
predictions