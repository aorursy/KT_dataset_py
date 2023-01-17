# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

paths=[]

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path = os.path.join(dirname, filename)

        paths.append(path)

        print(path)



# Any results you write to the current directory are saved as output.
data = pd.read_csv(paths[0])

data.describe()
data.head()
# data.drop(0, axis=0, inplace=True) Deleting first because it has maximum values, probably initial malfunction

data['SST'].replace(-99.9, data['SST'].values.mean(), inplace=True) # I am gonna predict temp, so i need to make sure that there won't be any -99 suddenly
data['SST'].describe()
data.hist(figsize=(10,10))
figure = plt.figure(figsize=(10,8))

corr_matrix = data.corr()

plt.matshow(corr_matrix, fignum=figure.number)

plt.xticks(np.arange(6), data.columns[1:])

plt.yticks(np.arange(6), data.columns[1:])

legend=plt.colorbar()

legend.ax.tick_params(labelsize=10)
corr_matrix.style.background_gradient(cmap='coolwarm')
plt.figure(figsize=(30,10), dpi=96)

# input comes every 30 minutes, so for one year we will have 17520 records

year=17520

month = 1440

day = 48

x_temp = data.iloc[:year,0]

y_temp = data.iloc[:year,6]

x=x_temp[23::day]

y=y_temp[23::day]

plt.plot(x,y)



step = month

xs = x[::31]

plt.xticks(xs)

plt.gca().set(title='temperature at every day in 2018 gathered at 12:00',xlabel='date', ylabel='temperature')

plt.show()

i = 1 # as 01.01.2018 01:00

X_tmp = data.iloc[:17520,0]

X_tmp = X_tmp[1::2]

Xs = [i for i in range(len(X_tmp))]

X = [Xs]





# Adjusting y



y_tmp = data.iloc[:17520,6]

y_tmp = y_tmp[1::2]

ys = [i for i in y_tmp]

y = [ys]







print(f"Oryginal data: {data['Date/Time'][1:10:2]} : {data['SST'][1:10:2]} \n\n")

print(f"Fromated data: {X[0][:5]} : {y[0][:5]}")

#everything is correct
X_train, X_test, y_train, y_test = train_test_split(X[0], y[0], test_size=0.2, random_state=42)
model = LinearRegression()

X_train_tmp = np.asarray(X_train).reshape(-1,1)

y_train_tmp = np.asarray(y_train).reshape(-1,1)

X_train_tmp = np.asarray(X_test).reshape(-1,1)

y_train_tmp = np.asarray(y_test).reshape(-1,1)



model.fit(X_train_tmp,y_train_tmp)

model.score(X_train_tmp, y_train_tmp)
model2 = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')



lab_encoder = LabelEncoder()

X_train_tmp = lab_encoder.fit_transform(X_train)

y_train_tmp = lab_encoder.fit_transform(y_train)

X_train_tmp = np.asarray(X_train_tmp).reshape(-1,1)





model2.fit(X_train_tmp,y_train_tmp)

model2.score(X_train_tmp, y_train_tmp)
model3 = svm.SVC(gamma='scale')

model3.fit(X_train_tmp,y_train_tmp)

model3.score(X_train_tmp, y_train_tmp)