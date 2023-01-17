# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
temp = pd.read_csv("../input/temps.csv")
temp.info()
temp = pd.get_dummies(temp)

temp.head()
temp1 = plt.figure()

temp1 = plt.plot(temp.index,temp.temp_1)

temp2 = plt.figure()

temp2 = plt.plot(temp.index,temp.temp_2)

temp3 = plt.figure()

temp3 = plt.plot(temp.index,temp.average)

temp4 = plt.figure()

temp4 = plt.plot(temp.index,temp.actual)
labels = np.array(temp['actual'])

features = np.array(temp.drop('actual',axis=1))

features_list = list(temp.columns.drop('actual'))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=.25,random_state=45)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000 , random_state = 45)

rf.fit(x_train,y_train);
predictions = rf.predict(x_test)
error = abs(y_test-predictions)
a = pd.DataFrame({'Pred': predictions[:],'Actual' : y_test[:],'error':abs(predictions[:]-y_test[:])})
a
accuracy = 100 - 100*(np.mean(error/y_test))

print ("Accuracy = ",accuracy)
err = plt.figure(figsize=(20,8))

err = plt.plot(a.index,a.error)
importance = pd.DataFrame(rf.feature_importances_,features_list)

importance = importance.rename(index = str , columns = {0:"error"})
fig = plt.figure(figsize = (30,10))

fig = plt.bar(importance.index,importance.error)