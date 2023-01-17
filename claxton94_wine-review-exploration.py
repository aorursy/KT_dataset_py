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
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
# Inspecting the head of the dataframe to see what what variables we have
df.head()
# Inspecting what the data types are and if we have any missing values 
df.info()
# No missing values apparant, all data seems to be in the correct format. 

# Now I want to check some quick statistics
df.describe()
# Now I want to see how the variables are correlated and do some EDA 
import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(15,8))

sns.set_style(sns.set_palette('viridis'))

sns.heatmap(df.corr(),annot = True,cmap = 'viridis')
sns.boxplot(x= 'quality',y = 'alcohol',data = df)
sns.distplot(df.alcohol,bins = 30)
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
X = df.drop('quality',axis = 1)

Y = df['quality'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
error_rate = []

for i in range(1,100):

    KNN = KNeighborsClassifier(n_neighbors=i)

    KNN.fit(X_train,y_train)

    Pred = KNN.predict(X_test)

    error_rate.append(np.mean(Pred != y_test))
plt.plot(range(1,100),error_rate)
KNN = KNeighborsClassifier(n_neighbors=45)

KNN.fit(X_train,y_train)

Pred = KNN.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,Pred))

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators =250)
RF.fit(X_train,y_train)
PredRf = RF.predict(X_test)
print(classification_report(y_test,PredRf))
from sklearn.preprocessing import MinMaxScaler 

scaler  = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.optimizers import Adam
model = Sequential()



model.add(Dense(11,activation='relu'))

model.add(Dense(11,activation='relu'))

model.add(Dense(11,activation='relu'))

model.add(Dense(1))



model.compile(optimizer='adam',loss='mse')
model.fit(x=X_train,y=y_train,

          validation_data=(X_test,y_test),

          batch_size=128,epochs=400)
losses = pd.DataFrame(model.history.history)

losses.plot()
predictions = model.predict(X_test)
error = y_test.reshape(528,1) - predictions.round(0)
predictions.round(0)
y_test.reshape(528,1)
plt.scatter(y_test,predictions.round(0))