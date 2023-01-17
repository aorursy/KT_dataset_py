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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
df=pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')

print('Number of rows and columns:', df.shape)
df.head(5)
df.info()
df.dtypes
df['Gender'].value_counts()
sns.factorplot('Gender',data=df,kind='count')
df['Geography'].value_counts()
sns.factorplot('Geography',data=df,kind='count')
sns.factorplot('Gender',data=df,hue='Geography',kind='count')
pd.crosstab([df["Gender"], df["Geography"]], df["Exited"], 

            margins = True).style.background_gradient(cmap = "summer_r")
fig, ax = plt.subplots(1, 2, figsize = (18, 8))

sns.countplot('Gender', hue = "Exited", data = df, ax = ax[1])

sns.countplot("Geography", hue = "Exited", data = df, ax = ax[0])

ax[0].set_title("Gender: Stay vs Exited")

ax[1].set_title("Geography: Stay vs Exited")

plt.show()
pd.crosstab([df["Geography"], df["IsActiveMember"]], df["Exited"], 

            margins = True).style.background_gradient(cmap = "summer_r")
df['Exited'].value_counts()
sns.catplot(x="Gender", col = 'Exited', data=df, kind = 'count', palette='pastel')

sns.catplot(x="Geography", col = 'Exited', data=df, kind = 'count', palette='pastel')

sns.catplot(x="Gender", col = 'IsActiveMember', data=df, kind = 'count', palette='pastel')

plt.show()
df.drop(['RowNumber', 'CustomerId', 'Surname', 'Geography'], axis=1, inplace=True)
df.Gender = [1 if each == 'Male' else 0 for each in df.Gender]
df.head(5)
y = df.Exited.values

xdata = df.drop(['Exited'], axis=1)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(xdata)

scaled_xdata = scaler.transform(xdata)
x = (xdata - np.min(xdata)) / (np.max(xdata)-np.min(xdata))

x.head()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(scaled_xdata, y, test_size=0.10, random_state=7)
print('x_train shape: ', x_train.shape)

print('y_train shape: ', y_train.shape)

print('x_test shape: ', x_test.shape)

print('y_test shape: ', y_test.shape)
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()



lr.fit(x_train, y_train)



y_predlr = lr.predict(x_test)
score_lr = lr.score(x_test, y_test)

print(score_lr*100,'%')
mae = metrics.mean_absolute_error(y_test,y_predlr)

mse = metrics.mean_squared_error(y_test, y_predlr)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_predlr))

r2_square = metrics.r2_score(y_test, y_predlr)

print('Total Mean Absolute Error:', mae)

print('Total Mean Squared Error:', mse)

print('Total Root Mean Squared Error:', rmse)

print('Total R2 Square', r2_square)
lrcm = metrics.confusion_matrix(y_test, y_predlr)

plt.figure(figsize=(9,9))

sns.heatmap(lrcm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score_lr)

plt.title(all_sample_title, size = 15);
from sklearn.naive_bayes import GaussianNB



# Defining model:

nb = GaussianNB()



# Training the model:

nb.fit(x_train, y_train)



# Predicting:

y_prednb = nb.predict(x_test)
score_nb = nb.score(x_test, y_test)

print(score_nb)
nbcm = metrics.confusion_matrix(y_test, y_prednb)

plt.figure(figsize=(9,9))

sns.heatmap(nbcm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score_nb)

plt.title(all_sample_title, size = 15);
mae = metrics.mean_absolute_error(y_test,y_prednb)

mse = metrics.mean_squared_error(y_test, y_prednb)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_prednb))

r2_square = metrics.r2_score(y_test, y_prednb)

print('Total Mean Absolute Error:', mae)

print('Total Mean Squared Error:', mse)

print('Total Root Mean Squared Error:', rmse)

print('Total R2 Square', r2_square)
from sklearn.ensemble import RandomForestClassifier



# Defining:

rf = RandomForestClassifier(n_estimators=50, random_state=3)



# Training:

rf.fit(x_train, y_train)



# Predicting:

y_predrf = rf.predict(x_test)
score_rf = rf.score(x_test, y_test)

print(score_rf)
rfcm = metrics.confusion_matrix(y_test, y_predrf)

plt.figure(figsize=(9,9))

sns.heatmap(rfcm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score_rf)

plt.title(all_sample_title, size = 15);
mae = metrics.mean_absolute_error(y_test,y_predrf)

mse = metrics.mean_squared_error(y_test, y_predrf)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_predrf))

r2_square = metrics.r2_score(y_test, y_predrf)

print('Total Mean Absolute Error:', mae)

print('Total Mean Squared Error:', mse)

print('Total Root Mean Squared Error:', rmse)

print('Total R2 Square', r2_square)
from sklearn.svm import SVC



svm = SVC(random_state=2)



svm.fit(x_train, y_train)



y_predsvm = svm.predict(x_test)
score_svm = svm.score(x_test, y_test)

print(score_svm)
svmcm = metrics.confusion_matrix(y_test, y_predsvm)

plt.figure(figsize=(9,9))

sns.heatmap(svmcm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score_svm)

plt.title(all_sample_title, size = 15);
mae = metrics.mean_absolute_error(y_test,y_predsvm)

mse = metrics.mean_squared_error(y_test, y_predsvm)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_predsvm))

r2_square = metrics.r2_score(y_test, y_predsvm)

print('Total Mean Absolute Error:', mae)

print('Total Mean Squared Error:', mse)

print('Total Root Mean Squared Error:', rmse)

print('Total R2 Square', r2_square)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout
# creating the model

model = Sequential()



# first hidden layer

model.add(Dense(8, activation = 'relu', input_dim =  9))



# second hidden layer

model.add(Dense(8, activation = 'relu'))



# third hidden layer

model.add(Dense(8,activation = 'relu'))



# fourth hidden layer

model.add(Dense(8, activation = 'relu'))



# fifth hidden layer

model.add(Dense(8,activation = 'relu'))



# output layer

model.add(Dense(1, activation = 'sigmoid'))



# Compiling the NN

# binary_crossentropy loss function used when a binary output is expected

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 



model.fit(x_train, y_train, batch_size = 10, epochs = 50)
y_predann = model.predict(x_test)
from sklearn.metrics import accuracy_score
score_ann = accuracy_score(y_test, y_predann.round())
svmcm = metrics.confusion_matrix(y_test, y_predann.round())
svmcm = metrics.confusion_matrix(y_test, y_predann.round())

plt.figure(figsize=(9,9))

sns.heatmap(svmcm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score_ann)

plt.title(all_sample_title, size = 15);
mae = metrics.mean_absolute_error(y_test,y_predann.round())

mse = metrics.mean_squared_error(y_test, y_predann.round())

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_predann.round()))

r2_square = metrics.r2_score(y_test, y_predann.round())

print('Total Mean Absolute Error:', mae)

print('Total Mean Squared Error:', mse)

print('Total Root Mean Squared Error:', rmse)

print('Total R2 Square', r2_square)