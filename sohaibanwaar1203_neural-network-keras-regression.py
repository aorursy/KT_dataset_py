# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import numpy

import pandas

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense ,Dropout,BatchNormalization

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

#changing names because previous names are little bit confusing

df=df.rename(index=str, columns={"GRE Score": "GRE", "TOEFL Score": "TOEFL", "Chance of Admit ": "Admission_Chance"})

#we donot need serial number so its good to drop it because its just a number

df=df.drop("Serial No.",axis=1)

df.head(10)
df.describe()
admit=np.asarray(df["Admission_Chance"])

len(np.unique(admit))

#we have 60 different values in the coloum [chance to predict]
import pandas as pd

import matplotlib.pyplot as plt

corr = df.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(df.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(df.columns)

ax.set_yticklabels(df.columns)

plt.show()
import seaborn as sns

corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
fig = plt.figure(figsize = (20, 25))

j = 0

for i in df.columns:

    plt.subplot(6, 4, j+1)

    j += 1

    sns.distplot(df[i][df['Admission_Chance']<0.72], color='r', label = 'Not Got Admission')

    sns.distplot(df[i][df['Admission_Chance']>0.72], color='g', label = 'Got Admission')

    plt.legend(loc='best')

fig.suptitle('Admission Chance In University ')

fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
for column in df:

    plt.figure()

    sns.boxplot(x=df[column])
for column_1st in df:

    for coloum_2nd in df:

        jet=plt.get_cmap('jet')

        plt.figure(figsize=(15,5))

        plt.scatter(df[column_1st], df[coloum_2nd], s=30, c=df['Admission_Chance'], vmin=0, vmax=1, cmap=jet)

        plt.xlabel(column_1st,fontsize=40)

        plt.ylabel(coloum_2nd,fontsize=40)

        plt.colorbar()

        plt.show()
# define base model

def baseline_model():

    # create model

    model = Sequential()

    

    

    

    model.add(Dense(16, input_dim=7, activation='relu'))

    

    

    model.add(Dense(8, input_dim=7, activation='relu'))

    

    

    model.add(Dense(1))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
X=np.asarray(df.drop("Admission_Chance",axis=1))

Y=np.asarray(df["Admission_Chance"])
X_train, X_test, y_train, y_test = train_test_split(

     X,Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler =  MinMaxScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.fit_transform(X_test)


estimator = KerasRegressor(build_fn=baseline_model, epochs=30, batch_size=3, verbose=1)
history=estimator.fit(X_train,y_train)
# Plot training & validation loss values

plt.plot(history.history['loss'])



plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
from sklearn.metrics import accuracy_score

prediction = estimator.predict(X_test)



train_error =  np.abs(y_test - prediction)

mean_error = np.mean(train_error)

min_error = np.min(train_error)

max_error = np.max(train_error)

std_error = np.std(train_error)



print("std_error: ",std_error)

print("mean_error: ",mean_error)

print("min_error: ",min_error)

print("max_error: ",max_error)
#Visualising the Acutal and predicted Result

plt.plot(y_test, color = 'green', label = 'Actual')

plt.plot(prediction, color = 'blue', label = 'Predicted')

plt.grid(alpha = 0.3)

plt.xlabel('Number of Candidate')

plt.ylabel('Score')

plt.title('Actual vs Predicted')

plt.legend()

plt.show()
from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y_test,prediction))

print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(estimator.predict(X_test[[1],:])))

print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(estimator.predict(X_test[[2],:])))



train_prediction = estimator.predict(X_train)

print("r_square score (train dataset): ", r2_score(y_train,train_prediction))
# Save the weights

estimator.model.save_weights('model_weights.h5')



# Save the model architecture

with open('model_architecture.json', 'w') as f:

    f.write(estimator.model.to_json())