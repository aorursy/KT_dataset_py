import os

import sys

from pandas import Series

import pandas as pd

import numpy as np

import traceback

import time

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



#import RPi.GPIO as GPIO

#from Adafruit_BME280 import *

#import paho.mqtt.publish as publish

degrees = 0

fah = 0

hectopascals = 0

humidity = 0

global iteri

global df
df = pd.DataFrame(columns=['Date','Temperature','Humidity','Pressure'])

iteri=0
print(os.listdir("../input"))
data = pd.read_csv("../input/bme280sensordata/pandas_simple.csv")



data2 = data[['Temperature','Humidity','Pressure','class']]

data.head()
print('Shape of the data set: ' + str(data.shape))
print("describe: ")

print(data.describe())
print("info: ")

print(data.info())
plt.title('Temperature')

plt.xlabel('Time(s)')

plt.ylabel('Temperature')

aX=data['Temperature']





plt.plot(aX, label = "Temperature", color='red')



plt.legend()
minimum = data['Temperature'].min()

maximum = data['Temperature'].max()

average = data['Temperature'].mean()



print("Minimum Temperature is " + str(minimum))

print("Maximum Temperature is " + str(maximum))

print("Average Temperature is " + str(average))
plt.title('Humidity')

plt.xlabel('Time(s)')

plt.ylabel('Humidity')

aX=data['Humidity']





plt.plot(aX, label = "Humidity", color='red')



plt.legend()
minimum = data['Humidity'].min()

maximum = data['Humidity'].max()

average = data['Humidity'].mean()



print("Minimum Humidity is " + str(minimum))

print("Maximum Humidity is " + str(maximum))

print("Average Humidity is " + str(average))
plt.title('Pressure')

plt.xlabel('Time(s)')

plt.ylabel('Pressure')

aX=data['Pressure']





plt.plot(aX, label = "Pressure", color='red')



plt.legend()
minimum = data['Pressure'].min()

maximum = data['Pressure'].max()

average = data['Pressure'].mean()



print("Minimum Pressure is " + str(minimum))

print("Maximum Pressure is " + str(maximum))

print("Average Pressure is " + str(average))
data.dropna(inplace=True)

data.drop_duplicates(inplace=True)

sns.pairplot(data);
X = data2.drop('class',axis=1).values

y = data2['class'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)
neighbors = np.arange(1,9)

train_accuracy =np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



for i,k in enumerate(neighbors):

    #Setup a knn classifier with k neighbors

    knn = KNeighborsClassifier(n_neighbors=k)

    

    #Fit the model

    knn.fit(X_train, y_train)

    

    #Compute accuracy on the training set

    train_accuracy[i] = knn.score(X_train, y_train)

    

    #Compute accuracy on the test set

    test_accuracy[i] = knn.score(X_test, y_test)
plt.title('k-NN Varying number of neighbors')

plt.plot(neighbors, test_accuracy, label='Testing Accuracy')

plt.plot(neighbors, train_accuracy, label='Training accuracy')

plt.legend()

plt.xlabel('Number of neighbors')

plt.ylabel('Accuracy')

plt.show()
#Setup a knn classifier with k neighbors

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
#let us get the predictions using the classifier we had fit above

y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
print(classification_report(y_test,y_pred))
y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='Knn')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('Knn(n_neighbors=3) ROC curve')

plt.show()
roc_auc_score(y_test,y_pred_proba)
def readSensor():

    global degrees, fah, pascals, hectopascals, humidity,iteri,df



    # Changing the mode to BOARD (pin number on the board)

    degrees = sensor.read_temperature()

    fah = 9.0 / 5.0 * degrees + 32

    pascals = sensor.read_pressure()

    hectopascals = pascals / 100

    humidity = sensor.read_humidity()

    

    df=df.append({'Date':time.strftime("%Y-%m-%d %H:%M:%S"),'Temperature': fah, 'Humidity': humidity, 'Pressure': hectopascals}, ignore_index=True)

    

    iteri = iteri + 1

    if iteri ==100:

        df.to_csv("pandas_simple.csv", encoding='utf-8', doublequote=False, index=False, mode="a", header=False)

        df = df[0:0]

        iteri= 0

        print('dataframe eptied, records added')

    

def printInfo():

    

    print('Fahrenheit= {0:0.3f} deg F'.format(fah))

    print('Celsius   = {0:0.3f} deg C'.format(degrees))

    print('Pressure  = {0:0.2f} hPa'.format(hectopascals))

    print('Humidity  = {0:0.2f} %'.format(humidity))

    print('==========={0}==========='.format(time.strftime("%Y-%m-%d %H:%M:%S")))
def main_loop() :

    

    try:

        while True:

            readSensor()

            printInfo()

            time.sleep(1.0)

    except KeyboardInterrupt:

        print('interruption')

        print(df.head())
# The following makes this program start running at main_loop() 

# when executed as a stand-alone program.    F

if __name__ == '__main__':

    try:

        main_loop()

    except :

        traceback.print_exc(file=sys.stdout)

    finally:

    # reset the sensor before exiting the program

        GPIO.cleanup()