import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os 

import glob

from tqdm import tqdm # for  well-established ProgressBar

import seaborn as sb

from random import shuffle #only shuffles the array along the first axis of a multi-dimensional array. The order of sub-arrays is changed but their contents remains the same.

import os

data_dir = '../input/data/Data'

train_dir = os.path.join(data_dir, 'TrainData')

test_dir = os.path.join(data_dir, 'TestData')

CATEGORIES = ['ASD', 'Normal']
'''function that will create train data , will go thought all the file do this 

----read the csv file  

---change it to numpy arrays and  append it to dataframe train with it`s associated category '''

def create_train_dataframe():

    train = []

    for category_id, category in enumerate(CATEGORIES):

        for csvfile in tqdm(os.listdir(os.path.join(train_dir, category))):

            label=label_data_singleValue(category)

            path=os.path.join(train_dir,category,csvfile)

            traincsv = pd.read_csv(path,header=0,index_col=None)

            traincsv['ASD']=label

            if 'Data/TrainData/Normal/.DS_Store' in path :

                continue

            else :

                train.append(traincsv)

    frame = pd.concat(train, axis=0, ignore_index=True)

    return  frame
'''sample data of gaze position for person looking on left side of screen   '''

asd_df = pd.read_csv("../input/data/Data/TrainData/ASD/log.csv")

asd_df.describe()
'''sample data of gaze position for person looking on right side of screen   '''

asd_df = pd.read_csv("../input/data/Data/TrainData/Normal/log1.csv")

asd_df.describe()
def label_data_singleValue(word_label):                       

    if word_label == 'ASD': return 1

    elif word_label == 'Normal': return 0
frame=create_train_dataframe()

trainData=frame
frame.describe()
X = trainData.iloc[:,0:16]

Y = trainData.iloc[:,16]
from keras.models import Sequential

from keras.layers import Dense

from sklearn.model_selection import train_test_split

import numpy as np



# split the data into training (67%) and testing (33%)

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=16)





# create the model

model = Sequential()

model.add(Dense(64, input_dim=16, kernel_initializer='uniform', activation='relu'))

model.add(Dense(32, kernel_initializer='uniform', activation='relu'))

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))



# compile the model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# fit the model

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=200, batch_size=16, verbose=0)



# evaluate the model

scores = model.evaluate(X_test, Y_test)

print("Accuracy: %.2f%%" % (scores[1]*100))
def decode(datum):

    if(np.argmax(datum)==0):  return 'Normal'

    elif(np.argmax(datum)==1): return 'ASD'

    else: return 'Unknow'
test_df = pd.read_csv("../input/data/Data/TestData/log1.csv")

test_df.describe()
test_df2 = pd.read_csv("../input/data/Data/TestData/log.csv")

test_df2.describe()
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import accuracy_score,mean_squared_error



y_pred = model.predict(test_df)

print(y_pred.shape)

y_pred = np.where(y_pred<.5,0,1)



plt.plot(y_pred)

plt.title('Prediction')





y_pred2 = model.predict(test_df2)

print(y_pred2.shape)

y_pred2 = np.where(y_pred2<.5,0,1)

plt.plot(y_pred2)

plt.title('Prediction')
