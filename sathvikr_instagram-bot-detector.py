# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

input_data = pd.read_csv("/kaggle/input/instagram-fake-spammer-genuine-accounts/train.csv")
Y_train = input_data["fake"]
X_train = input_data.drop("fake", axis=1)
test_data = pd.read_csv("/kaggle/input/instagram-fake-spammer-genuine-accounts/test.csv")
Y_test = test_data["fake"]
X_test = test_data.drop("fake", axis = 1)
import keras
from keras.layers import *
model = keras.models.Sequential()
model.add(Dense(60, input_dim = 11, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
from keras import metrics
model.compile(optimizer="adam", loss = "binary_crossentropy", metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
model.fit(X_train, Y_train, batch_size = 5, epochs = 200, verbose = 0)
loss = model.evaluate(X_test, Y_test, verbose=1)

model.summary()
print(loss)
import csv

fake = 0
with open('fakeAcc', 'w') as csv_file:
    write = csv.writer(csv_file, delimiter=',')
    with open('/kaggle/input/jarviscsv/Jarvis.csv', 'rt', encoding='UTF8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',', lineterminator='\n')
        for row in readCSV:
            
                user = row[0]
                name = row[1]
                pic = row[2]
                bio = row[3]
                col10 = int(row[4])
                col11 = int(row[5])
                col7 = int(row[6])
                col8 = int(row[7]) 
                col9 = int(row[8])

                #Column 1
                col1 = 1
                if pic == 'https://instagram.faep8-2.fna.fbcdn.net/v/t51.2885-19/44884218_345707102882519_2446069589734326272_n.jpg?_nc_ht=instagram.faep8-2.fna.fbcdn.net&_nc_ohc=CgRv3KotZXUAX92x_dJ&oh=9c0e8a7b8e58234fa2e781c4f25d4982&oe=5F17A10F&ig_cache_key=YW5vbnltb3VzX3Byb2ZpbGVfcGlj.2':
                    col1 = 0

                #Column 2
                userDigits = 0
                for i in user:
                    if i.isdigit():
                        userDigits += 1
                col2 = float(userDigits/len(user))

                #Column 3
                tokens = name.split(" ")
                col3 = len(tokens)

                #Column 4
                nameDigits = 0
                for i in name:
                    if i.isdigit():
                        nameDigits += 1
                name4 = name.replace(' ', '')
                if len(name) == 0:
                    name4 = " "
                col4 = float(nameDigits/len(name4))

                #Column 5
                col5 = 0
                if name == user:
                    col5 = 1

                #Column 6
                col6 = len(bio)
                arr = [[col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11]]
                y_pred = model.predict(arr)

                if y_pred[0] == 1:
                    fake+= 1  

                #Write to file 

                write.writerow([user, y_pred[0]])
