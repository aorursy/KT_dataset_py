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
import time
import os

from os import environ

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from keras.models import Sequential

from keras.layers import Dense 
from keras.models import model_from_json

from keras.layers import Dropout

from keras.layers import Dense 
from keras.optimizers import SGD
#Time

start_time = time.time()

# load dataset 
dataset=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
x=dataset.iloc[:,0:8]
y=dataset.iloc[:,8]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

# create model

model = Sequential()

model.add(Dense(12, activation="relu", input_dim=8,kernel_initializer="normal"))
model.add(Dropout(0.1)) 
model.add(Dense(8, activation="relu",kernel_initializer="normal")) 
model.add(Dense(80, activation="relu",kernel_initializer="normal"))
model.add(Dense(16, activation="relu",kernel_initializer="normal")) 
model.add(Dropout(0.1))
model.add(Dense(4, activation="relu", kernel_initializer="normal"))
model.add(Dropout(0.1))
model.add(Dense(8, activation="relu",kernel_initializer="normal"))
model.add(Dense(1, activation="sigmoid",kernel_initializer="normal"))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
# Fit the model

model.fit(x_train, y_train, epochs=500, batch_size=5, verbose=1) # Evaluate and save the weights to the file

scores = model.evaluate(x_train, y_train, verbose=0)

print("%s:%.2f%%"%(model.metrics_names[1],scores[1]*100))

import matplotlib.pyplot as plt

import seaborn as sns 
from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)

y_pred = [round(x[0]) for x in y_pred] 
cm=confusion_matrix(y_test,y_pred) 
sns.heatmap(cm, annot=True, fmt='g') 
plt.show()
print("Accuracy ", accuracy_score(y_pred, y_test)*100)

# serialize model to JSON

model_json = model.to_json() 
with open("model83.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model83.h5")

print("Saved model to disk") 
print("Execution took {} seconds".format(time.time()-start_time))

from keras.models import Sequential

from keras.layers import Dense

from keras.models import model_from_json

import numpy as np

import time

import os

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score

import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix

# Load JSON
start_time = time.time()

dataset=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv') 
x=dataset.iloc[:,0:8]

y=dataset.iloc[:,8]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

json_file = open('model83.json','r') 
loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model83.h5")


# load weights into new model

print("Loaded model and weights from disk")

# evaluate loaded model on test data

loaded_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 
# calculate predictions

test = np.array([[6,148,72,35,0,33.6,0.627,50]])

score = loaded_model.predict(test)

print(score)



# round predictions

rounded = [round(x[0]) for x in score]

print(rounded[0]) 
#print("%s:%.2f%%"%(loaded_model.metrics_names[1],score[0]*100))

y_pred = loaded_model.predict(x_test)

y_pred = [round(x[0]) for x in y_pred] 
cm=confusion_matrix(y_test,y_pred) 
sns.heatmap(cm, annot=True, fmt='g')

plt.show() 
print("Accuracy", accuracy_score(y_pred, y_test)*100)

print("Execution took {} seconds".format(time.time() - start_time))
