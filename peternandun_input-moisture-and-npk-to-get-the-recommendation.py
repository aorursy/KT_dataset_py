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
import matplotlib.pyplot as plt

dataset = pd.read_csv('/kaggle/input/crop-nutrient-database/crops.csv')

dataset.head()
variable_split = dataset['Symbol'].str.split(', ')

variable_split
dataset_modif = dataset
dataset_modif

dataset_modif['Category1'] = variable_split.str.get(0)

dataset_modif['Category2'] = variable_split.str.get(1)

dataset_modif
col_list = list(dataset.columns)

col_list.pop()

col_list.pop()

col_list
dataset1 = pd.melt(dataset, id_vars=col_list, value_name="Crops_Symbol")

dataset1
dataset1.isnull().sum()
dataset2 = dataset1.dropna(subset=['Crops_Symbol'], how='any')

dataset2
dummies = pd.get_dummies(dataset2.Crops_Symbol)

np.save('Index_Names',dummies.columns)
dataset3 = dataset2[['Crop', 'ScientificName', 'Symbol', 'NuContAvailable', 'PlantPartHarvested', 'CropCategory', 'AvMoisture%', 'AvN%(dry)', 'AvP%(dry)', 'AvK%(dry)']]

dataset3
merged = pd.concat([dataset3,dummies], axis='columns')

merged
merged.to_csv('Categorical_Crop.csv')

import keras
# Importing dataset and separating dependent and independent variables

dataset = pd.read_csv('Categorical_Crop.csv', na_values=['#VALUE!', '#DIV/0!']) # it is to replace the #DIV/0 with Nan while reading the csv file

X = dataset.iloc[:, 7:11].values

Y = dataset.iloc[:, 11:].values
# To replace any of the missing values in dependent variable with column mean

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(X)

X = imputer.transform(X)
# Scaling the column values with standard scaling to keep variance in the range of 1

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = sc_X.fit_transform(X)
# Splitting the data into test and train set for validation

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
from keras.models import Sequential

from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(activation = 'relu', input_dim = 4, units = 109, kernel_initializer= 'uniform'))

classifier.add(Dense(activation = 'relu', units = 109, kernel_initializer= 'uniform'))

classifier.add(Dense(activation = 'softmax', units = 214, kernel_initializer= 'uniform'))

classifier.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, Y_train,validation_data=(X_test,Y_test), batch_size=10, epochs=300)

# Training the model from my pc s unable to figure out the nan loss and then uploading the json model and h5 weights of the architecture
# classifier.summary()

# model_json = classifier.to_json()

# with open("model.json", "w") as json_file:

#     json_file.write(model_json)

# # serialize weights to HDF5

# classifier.save_weights("model.h5")

# print("Saved model to disk")
#Loading model if after saving we want to test the performance

from keras.models import model_from_json

json_file = open("/kaggle/input/json-file/model3.json", 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("/kaggle/input/json-file/model3.h5")

print("Loaded model from disk")

# evaluate loaded model on test data

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

score = loaded_model.evaluate(X_test, Y_test, verbose=0)

Y_pred = loaded_model.predict(X_test)

print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
Y_pred_index = np.argmax(Y_pred, axis = 1)

Y_test_index = np.argmax(Y_test, axis = 1)



out = np.zeros((Y_test.shape[0],5))

result = np.zeros((Y_test.shape[0],1))

count_1 = 0



for i in range(Y_test.shape[0]):

    arr = Y_pred[i,:]

    out[i] = arr.argsort()[-5:][::-1]

    if Y_test_index[i] in out[i,:]:

        result[i] = 1

        count_1 += 1



print("Accuarcy is : ", count_1/Y_test.shape[0])



for i in range(Y_test.shape[0]):

    print("Predicted Label: ", out[i], " True Label : ", Y_test_index[i], " ", result[i])
