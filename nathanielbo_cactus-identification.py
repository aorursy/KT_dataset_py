import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image



from sklearn.model_selection import train_test_split



from tqdm import tqdm



import os as os

os.getcwd()



train = pd.read_csv('../input/train.csv')



train.head()
train_image = []

for i in tqdm(range(train.shape[0])):

    img = image.load_img('../input/train/train/'+ train['id'][i], target_size=(32, 32, 1), 

                         grayscale=False)

    img = image.img_to_array(img)

    img = img/255

    train_image.append(img)

X = np.array(train_image)



X.shape
y=train['has_cactus']

y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42,

                                                   test_size=.33)
#verifying shape of each object

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
nn1 = Sequential()

nn1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 

               input_shape=(32,32,3)))

nn1.add(Conv2D(64, (3,3), activation = 'relu'))

nn1.add(MaxPooling2D(pool_size=(2,2)))

nn1.add(Dropout(0.25))

nn1.add(Flatten())

nn1.add(Dense(128, activation='relu'))

nn1.add(Dropout(0.5))

nn1.add(Dense(2, activation='softmax'))
nn1.compile(loss='categorical_crossentropy',

            optimizer = 'Adam', metrics = ['accuracy'])
nn1.fit(X_train, y_train, epochs=10, 

        validation_data=(X_test, y_test))
nn1_score = nn1.evaluate(X_test, y_test, batch_size=128)
print('Loss ----------------- Accuracy')

print(nn1_score)
from sklearn.metrics import classification_report, confusion_matrix

from keras.utils import np_utils



nn1_pred = nn1.predict(X_test)



nn1_pred_as_class = nn1_pred.argmax(axis=-1)

y_test_as_class = y_test.argmax(axis=-1)



print(classification_report(y_test_as_class, nn1_pred_as_class))

import glob

from PIL import Image

folder = glob.glob('../input/test/test/*.jpg')
Z = np.array([np.array(Image.open(img)) for img in folder])

Z.shape
sub = nn1.predict_proba(Z) #This command gives us probability for each class. 
sub_df = pd.DataFrame(sub, columns = ['no_cactus','has_cactus'])
sub_df.head()
img_names = os.listdir('../input/test/test/')



sub_df['id'] = img_names
del sub_df['no_cactus'] #We only need the probability that the image does in fact have a cactus

sub_df = sub_df[['id', 'has_cactus']]

sub_df.head()
sub_df.to_csv('sub_2.csv', index=False) #Creating the CSV for submission
X_train_flat = []

for sublist in X_train:

    for item in sublist:

        X_train_flat.append(item)

        

X_test_flat = []

for sublist in X_test:

    for item in sublist:

        X_test_flat.append(item)

        

#X_train_flat

X_train_pixels = X_train.flatten().reshape(11725, 3072)

#these shape sizes are calculated as such: 

    #first number is number of records in array

    #second number is multiplcation of the subsequent numbers in array, 

            #in this case: 32 * 32 * 3



X_test_pixels = X_test.flatten().reshape(5775, 3072)
#Here I'm transforming the target list to a dataframe so I can then delete the list of values for

#the 0 (no cactus) class. 

y_train_df = pd.DataFrame(y_train, index=y_train[:,0])

del y_train_df[0]

y_train_array = y_train_df.values



y_test_df = pd.DataFrame(y_test, index=y_test[:,0])

del y_test_df[0]

y_test_array = y_test_df.values

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score



MLP1 = MLPClassifier(activation='relu', hidden_layer_sizes=(18,9,5), learning_rate='constant',

       random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, warm_start=False)

MLP1.fit(X_train_pixels, y_train_array)

MLP1_preds = MLP1.predict(X_test_pixels)

print("Accuracy", accuracy_score(y_test_array, MLP1_preds))

target_names = ["No Cactus", "Cactus"]

print(classification_report(y_test_array, MLP1_preds, target_names=target_names))
'''

from sklearn.model_selection import GridSearchCV

import time

start_time = time.clock()

parameters = {'hidden_layer_sizes':[(1500, 1500, 1500), (1500, 800, 400), (500, 500, 500)]}

MLP2 = MLPClassifier(activation='relu', learning_rate='constant', random_state=1, solver='adam')

grid_MLP2 = GridSearchCV(MLP2, parameters, n_jobs=-1, cv=5)

grid_MLP2.fit(X_train_pixels, y_train_array)

print("BEST PARAM", MLP2.best_params_)

print("Time to run", time.clock() - start_time, "seconds")

'''
'''

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

import time

start_time = time.clock()

parameters = {

    'activation': ['tanh', 'relu'],

    'solver': ['sgd', 'adam'],

    'alpha': [0.0001, 0.05],

    'learning_rate': ['constant','adaptive']}

MLP3 = MLPClassifier(hidden_layer_sizes = (30, 50, 30), random_state=1)

grid_MLP3 = GridSearchCV(MLP3, parameters, n_jobs=-1, cv=5)

grid_MLP3.fit(X_train_pixels, y_train_array)

print("BEST PARAM", MLP3.best_params_)

print("Time to run", time.clock() - start_time, "seconds")

'''
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix



MLP2 = MLPClassifier(activation='relu', hidden_layer_sizes=(50, 30, 50), learning_rate='constant',

       random_state=1, shuffle=True, solver='adam', tol=0.0001, warm_start=False)

MLP2.fit(X_train_pixels, y_train_array)

MLP2_preds = MLP2.predict(X_test_pixels)

print("Accuracy", accuracy_score(y_test_array, MLP2_preds))

target_names = ["No Cactus", "Cactus"]

print(classification_report(y_test_array, MLP2_preds, target_names=target_names))
MLP3 = MLPClassifier(activation='relu', hidden_layer_sizes=(50, 30, 30), learning_rate='constant',

       random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, warm_start=False)

MLP3.fit(X_train_pixels, y_train_array)

MLP3_preds = MLP3.predict(X_test_pixels)

print("Accuracy", accuracy_score(y_test_array, MLP3_preds))

target_names = ["No Cactus", "Cactus"]

print(classification_report(y_test_array, MLP3_preds, target_names=target_names))
''''''

MLP4 = MLPClassifier(activation='relu', hidden_layer_sizes=(50, 30, 30, 30), learning_rate='constant',

       random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, warm_start=False)

MLP4.fit(X_train_pixels, y_train_array)

MLP4_preds = MLP3.predict(X_test_pixels)

print("Accuracy", accuracy_score(y_test_array, MLP4_preds))

target_names = ["No Cactus", "Cactus"]

print(classification_report(y_test_array, MLP4_preds, target_names=target_names))

''''''