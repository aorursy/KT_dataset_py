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

        #print(os.path.join(dirname, filename))

        pass



# Any results you write to the current directory are saved as output.
#reqad the csv file that contains the details

ds = pd.read_csv("/kaggle/input/fashion-product-images-small/myntradataset/styles.csv",error_bad_lines=False)

ds.head()

#combine the id with .jpg to get the filenames...

ds['image'] = ds.apply(lambda row: str(row['id']) + ".jpg", axis=1)
from keras.preprocessing.image import ImageDataGenerator



#image generator object from keras. reference : Keras Docs

image_generator = ImageDataGenerator(

    validation_split=0.2

)



#create a flow of images for training the model.

training_generator = image_generator.flow_from_dataframe(

    dataframe=ds,

    directory= "/kaggle/input/fashion-product-images-small/myntradataset/images",

    x_col="image",

    y_col="masterCategory",

    target_size=(80,60),

    batch_size=256,

    subset="training"



)



#create a flow of images for validating(testing) the trained model.

validation_generator = image_generator.flow_from_dataframe(

    dataframe=ds,

    directory="/kaggle/input/fashion-product-images-small/myntradataset/images",

    x_col="image",

    y_col="masterCategory",

    target_size=(80,60),

    batch_size=256,

    subset="validation"

)
from keras import layers,models

#create a sequential model

model = models.Sequential()



#add the necessary layers.

model.add(layers.Conv2D(32, (5,5), strides = (2,2), activation = 'relu' , input_shape = (80,60,3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu')) 

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(7, activation='softmax'))



#compile

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#fit the model

history = model.fit_generator(training_generator, epochs=5, steps_per_epoch = 139 , verbose=1)

#keras provides an evaluate function that returns [metric, accuracy]

#this model takes the validation generator and number of steps/batches to validate on. (test_set_size/batch_size = 8960/256)

model.evaluate_generator(validation_generator,35)
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt # plotting

import matplotlib.image as mpimg



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.decomposition import PCA #PCA

import cv2



import os # accessing directory structure
#path to the dataset..

DATASET_PATH = "/kaggle/input/fashion-product-images-small/myntradataset/"

print(os.listdir(DATASET_PATH))
#read the csv file with the details. find the file name by appending .jpg to the image id

df = pd.read_csv(DATASET_PATH + "styles.csv", error_bad_lines=False)

df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)

#shuffle the dataframe

df = df.sample(frac=1).reset_index(drop=True)

df.head(10)
images = []

rowstoberemoved = []



#code to read the images based on the dataframe

for img_id in range(len(df['id'])):

    img_path = DATASET_PATH + 'images/' + str(df.loc[img_id,"image"])

    #read the image

    img = cv2.imread(img_path)

    try:

        #resize to the required size

        img = cv2.resize(img, (28,28)) 

        #flatten the image

        img = img.flatten()

    except:

        #remove the row corresponding to the image with error

        rowstoberemoved.append(img_id)

        continue

    img = pd.Series(img,name=img_path)

    images.append(img)

#drop rows with errors

df = df.drop(df.index[rowstoberemoved])

print("number of proper images:",len(images))
indices = list(set(df["masterCategory"]))



#convert categorical to class numbers(sklearn handles categories given as numbers)

ylabels = np.asarray([indices.index(i) for i in df["masterCategory"]])

ylabels_onehot = []

for i in ylabels:

    ylabels_onehot.append([0 for i in range(len(indices))])

    ylabels_onehot[-1][i] = 1

ylabels_onehot = np.asarray(ylabels_onehot)

print(ylabels_onehot[0])
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from tqdm import tqdm

KNN_Scores = []

ANN_Scores = []

KNN_train_scores = []

ANN_train_scores = []

#for each value of number of components 

for i in tqdm(range(2, 6)):

    #perform pca

    pca = PCA(n_components  = i)

    pca.fit(images)

    

    #take only the required rows

    new_images = pca.transform(images)

    

    #split into train and test set

    X_train, X_test, y_train, y_test = train_test_split(new_images, ylabels, test_size=0.2)

    

    #scale values for better fit

    scaler = StandardScaler()

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)

    

    #fit knn

    model = KNeighborsClassifier(n_neighbors=7)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    y_train_pred = model.predict(X_train)

    KNN_Scores.append(accuracy_score(y_test, y_pred))

    KNN_train_scores.append(accuracy_score(y_train, y_train_pred))

    

    #fit ann

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,8,8,4), random_state = 4)

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    y_train_pred = clf.predict(X_train)

    ANN_train_scores.append(accuracy_score(y_train, y_train_pred))

    ANN_Scores.append(accuracy_score(y_test, y_pred))

print("The accuracies for c = 2 to 5 for kNN on test set are",KNN_Scores)

print("The accuracies for c = 2 to 5 for ANN on test set are",ANN_Scores)

print("The accuracies for c = 2 to 5 for kNN on train set are",KNN_train_scores)

print("The accuracies for c = 2 to 5 for ANN on train set are",ANN_train_scores)
import matplotlib.pyplot as plt

import librosa.display

audio_path = '/kaggle/input/audio-data/The Big Bang Theory Season 6 Ep 21 - Best Scenes.wav'

x , sr = librosa.load(audio_path)
x , sr = librosa.load(audio_path)

print(type(x), type(sr))

librosa.load(audio_path, sr=None)

#display waveform





plt.figure(figsize=(30, 10))

plt.xlabel("Time")

plt.title("Waveform of the given audio file")

librosa.display.waveplot(x, sr=sr)
mfccs = librosa.feature.mfcc(x, sr=sr)

print(mfccs.shape)

#Displaying  the MFCCs:

librosa.display.specshow(mfccs, sr=sr, x_axis='time')
#Zero Crossing rate

zc = librosa.feature.zero_crossing_rate(x)

print("The dimension of ZCR is", sum(zc).shape)

plt.figure(figsize = (20,10))

plt.plot(sum(zc))

plt.xlabel("Time")

plt.ylabel("Zero crossing Rate")

plt.title("Zero crossing rate of the given audio clip")
cent = librosa.feature.spectral_centroid(x, sr=sr)

plt.figure(figsize = (20,10))

plt.plot(sum(cent))

plt.xlabel("time")

plt.ylabel("Spectral Centroid")

plt.title("Spectral centroids of the audio segment")
pitches, magnitudes = librosa.core.piptrack(x, sr=sr)

pitches, magnitudes = librosa.piptrack(y=x, sr=sr)

print("shape of pitch:",pitches.shape)

a=plt.plot(sum(pitches))

plt.xlabel("Time")

plt.ylabel("Pitch")

plt.title("Pitch of the given audio segment")
rootMeanSquare=librosa.feature.rms(y=x)

print("Dimensions of root mean square feature are :",rootMeanSquare.shape)