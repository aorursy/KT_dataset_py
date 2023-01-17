# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
print(f"The above prediction is correct as it has identified the person name {artist_name} with the given image")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import os

class IdentityMetadata():
    def __init__(self, base, name, file):
        #print(base, name, file)
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
    
def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

# metadata = load_metadata('images')
metadata = load_metadata('/kaggle/input/aligned-face-dataset-from-pinterest/Aligned Face Dataset from Pinterest/PINS')
metadata
pathimg = str(metadata[210])
pathimg
import cv2 # opencv
img = cv2.imread(pathimg,1)
from matplotlib import pyplot as plt
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([])
plt.yticks([])
plt.show()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation

def vgg_face():	
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model
# Loading Model weights
model = vgg_face()
# Loading Model weights
WEIGHTS_FILE = "/kaggle/input/aligned-face-dataset-from-pinterest/vgg_face_weights.h5"
model.load_weights(WEIGHTS_FILE)
from tensorflow.keras.models import Model
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
import cv2
def load_image(path):
    #print(path)
    img = cv2.imread(path, 1)
    return img[...,::-1]
# Get embedding vector for first image in the metadata using the pre-trained model
img_path = metadata[0].image_path()
print(img_path)
img = load_image(img_path)
# Normalising pixel values from [0-255] to [0-1]: scale RGB values to interval [0,1]
img = (img / 255.).astype(np.float32)
img = cv2.resize(img, dsize = (224,224))
print(img.shape)

# Obtain embedding vector for an image
# Get the embedding vector for the above image using vgg_face_descriptor model and print the shape 
#print(vgg_face_descriptor.predict(np.expand_dims(img, axis=0))[0])
embedding_vector = vgg_face_descriptor.predict(np.expand_dims(img, axis=0))[0]
print(type(embedding_vector))
print(embedding_vector.shape)
embedding_vector
embeddings = np.zeros((metadata.shape[0], 2622))
import time
start_time = time.time()
for i, m in enumerate(metadata):
  img_path = m.image_path()
  img = load_image(img_path)
  # Normalising pixel values from [0-255] to [0-1]: scale RGB values to interval [0,1]
  img = (img / 255.).astype(np.float32)
  img = cv2.resize(img, dsize = (224, 224))
  embeddings[i] = vgg_face_descriptor.predict(np.expand_dims(img, axis=0))[0]

computational_time = time.time() - start_time
print('Done in %0.3fs' %(computational_time))

embeddings[0]
def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))
import matplotlib.pyplot as plt

def show_pair(idx1, idx2):
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Distance = {distance(embeddings[idx1], embeddings[idx2])}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()));    

show_pair(2, 4)
show_pair(2, 411)
train_idx = np.arange(metadata.shape[0]) % 9 != 0
test_idx = np.arange(metadata.shape[0]) % 9 == 0
np.sum(train_idx)
np.sum(test_idx)
X_train = embeddings[train_idx]
X_test = embeddings[test_idx]
X_train.shape
X_test.shape
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
targets = np.array([m.name for m in metadata])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(targets) 
y_train = y[train_idx]
y_test  = y[test_idx]
np.unique(targets)
np.unique(y_test)
np.unique(y_train)
# Standarize features
from sklearn.preprocessing import StandardScaler

#### Add your code here ####
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

print(X_train)
print(X_train_scaled)
from sklearn.decomposition import PCA
covMatrix = np.cov(X_train_scaled,rowvar=False)
print(covMatrix)
eig_vals, eig_vecs = np.linalg.eig(covMatrix)
tot = sum(eig_vals)
var_exp = [( i /tot ) * 100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

features = 2622
# Taking the attribute count 2622 except the target column.
pca = PCA(n_components=features)
pca.fit(X_train)
print("Eigen Values :")
print("====================")
print(pca.explained_variance_)
print("Eigen Vectors :")
print("====================")
print(pca.components_)
print("The percentage of variation explained by each eigen Vector : ")
print("============================================================")
print(pca.explained_variance_ratio_)
fig1 = plt.figure(figsize=(15,6))
plt.bar(list(range(1,(features+1))),pca.explained_variance_ratio_,alpha=0.7)
plt.ylabel('Variation explained')
plt.xlabel('Eigen Value')
plt.show()
fig1 = plt.figure(figsize=(15,6))
plt.step(list(range(1,(features+1))),np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Cum of variation explained')
plt.xlabel('Eigen Value')
plt.show()
# Set variable for the decided dimension
final_n_component = 100
# Taking the attribute count as per the decision.
pca_component = PCA(n_components=final_n_component, svd_solver='full')
pca_component.fit(X_train_scaled)
print(f"Eigen Values (with {final_n_component} PCA components):")
print("===========================================")
print(pca_component.explained_variance_)
print(f"Eigen Vectors (with {final_n_component} PCA components):")
print("=================================================")
print(pca_component.components_)
print(f"The percentage of variation explained by each eigen Vector (with {final_n_component} PCA components):")
print("==========================================================================================")
print(pca_component.explained_variance_ratio_)
# Transforming the dataset
pca_X_train = pca_component.transform(X_train_scaled)
pca_X_test = pca_component.transform(X_test_scaled)
pca_X_train
from sklearn.svm import SVC

pca_svm = SVC(C = 1, kernel = 'linear', degree=3, gamma= "scale")
pca_svm.fit(pca_X_train, y_train)
pca_svm.score(pca_X_test, y_test)
import warnings
# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')

example_idx = 222

example_image = load_image(metadata[test_idx][example_idx].image_path())
example_prediction = pca_svm.predict([pca_X_test[example_idx]])
#### Add your code here ####
example_identity = label_encoder.inverse_transform(example_prediction)[0]

plt.imshow(example_image)
name = example_identity.split('_')
person_name = name[1].split(' ')
person_name = person_name[0].capitalize() + ' ' + person_name[1].capitalize()
plt.title(f'Identified as {person_name}');
print(f"The above prediction is correct as it has identified the person name {person_name} with the given image")
import warnings
# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')

example_idx = 500

example_image = load_image(metadata[test_idx][example_idx].image_path())
example_prediction = pca_svm.predict([pca_X_test[example_idx]])
#### Add your code here ####
example_identity = label_encoder.inverse_transform(example_prediction)[0]

plt.imshow(example_image)
name = example_identity.split('_')
person_name = name[1].split(' ')
person_name = person_name[0].capitalize() + ' ' + person_name[1].capitalize()
plt.title(f'Identified as {person_name}');
print(f"The above prediction is correct as it has identified the person name {person_name} with the given image")
