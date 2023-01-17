# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from sklearn import decomposition

from sklearn.svm import LinearSVC, SVC



np.random.seed(64)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = (pd.read_csv("../input/test.csv").values).astype('float32')
train.head()
test.shape
train_label = train['label']

train_label = to_categorical(train_label)

classes = train_label.shape[1]

classes
train_image = (train.ix[:,1:].values).astype('float32')
train_image.shape
#train_image = train_image.reshape(train_image.shape[0], 28, 28)
train_image = train_image / 255

test = test / 255
from sklearn import decomposition



pca = decomposition.PCA(n_components = 784)

#pca.fit(train_image)

#plt.plot(pca.explained_variance_ratio_)
pca = decomposition.PCA(n_components = 100)

#pca.fit(train_image)

#plt.plot(pca.explained_variance_ratio_)
#pca = decomposition.PCA(n_components = 40)

#pca.fit(train_image)

#train_pca = np.array(pca.transform(train_image))

#test_pca = np.array(pca.transform(test))
from keras.models import Sequential

from keras.layers import Dense , Dropout



model=Sequential()

model.add(Dense(32,activation='relu',input_dim=(40)))

model.add(Dense(16,activation='relu'))

model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#model_fit = model.fit(train_pca, train_label, validation_split = 0.05, epochs=24, batch_size=64)
#predictions = model.predict_classes(test_pca, verbose=0)

#result = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})

#result.to_csv("output.csv", index=False, header=True)
#train_image = train_image / 255

#test = test / 255
from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop



model=Sequential()

model.add(Dense(32,activation='relu',input_dim=(28 * 28)))

model.add(Dense(16,activation='relu'))

model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
train_image.shape
#model_fit = model.fit(train_image, train_label, validation_split = 0.05, epochs=24, batch_size=64)
#predictions = model.predict_classes(test, verbose=0)

#result = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})

#result.to_csv("output.csv", index=False, header=True)
from keras import backend



backend.image_data_format()
train = pd.read_csv("../input/train.csv")

test = (pd.read_csv("../input/test.csv").values).astype('float32')
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28, 28, 1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(classes, activation='softmax'))
#train_x = train.values[:,1:].reshape(train.shape[0], 28, 28, 1).astype('float32') / 255

#train_y = to_categorical(train.values[:, 0], 10)



#test_x = test.reshape(test.shape[0], 28, 28, 1).astype('float32') / 255
##model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#model_fit = model.fit(train_x, train_y, batch_size=128, epochs=4)
#predictions = model.predict_classes(test_x)

#result = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
#result
#result.to_csv("output.csv", index=False, header=True)
#train_pca.shape
#test_pca.shape
#train_label.shape
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=4,algorithm='auto',n_jobs=10)
#clf.fit(train_pca, train_label)
#predictions = clf.predict(test_pca)
#pred = np.argmax(predictions, 1)

#pred
#result = pd.DataFrame({"ImageId": list(range(1,len(pred)+1)),"Label": pred})
##result.to_csv("output.csv", index=False, header=True)
clf = LinearSVC(C=0.05, 

                fit_intercept=True, 

                class_weight='balanced',

                multi_class='crammer_singer',

                dual=False,

                random_state=42)
train_label = train['label']

#clf.fit(train_pca, train_label)
#predictions = clf.predict(test_pca)

#predictions
#result = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})

#result.to_csv("output.csv", index=False, header=True)
clf = SVC(C=12,

          kernel='rbf',

          class_weight='balanced',

          random_state=42

         )
#clf.fit(train_pca, train_label)
#predictions = clf.predict(test_pca)

#predictions
#result = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})

#result.to_csv("output.csv", index=False, header=True)
from scipy.ndimage.interpolation import map_coordinates

from scipy.ndimage.filters import gaussian_filter



def elastic_deformation(image, alpha, sigma):

    random_state = np.random.RandomState(None)

    shape = image.shape

    

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')

    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    

    return map_coordinates(image, indices, order=1).reshape(shape)

    
t = train_image.reshape(train_image.shape[0], 28, 28)

train_label = train['label']
t[0].shape
for i in range(4000):

    img = elastic_deformation(t[i], 36, 8)

    img = img[np.newaxis, :, :]

    t = np.concatenate((t, img))
t.shape
labels = train_label.append(train_label[0:4000], ignore_index = True)
for i in range(6, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(t[i], cmap=plt.get_cmap('gray'))

    plt.title(labels[i]);
for i in range(42006, 42009):

    plt.subplot(330 + (i+1) - 42000)

    plt.imshow(t[i], cmap=plt.get_cmap('gray'))

    plt.title(labels[i]);
temp = t.reshape(t.shape[0], -1)
pca = decomposition.PCA(n_components = 40)

pca.fit(temp)

train_pca = np.array(pca.transform(temp))

pca.fit(test)

test_pca = np.array(pca.transform(test))
labels.shape
t.shape
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=4,algorithm='auto',n_jobs=10)
clf.fit(train_pca, labels)
predictions = clf.predict(test_pca)

predictions
result = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})

result.to_csv("output.csv", index=False, header=True)