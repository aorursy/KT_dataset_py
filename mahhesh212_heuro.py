!pip install ../input/mctnn-package/mtcnn-0.1.0-py3-none-any.whl
from os import listdir
import os.path 
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import re
import numpy as np
from numpy import load
detector = MTCNN()
from os import listdir
faces=[]
d="../input/utkfacedata/UTKFace/"
c=1
y=[]
g=[]
r=[]
for filename in listdir(d):
    if(c%200==0):
        print(c)
    filename1= d+filename
    image = Image.open(filename1)
    image = image.convert('RGB')
    pixels = asarray(image)
    results = detector.detect_faces(pixels)
    if(any(results)==False):
        continue
    y.append(int(re.split('_+', filename)[0]))
    g.append(int(re.split('_+', filename)[1]))
    r.append(int(re.split('_+', filename)[2]))
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize((160, 160))
    faces.append(asarray(image))
    c=c+1
g= np.array(g)
r= np.array(r)
cc=np.concatenate((X, g[:, None]), axis=1)
X = np.concatenate((cc,r[:, None]), axis=1)
from numpy import load
data = load('../input/racegender/include.npz')
X, y = data['arr_0'], data['arr_1']

np.shape(y)
#Onehotencode_y
import numpy as np
y=y//5
b = np.zeros((y.size, y.max()+1))
b[np.arange(y.size),y] = 1
np.shape(b)
from numpy import savez_compressed
savez_compressed('include.npz', X,y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, b, test_size=0.1, random_state=42)

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
in_encoder = Normalizer(norm='l2')
X_train = in_encoder.transform(X_train)
X_test = in_encoder.transform(X_test)
np.shape(b)
from keras.models import Sequential
from keras.layers import Dense
trick = Sequential()
trick.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu', input_dim = 130))
trick.add(Dense(output_dim = 32,  activation = 'relu'))
trick.add(Dense(output_dim = 8,  activation = 'relu'))
trick.add(Dense(output_dim = 4,  activation = 'relu'))
trick.add(Dense(output_dim = 24, activation = 'softmax'))
trick.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
trick.fit(X_train, y_train, batch_size = 64, nb_epoch = 300)

np.argmax(y_train, axis=1)
yHat_test = trick.predict(X_test)
yHat_train = trick.predict(X_train)

yHat_train=(np.argmax(yHat_train, axis=1))
yHat_test=(np.argmax(yHat_test, axis=1))
y_train1=(np.argmax(y_train, axis=1))
y_test1=(np.argmax(y_test, axis=1))


score_train = accuracy_score(yHat_train, y_train1)
score_test = accuracy_score(yHat_test, y_test1)
print(score_train)
print(score_test)



faces=np.array(faces)
y= np.array(y)
y=y.toList()
faces=faces.toList()
from keras import models   
modelo = models.load_model('../input/facenetoo/faceneto_keras.h5')
print('Loaded Model')
from numpy import savez_compressed
savez_compressed('final.npz', faces,y)
from numpy import load
data = load('../input/npzzzz/final.npz')
trainX, trainy = data['arr_0'], data['arr_1']
trainy=trainy.astype(int)
from numpy import savez_compressed
savez_compressed('try1.npz', trainX,trainy)
import numpy as np
X=[]
c=1
for face_pixels in trainX:
    if(c%200==0):
        print(c)
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = modelo.predict(samples)
    X.append(yhat[0])
    c=c+1
X = np.array(X) 
from numpy import savez_compressed
savez_compressed('smh128.npz', X,trainy)
np.shape(g)


#Gradient Boosting
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn import datasets, ensemble

params = {'n_estimators': 1000,
          'max_depth': 5,
          'min_samples_split': 5,
          'learning_rate': 0.1,
          'loss': 'ls'}
reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
yHat_test = reg.predict(X_test)//10
yHat_train = reg.predict(X_train)//10
y_train1= y_train//10
y_test1 = y_test//10

score_train = accuracy_score(yHat_train, y_train1)
score_test = accuracy_score(yHat_test, y_test1)
print(score_train)
print(score_test)


y_train
detector = MTCNN()
from PIL import Image
from numpy import asarray
#Individual checker
data= "../input/utktrail/Trial/34_0_1_20170112214929038.jpg.chip.jpg"
y1=1
image = Image.open(data)
image = image.convert('RGB')
pixels = asarray(image)
results = detector.detect_faces(pixels)
x1, y1, width, height = results[0]['box']
x1, y1 = abs(x1), abs(y1)
x2, y2 = x1 + width, y1 + height
face = pixels[y1:y2, x1:x2]
# resize pixels to the model size
image = Image.fromarray(face)
image = image.resize((160, 160))
face_pixels = asarray(image)
from numpy import expand_dims
face_pixels = face_pixels.astype('float32')
# standardize pixel values across channels (global)
mean, std = face_pixels.mean(), face_pixels.std()
face_pixels = (face_pixels - mean) / std
# transform face into one sample
samples = expand_dims(face_pixels, axis=0)
Xactual = modelo.predict(samples)
print(reg.predict(Xactual))
import numpy as np
np.shape(trainX)
#Split for NN Alone
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trainX, trainy, test_size=0.2, random_state=84)
y_train=y_train//10
y_test = y_test//10

#NN
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Conv2D(64, kernel_size=4, activation='relu', input_shape=(160,160,3)))
# Adding the second hidden layer
classifier.add(Conv2D(8, kernel_size=2, activation='relu'))
classifier.add(Conv2D(4, kernel_size= 1, activation='relu'))
classifier.add(Flatten())
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'linear'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 20, nb_epoch = 100)

y1 = classifier.predict(X_train) //10
y2= classifier.predict(X_test) //10
ytr= y_train//10
yte = y_test//10
score_train = accuracy_score(y1, ytr)
score_test = accuracy_score(y2, yte)
print(score_train)
print(score_test)

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
model.fit(X_train, y_train)

yhat_train = classifier.predict(trainX)
#yhat_test = classifier.predict(X_test)
# score

from sklearn.metrics import accuracy_score
yhato_train =yhat_train//10
y1=trainy//10
score_train = accuracy_score(yhato_train,y1)
print(score_train)
yhato_train =yhat_train//10
yhato_test = yhat_test//10
y1=y_train//10
y2= y_test//10
score_train = accuracy_score(y1, yhato_train)
score_test = accuracy_score(y2, yhato_test)
print(score_train)
print(score_test)

