#Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Other Imports
import matplotlib.pyplot as plt
%matplotlib inline

#Keras inputs
from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
#Auflisten der Daten in der Input-Ablage
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# create the training & test sets, skipping the header row with [1:]
train = pd.read_csv("../input/train.csv")
print(train.shape)
test= pd.read_csv("../input/test.csv")
print(test.shape)
X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits (Die zuordnung: z.B. Zahl 1)
X_test = test.values.astype('float32')
#iloc statt ix -> Indexbasiertes Zugreifen
#loc wäre zeichenbasiert

#X_train entspricht: Arrays in Array
#Y_train entspricht: Zahlen (0-9) in Array
#Convert train datset to (num_images, img_rows, img_cols) format 
X_train = X_train.reshape(X_train.shape[0], 28, 28)

#expand 1 more dimention as 1 for colour channel gray
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_train.shape
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_test.shape
#It is important preprocessing step. It is used to centre the data around zero mean and unit variance.

#astype() convertiert alle Values in X_train in den angegebenen Typ
#float32 -> Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
#mean() -> arithm. durchschnitt
#std() -> Standardabweichung
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x): 
    return (x-mean_px)/std_px
#one-hot vector -> For example, 3 would be [0,0,0,1,0,0,0,0,0,0].
#In this case, the nth digit will be represented as a vector which is 1 in the nth dimension.
from keras.utils.np_utils import to_categorical
y_train= to_categorical(y_train)
# fix random seed (number) for reproducibility
seed = 43
np.random.seed(seed)
#Imports
from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D
#Man merke sich den Var-Namen: 'model'!
model= Sequential()
model.add(Lambda(standardize,input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
print("input shape ",model.input_shape)
print("output shape ",model.output_shape)
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001),
 loss='categorical_crossentropy',
 metrics=['accuracy'])
#Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).
from keras.preprocessing import image
gen = image.ImageDataGenerator()
#rain_test_split(*arrays, **options) -> Split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

#use ImageDataGenerator
#flow(x, y=None, batch_size=32) -> Takes numpy data & label arrays, and generates batches of augmented/normalized data (Datenstapel).
batches = gen.flow(X_train, y_train, batch_size=64)
val_batches=gen.flow(X_val, y_val, batch_size=64)
#möge das lernen beginnen ;)
history=model.fit_generator(batches, batches.n, nb_epoch=1, 
                    validation_data=val_batches, nb_val_samples=val_batches.n)
#Optimieren
#Attribute Error -> präzisere Lösungen

model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])
gen = image.ImageDataGenerator()
batches = gen.flow(X_train, y_train, batch_size=64)
history=model.fit_generator(batches, batches.n, nb_epoch=2)
predictions = model.predict_classes(X_test, verbose=0)
#submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
#                         "Label": predictions})
#submissions.to_csv("submission.csv", index=False, header=True)
#Darstellung
def value(v):
    akt_value = -1
    for i in range(0, 10):
        if (v[i] == 1):
            akt_value = i
    return akt_value

anzahl = 1000

def showKorrekt(debug=1):
    global anzahl
    korrekt = 0
    for i in range(0,anzahl):
        ist = predictions[i]
        soll = value(y_train[i])
        if (ist == soll):
            korrekt += 1
        if (debug):
            print ("ImageId:",i+1," Ist:",ist," Soll:",soll," ->",ist == soll)
    if (debug):
        print("")
    print("Anzahl korrekt:",korrekt,"/",anzahl)
    print(korrekt/anzahl*100,"%")
showKorrekt()
#Kurzform
showKorrekt(debug=0)
anzahl = 28000
showKorrekt(debug=0)
model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])
gen = image.ImageDataGenerator()
batches = gen.flow(X_train, y_train, batch_size=64)
history=model.fit_generator(batches, batches.n, nb_epoch=4)
predictions = model.predict_classes(X_test, verbose=0)
showKorrekt(debug=0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("submission.csv", index=False, header=True)