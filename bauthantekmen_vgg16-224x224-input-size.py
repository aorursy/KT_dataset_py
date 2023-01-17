from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model #import functional api
from pickle import dump

class Build_Model():
    def init(self):
        pass

    def build(self):
        model = VGG16()
        model.layers.pop()
        last_tensor = model.layers[-1].output
        output = Dense(2 , activation="softmax")(last_tensor)
        model = Model(inputs=model.inputs, outputs=output)

        #Freeze feature layers
        for layer in model.layers[:-3]:
            layer.trainable = False
          
        print("[Info] model build is completed")
        model.summary()

        model.save("model.h5")
        print("[Info] Saved model to disk")
        
if __name__ == "__main__":
    model = Build_Model()
    model.build()
import os
from tqdm import tqdm
import cv2
import numpy as np

DATADIR = "../input/gaze-locking-interpreted-from-columbia-gaze/DataDir-20200408T153703Z-001/DataDir"

input_size = (224,224)

def create_training_array(data_path,input_size):
  data = []
  for category in os.listdir(data_path): 
    path = (os.path.join(data_path,category))
    class_num = 1 if category == "looking" else 0;
    for img in tqdm(os.listdir(path)):
      img_array = cv2.imread(os.path.join(path,img))  # convert to array
      new_array = cv2.resize(img_array, input_size)  # resize to normalize data size
      data.append([new_array, class_num-1])  # add this to our training_data
      
  X = []
  y = []
  for features,label in data:
    X.append(features)
    y.append(label)

  X = np.array(X).reshape(-1, input_size[0], input_size[0], 3) #-1 sizeım belli değil diyor yani sample sayısı belli değil
  print("[I] Saving data arrays as .npy file")
  np.save("X.npy", X)
  np.save("y.npy", y)

if __name__ == "__main__":
  create_training_array(DATADIR, input_size)
  print("[I] Arrays saved succesfully!")
from keras.applications.vgg16 import preprocess_input #255'e scale etmeden preprocesse atıyoruz
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

EPOCHS = 450
INIT_LR = 0.00002
BS = 32 #batch_size

X = np.load("X.npy")
y = np.load("y.npy")

X = preprocess_input(X)

(trainX, testX, trainY, testY) = train_test_split(X,
	y, test_size=0.25)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

#optimizer and imagedata generator
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

#load and compile model
print("[INFO] compiling model...")
model = load_model('model_50_0.0001_32.h5')
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS) ###??? bu kısmı nasıl hazırlarız ???
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#start training
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1) #steps per epochs nedir?

model.save("model_{}_{}_{}.h5".format(EPOCHS+50,INIT_LR,BS))

#plot loss and acc
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(".")
#prediction yap
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

import cv2
import numpy as np
import os
#from train import input_size

def decode(arr):
    max_prob = 0
    cls = 0
    for i,prob in enumerate(list(arr[0,])):
        if prob > max_prob:
            max_prob = prob
            cls = i
    return (labels[cls],max_prob)
        
labels = ["bakiyor", "bakmiyor"]

input_size = (224,224)
img_channels = 3

address_prefix = "../input/gaze-locking-interpreted-from-columbia-gaze/VGG16-v1/VGG16-v1/"
model = load_model(address_prefix + "model_50_0.0001_32.h5")

test_path = address_prefix + "Processed Test Set/"

if not(os.path.isdir("./Results/")):
    os.mkdir("Results")
    
for img_name in os.listdir(test_path):
    print(test_path + img_name)
    image = cv2.imread(test_path + img_name)
    
    img_array = cv2.resize(image,input_size)
    img_array = img_array.reshape((1, input_size[0], input_size[1], img_channels))
    img_input = preprocess_input(img_array)
    
    prediction = model.predict(img_input)
    result = decode(prediction)
    
    string = "%{}".format(int(result[1]*100))
    img = cv2.putText(image, string, (5,15), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.5, color = (255,255,255))
    string = result[0]
    img = cv2.putText(image, string, (5,30), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.5, color = (255,255,255))
    cv2.imwrite("./Results/res-" + img_name, img)

print("Succeed!")

#çıkan sonuçları decode et
#def decode predictions(classlist, )
#ımageye çıkan sonuçları sol üstüne yaz
#imageyi kaydet








!apt-get install zip
!zip -r Results.zip Results/
import os
from IPython.display import FileLinks,display
(FileLink(r'Results.zip'))
#console
import numpy as np
a = np.arange(4).reshape((2,2))
print(list(a[1,]))


