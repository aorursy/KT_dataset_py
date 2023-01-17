import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
from tqdm import  tqdm
print(os.listdir("../input"))
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,GlobalAvgPool2D
from keras.layers import Conv2D, MaxPooling2D
import keras
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3
from keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
df = pd.read_csv("../input/labels.csv")
df.info()
sam = pd.read_csv("../input/sample_submission.csv")
sam.head(5)
breed_ls = list(df.groupby('breed').count().sort_values(by='id', ascending=False).index)
import random
import cv2
from keras.preprocessing import image

def getRandomImageList(breed_name, no_of_samples=100):
    global df
    random_images = []
    for index, row in df.iterrows():
        if row['breed'] == breed_name:
            random_images.append(row['id'])
    random_images = random.sample(random_images, no_of_samples)
    return random_images

def readImgResize(name,path):
    img = cv2.imread(path+name)
    img = cv2.resize(img,(128,128))
    return image.img_to_array(img)
INPUT_SIZE = 128
num_class=120
# breedToTrain = breed_ls[num_class]#incase we just want to train top-n freq classes
samples = 65

# There must exist a better way of doing the sampling. This is pretty slow 

image_label = []
num = 0
import tqdm
for i,breed in tqdm.tqdm( enumerate(breed_ls[:num_class])):

    ls = getRandomImageList(breed,samples) 
    image_label.extend(ls)
    
new_df = pd.DataFrame({"id":image_label})
new_df = pd.merge(new_df, df, how='inner', on=['id'])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cat = le.fit_transform(new_df.breed)
from keras.utils.np_utils import to_categorical
mat = to_categorical(cat)


training_data = np.zeros(shape=(len(new_df.id),128,128,3))
for i,j in tqdm.tqdm(enumerate(new_df["id"])):
    training_data[i]=readImgResize(j+".jpg",path="../input/train/")
    


# from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,GlobalAvgPool2D
from keras.layers import Conv2D, MaxPooling2D
import keras
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( training_data, mat, test_size=0.05, random_state=11)
del training_data, mat, new_df, df
import matplotlib.pyplot as plt
plt.figure(figsize=(15,7))
for i in range(6):
    idx = random.randint(0,len(X_train))
    itemindex = np.squeeze(np.where(y_train[idx]==1.)).tolist()
    plt.subplot(2,3,i+1)
    plt.imshow(X_train[idx]/255)
    name = breed_ls[itemindex]
    plt.xlabel(str(name))
    # np.squeeze(itemindex[0]).tolist()
    name
# itemindex
def createModel(nClasses=120):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128,128,3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nClasses, activation='softmax'))
     
    return model
from keras.optimizers import adam
model1 = createModel()
batch_size = 16
epochs = 100

model1.compile(adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
 
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen=ImageDataGenerator(rescale=1./255)
train_set=train_datagen.flow(X_train,y=y_train,batch_size=64)
test_set=test_datagen.flow(X_test,y=y_test,batch_size=64)
model1.fit_generator(train_set,
                      steps_per_epoch = 256,
                      validation_data = test_set,
                      validation_steps = 4,
                      epochs = 25,
                      verbose = 1)

model1.save('my_model.h5')
del train_set, test_set, image_label
del df ,new_df
te = os.listdir("../input/test/")
te_in = np.zeros((len(te),128,128,3))
for num , i in enumerate(te):
    img = readImgResize(i,path="../input/test/")/255
    te_in[num]=img
    

# create submission
pred = model1.predict(te_in) 
submission = pd.DataFrame(pred , columns =le.classes_.tolist())
submission["id"]=[i.split(".")[0] for i in os.listdir("../input/test/")]
submission = submission[["id"]+submission.columns[:-1].tolist()]
submission.to_csv("submission.csv",index = False)