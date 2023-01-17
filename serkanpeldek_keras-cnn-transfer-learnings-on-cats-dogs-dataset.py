import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from keras import layers

from keras import models

from keras import optimizers

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from keras import applications



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



import glob

import os

print("Cats&Dogs Dataset Folder Contain:",os.listdir("../input"))
import zipfile





zip_files = ['test1', 'train']

# Will unzip the files so that you can see them..

for zip_file in zip_files:

    with zipfile.ZipFile("../input/{}.zip".format(zip_file),"r") as z:

        z.extractall(".")

        print("{} unzipped".format(zip_file))
print(os.listdir('../input'))
IMAGE_FOLDER_PATH="../working/train"

FILE_NAMES=os.listdir(IMAGE_FOLDER_PATH)

WIDTH=150

HEIGHT=150
targets=list()

full_paths=list()

for file_name in FILE_NAMES:

    target=file_name.split(".")[0]

    full_path=os.path.join(IMAGE_FOLDER_PATH, file_name)

    full_paths.append(full_path)

    targets.append(target)



dataset=pd.DataFrame()

dataset['image_path']=full_paths

dataset['target']=targets
dataset.head(10)
target_counts=dataset['target'].value_counts()

print("Number of dogs in the dataset:{}".format(target_counts['dog']))

print("Number of cats in the dataset:{}".format(target_counts['cat']))
def get_side(img, side_type, side_size=5):

    height, width, channel=img.shape

    if side_type=="horizontal":

        return np.ones((height,side_size,  channel), dtype=np.float32)*255

        

    return np.ones((side_size, width,  channel), dtype=np.float32)*255



def show_gallery(show="both"):

    n=100

    counter=0

    images=list()

    vertical_images=[]

    rng_state = np.random.get_state()

    np.random.shuffle(full_paths)

    np.random.set_state(rng_state)

    np.random.shuffle(targets)

    for path, target in zip(full_paths, targets):

        if target!=show and show!="both":

            continue

        counter=counter+1

        if counter%100==0:

            break

        #Image loading from disk as JpegImageFile file format

        img=load_img(path, target_size=(WIDTH,HEIGHT))

        #Converting JpegImageFile to numpy array

        img=img_to_array(img)

        

        hside=get_side(img, side_type="horizontal")

        images.append(img)

        images.append(hside)



        if counter%10==0:

            himage=np.hstack((images))

            vside=get_side(himage, side_type="vertical")

            vertical_images.append(himage)

            vertical_images.append(vside)

            

            images=list()



    gallery=np.vstack((vertical_images)) 

    plt.figure(figsize=(12,12))

    plt.xticks([])

    plt.yticks([])

    title={"both":"Dogs and Cats",

          "cat": "Cats",

          "dog": "Dogs"}

    plt.title("100 samples of {} of the dataset".format(title[show]))

    plt.imshow(gallery.astype(np.uint8))
show_gallery(show="cat")

show_gallery(show="dog")

show_gallery(show="both")
def show_model_history(modelHistory, model_name):

    history=pd.DataFrame()

    history["Train Loss"]=modelHistory.history['loss']

    history["Validatin Loss"]=modelHistory.history['val_loss']

    history["Train Accuracy"]=modelHistory.history['accuracy']

    history["Validatin Accuracy"]=modelHistory.history['val_accuracy']

  

    history.plot(figsize=(12,8))

    plt.title(" Convulutional Model {} Train and Validation Loss and Accuracy History".format(model_name))

    plt.show()
model=models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(WIDTH, HEIGHT, 3)))

model.add(layers.Conv2D(32, (3,3), activation="relu"))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Dropout(0.25))



model.add(layers.Conv2D(64, (3,3), activation="relu"))

model.add(layers.Conv2D(64, (3,3), activation="relu"))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(2,2))

model.add(layers.Dropout(0.25))



model.add(layers.Conv2D(128, (3,3), activation="relu"))

model.add(layers.Conv2D(128, (3,3), activation="relu"))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Dropout(0.25))



model.add(layers.Conv2D(64, (3,3), activation="relu"))

model.add(layers.Conv2D(64, (3,3), activation="relu"))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Dropout(0.25))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation="relu"))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation="sigmoid"))

model.summary()
model.compile(loss="binary_crossentropy", 

             optimizer=optimizers.RMSprop(lr=1e-4),

             metrics=["accuracy"])

print("[INFO]: model compiled...")
dataset_train, dataset_test=train_test_split(dataset,

                                                 test_size=0.2,

                                                 random_state=42)
train_datagen=ImageDataGenerator(

rotation_range=15,

rescale=1./255,

shear_range=0.1,

zoom_range=0.2,

horizontal_flip=True,

width_shift_range=0.1,

height_shift_range=0.1)



train_datagenerator=train_datagen.flow_from_dataframe(dataframe=dataset_train,

                                                     x_col="image_path",

                                                     y_col="target",

                                                     target_size=(WIDTH, HEIGHT),

                                                     class_mode="binary",

                                                     batch_size=150)
test_datagen=ImageDataGenerator(rescale=1./255)

test_datagenerator=test_datagen.flow_from_dataframe(dataframe=dataset_test,

                                                   x_col="image_path",

                                                   y_col="target",

                                                   target_size=(WIDTH, HEIGHT),

                                                   class_mode="binary",

                                                   batch_size=150)
modelHistory=model.fit_generator(train_datagenerator,

                                epochs=50,

                                validation_data=test_datagenerator,

                                validation_steps=dataset_test.shape[0]//150,

                                steps_per_epoch=dataset_train.shape[0]//150

                                )
print("Train Accuracy:{:.3f}".format(modelHistory.history['accuracy'][-1]))

print("Test Accuracy:{:.3f}".format(modelHistory.history['val_accuracy'][-1]))

show_model_history(modelHistory=modelHistory, model_name="")
model=applications.VGG16(weights="imagenet", include_top=False, input_shape=(WIDTH, HEIGHT, 3))

model.summary()
counter=0

features=list()

for path, target in zip(full_paths, targets):

    img=load_img(path, target_size=(WIDTH, HEIGHT))

    img=img_to_array(img)

    img=np.expand_dims(img, axis=0)

    feature=model.predict(img)

    features.append(feature)

    counter+=1

    if counter%2500==0:

        print("[INFO]:{} images loaded".format(counter))



features=np.array(features)

print("Before reshape,features.shape:",features.shape)

features=features.reshape(features.shape[0], 4*4*512)

print("After reshape, features.shape:",features.shape)
le=LabelEncoder()

targets=le.fit_transform(targets)
print("features.shape:",features.shape)

print("targets.shape:",targets.shape)
X_train, X_test, y_train, y_test=train_test_split(features, targets, test_size=0.2, random_state=42)
from sklearn.model_selection import cross_val_score
clf=LogisticRegression(solver="lbfgs")

print("{} training...".format(clf.__class__.__name__))

clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)

print("The model trained and used to predict the test data...")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, y_pred))

print("Classification Report:\n",metrics.classification_report(y_test, y_pred, target_names=["cat", "dog"]))
cv_scores=cross_val_score(LogisticRegression(solver="lbfgs"), features, targets, cv=3 )

print("Cross validation scores obtained...")
print("Cross validated scores:{}".format(cv_scores))

print("Mean of cross validated scores:{:.3f}".format(cv_scores.mean()))