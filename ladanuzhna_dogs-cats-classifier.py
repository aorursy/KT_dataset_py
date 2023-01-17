import zipfile


zip_files = ['test1', 'train']
# Will unzip the files so that you can see them..
for zip_file in zip_files:
    with zipfile.ZipFile("../input/dogs-vs-cats/{}.zip".format(zip_file),"r") as z:
        z.extractall(".")
        print("{} unzipped".format(zip_file))
import os
import pandas as pd

filenames = os.listdir("/kaggle/working/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(df['filename'], df['category'], test_size=0.20)

import shutil

current_dir = os.path.abspath(os.getcwd())
training_dir = os.path.join(current_dir, "training")
os.mkdir(training_dir)
cats_dir = os.path.join(training_dir, "cats")
dogs_dir = os.path.join(training_dir, "dogs")
os.mkdir(cats_dir)
os.mkdir(dogs_dir)

for i in range(len(X_train)):
    filename = X_train.iloc[i]
    folder = cats_dir if y_train.iloc[i] == 0 else dogs_dir
    dst = os.path.join(folder,filename)
    os.rename("/kaggle/working/train/{}".format(filename), dst)
validation_dir = training_dir = os.path.join(current_dir, "validation")
os.mkdir(validation_dir)
cats_dir_v = os.path.join(validation_dir, "cats")
dogs_dir_v = os.path.join(validation_dir, "dogs")
os.mkdir(cats_dir_v)
os.mkdir(dogs_dir_v)

for i in range(len(X_valid)):
    filename = X_valid.iloc[i]
    folder = cats_dir_v if y_valid.iloc[i] == 0 else dogs_dir_v
    dst = os.path.join(folder,filename)
    os.rename("/kaggle/working/train/{}".format(filename), dst)
shutil.rmtree("train")
from keras import layers 
from keras import models
from keras import optimizers 

model = models.Sequential() 
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))) 
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Conv2D(128, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Conv2D(128, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Flatten()) 
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(512, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])


model.summary()
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255) 
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory( training_dir, target_size=(150, 150),batch_size=20, class_mode='binary')
validation_generator = test_datagen.flow_from_directory( validation_dir,target_size=(150, 150), batch_size=20, class_mode='binary')

train_datagen = ImageDataGenerator( rescale=1./255, rotation_range=40, width_shift_range=0.2, 
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255) #validation data shouldn't be augmented!!!
train_generator = train_datagen.flow_from_directory( training_dir, target_size=(150, 150), batch_size=32, class_mode='binary') 
#Because you use binary_crossentropy loss, you need binary labels.
validation_generator = test_datagen.flow_from_directory( validation_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
history = model.fit_generator( train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=50)

model.save_weights("model.h5")
test_dir = os.path.join(current_dir, 'test1')
test_gen = ImageDataGenerator(rescale=1./255)
test_filenames = os.listdir(test_dir)
dst_dir = os.path.join(test_dir,'sub')
os.mkdir(dst_dir)
#flowfromdirectory treats the sub-folder as the label of the images it contains. So it only search images in sub-folders.
for filename in test_filenames:
    dst = os.path.join(dst_dir,filename)
    src = os.path.join(test_dir,filename)
    os.rename(src, dst)


test_generator = test_gen.flow_from_directory(directory=test_dir, target_size=(150, 150), batch_size=32)
predict = model.predict_generator(test_generator, 100)
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = pd.DataFrame(predict)
submission_df.drop(['filename'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)