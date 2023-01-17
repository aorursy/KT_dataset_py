from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

import pandas as pd 

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random

import os

#print(os.listdir("../content"))

FAST_RUN = False

IMAGE_WIDTH=128

IMAGE_HEIGHT=128

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3
filenames = os.listdir("../input/nnfl-lab-1/training/training")

categories = []

for filename in filenames:

    category = filename.split('_')[0]

    if category == 'chair':

        categories.append(0)

    elif category == 'kitchen':

        categories.append(1)

    elif category == 'knife':

        categories.append(2)

    elif category == 'saucepan':

        categories.append(3)

    else:

        categories.append(-1)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
df.head()
df.tail()
df['category'].value_counts().plot.bar()
sample = random.choice(filenames)

image = load_img("../input/nnfl-lab-1/training/training/"+sample)

plt.imshow(image)
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization 

from tensorflow.python.keras import Sequential 



model = Sequential() 

model.add(Conv2D(64, (3, 3), activation='relu', padding='same',input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))) 

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Dropout(0.25)) 



model.add(Conv2D(64, (3, 3),padding='same', activation='relu')) 

model.add(BatchNormalization()) 

model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Dropout(0.25)) 



model.add(Conv2D(128, (3, 3),padding='same', activation='relu')) 

model.add(BatchNormalization()) 

model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Dropout(0.25)) 



model.add(Conv2D(128, (3, 3),padding='same', activation='relu')) 

model.add(BatchNormalization()) 

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25)) 



model.add(Conv2D(256, (3, 3),padding='same', activation='relu')) 

model.add(BatchNormalization()) 

model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Dropout(0.25)) 



model.add(Conv2D(256, (3, 3),padding='same', activation='relu')) 

model.add(BatchNormalization()) 

model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Dropout(0.25)) 



model.add(Flatten()) 

model.add(Dense(256, activation='relu')) 

model.add(BatchNormalization()) 

model.add(Dropout(0.5))



model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]
callbacks
df['category'].head()
df["category"] = df["category"].replace({0: 'chair', 1: 'kitchen', 2:'knife', 3:'saucepan'}) 
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
train_df['category'].value_counts().plot.bar()
validate_df['category'].value_counts().plot.bar()
total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

batch_size=15
print(total_train)

print(total_validate)
train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    "../input/nnfl-lab-1/training/training", 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    "../input/nnfl-lab-1/training/training", 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
example_df = train_df.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(

    example_df, 

    "../input/nnfl-lab-1/training/training", 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical'

)
plt.figure(figsize=(12, 12))

for i in range(0, 15):

    plt.subplot(5, 3, i+1)

    for X_batch, Y_batch in example_generator:

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
epochs=33

history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
model.save_weights("model.h5")
test_filenames = os.listdir("../input/nnfl-lab-1/testing/testing")

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "../input/nnfl-lab-1/testing/testing", 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    shuffle=False

)
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'chair': 0, 'kitchen': 1, 'knife':2, 'saucepan':3 })
test_df['category'].value_counts().plot.bar()
sample_test = test_df.head(10)

sample_test.head()

plt.figure(figsize=(12, 24))

for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img("../input/nnfl-lab-1/testing/testing/"+filename, target_size=IMAGE_SIZE)

    plt.subplot(6, 3, index+1)

    plt.imshow(img)

    plt.xlabel(filename + '(' + "{}".format(category) + ')' )

plt.tight_layout()

plt.show()
test_df.head()
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename']

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)
import base64 

from IPython.display import HTML 

import pandas as pd 

import numpy as np

def create_download_link(df, title = "Download CSV file",filename = "data.csv"): 

    csv = df.to_csv(index=False) 

    b64 =base64.b64encode(csv.encode()) 

    payload = b64.decode()



    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(submission_df)