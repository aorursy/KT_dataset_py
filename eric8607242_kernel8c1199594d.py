import os



import numpy as np 

import pandas as pd 



import PIL.Image



from tensorflow.python.keras.models import Model

from tensorflow.python.keras.models import load_model

from tensorflow.python.keras.layers import Flatten, Dense, Dropout, BatchNormalization

from tensorflow.python.keras.applications.resnet50 import ResNet50

from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator







TRAIN_DIR = "../input/flower/flower_classification/train/"

TEST_DIR = "../input/flower/flower_classification/test/"

CLASSES = "../input/flower/flower_classification/mapping.csv"

SUBMISSION = "./submission.csv"



WEIGHTS_FINAL = "./model-resnet50-final.h5"



INPUT_SIZE = 224

FREEZE_LAYERS = 2
classes = pd.read_csv(CLASSES)

classes.head()
imgs_path = []

labels = []

for i, row in classes.iterrows():

    data_dir = os.path.join(TRAIN_DIR, row["dirs"])

    img_path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    label = [row["class"] for l in range(len(img_path))]

    

    imgs_path.extend(img_path)

    labels.extend(label)
train_X = []



for img_path in imgs_path:

    image = PIL.Image.open(img_path)

        

    image = image.convert('RGB')

    image = image.resize((INPUT_SIZE, INPUT_SIZE), PIL.Image.ANTIALIAS)

            

    train_X.append(np.array(image))
train_X = np.asarray(train_X)

train_Y = np.asarray(labels)



idx = np.arange(train_X.shape[0])

np.random.shuffle(idx)



train_X = train_X[idx]

train_Y = train_Y[idx]



print(train_X.shape)

print(train_Y.shape)
train_datagen = ImageDataGenerator(rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   channel_shift_range=10,

                                   horizontal_flip=True,

                                   fill_mode='nearest')
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,

               input_shape=(INPUT_SIZE, INPUT_SIZE, 3))

x = net.output

x = Flatten()(x)



x = Dropout(0.5)(x)

x = Dense(200, activation='relu', name='dense1')(x)

x = BatchNormalization()(x)

x = Dense(200, activation='relu', name='dense2')(x)

x = BatchNormalization()(x)

output_layer = Dense(5, activation='softmax', name='softmax')(x)



net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:

    layer.trainable = False

for layer in net_final.layers[FREEZE_LAYERS:]:

    layer.trainable = True



net_final.compile(optimizer=Adam(lr=5e-5, decay=0.005),

                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(net_final.summary())
net_final.fit_generator(train_datagen.flow(train_X, train_Y, batch_size=64), 

                        steps_per_epoch=len(train_X) / 64,

                        epochs=15)
net_final.save(WEIGHTS_FINAL)

# net_final = load_model('model-resnet50-final.h5')
test_imgs_path = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if os.path.isfile(os.path.join(TEST_DIR, f))]
test_X = []

for img_path in test_imgs_path:

    image = PIL.Image.open(img_path)

        

    image = image.convert('RGB')

    image = image.resize((INPUT_SIZE, INPUT_SIZE), PIL.Image.ANTIALIAS)

    

    test_X.append(np.asarray(image))
test_X = np.asarray(test_X)

pred = net_final.predict(test_X)
pred_label = np.argmax(pred, axis=1)

print(pred_label.shape)
import csv



with open(SUBMISSION, 'w', newline='') as csvFile:

    writer = csv.writer(csvFile)

    

    writer.writerow(['id', 'class'])

    for index in range(pred_label.shape[0]):

        file_name = test_imgs_path[index][43:-4]

        writer.writerow([file_name, pred_label[index]])