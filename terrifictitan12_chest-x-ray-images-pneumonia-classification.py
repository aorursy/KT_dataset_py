TRAIN_DIR = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train'
TEST_DIR = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test'
VAL_DIR = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val'
from tqdm import tqdm
import cv2
CLASSES = ['NORMAL', 'PNEUMONIA']
import matplotlib.pyplot as plt
x_train_n = []
y_train_n = []

for i in tqdm(os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL')):
    img = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL' + '/' + i)
    img = cv2.resize(img, (256,256))
    
    x_train_n.append(img)
    y_train_n.append('Normal')
x_train_p = []
y_train_p = []

for i in tqdm(os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA')):
    img = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA' + '/' + i)
    img = cv2.resize(img, (256,256))
    
    x_train_p.append(img)
    y_train_p.append('Normal')
print(plt.imshow(x_train_n[28]))  #Sample normal image
print(plt.imshow(x_train_p[28])) #Sample Pneumonia image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255, width_shift_range=0.2, height_shift_range=0.2,
                                  shear_range=0.15, zoom_range=0.15, horizontal_flip=True, fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1/255)
train = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224), class_mode='binary', batch_size=32)
test = test_datagen.flow_from_directory(TEST_DIR, target_size=(224,224), class_mode='binary', batch_size=32)
valid = test_datagen.flow_from_directory(VAL_DIR, target_size=(224,224), class_mode='binary', batch_size=32)
import keras
from keras import layers
input_ = keras.layers.Input(shape=(224,224,3))

conv_1 = keras.layers.Conv2D(32, (3,3))(input_)
act_1 = keras.layers.ReLU()(conv_1)
max_1 = keras.layers.MaxPooling2D(2,2)(act_1)
drop_1 = keras.layers.Dropout(0.4)(max_1)

conv_2 = keras.layers.Conv2D(64, (3,3))(drop_1)
act_2 = keras.layers.ReLU()(conv_2)
max_2 = keras.layers.MaxPooling2D(2,2)(act_2)
drop_2 = keras.layers.Dropout(0.2)(max_2)

batch_1 = keras.layers.BatchNormalization()(drop_2)

flatten_ = keras.layers.Flatten()(batch_1)
dense_1 = keras.layers.Dense(128)(flatten_)
act_4 = keras.layers.ReLU()(dense_1)
drop_4 = keras.layers.Dropout(0.2)(act_4)

output_ = keras.layers.Dense(1, activation='sigmoid')(drop_4)
model = keras.models.Model(inputs=[input_], outputs=[output_])
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train, epochs=10, validation_data=valid)
model.evaluate(test)
pred = model.predict(test)
pred
