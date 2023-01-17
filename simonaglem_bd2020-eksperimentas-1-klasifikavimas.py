import os, cv2, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
%matplotlib inline 
import warnings  
warnings.filterwarnings('ignore')
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
# Realių duomenų iš "Dogs vs. Cats: Kernels Edition" užkrovimas į temp direktoriją
import zipfile
import glob

if not os.path.exists('../working/temp'):
    print('Kuriama temp direktorija..')
    zip_file = glob.glob('../input/*/*.zip')
    print(zip_file)

    def extract_zip(file):
        with zipfile.ZipFile(file,"r") as zip_ref:
            zip_ref.extractall("temp")

    for files in zip_file:
        extract_zip(files)
synth_train_dir = '../input/synth3000/synth3000/'
small_real_train_dir = '../input/my-real/real_images/'
real_train_dir = '../working/temp/train/'
test_dir = '../working/temp/test/'

channels = 3
class_size = 3000 # Su kiek realių ir sintetinių duomenų norime apmokyti tinklą
epochs = 200 # Klasifikavimo tinklo treniravimo epochų skaičius

# Duomenys įsikeliami iš direktorijų pagal klases
train_dogs = [real_train_dir+i for i in os.listdir(real_train_dir) if 'dog' in i]
train_cats = [real_train_dir+i for i in os.listdir(real_train_dir) if 'cat' in i]

my_train_dogs = [small_real_train_dir+i for i in os.listdir(small_real_train_dir) if 'dog' in i]
my_train_cats = [small_real_train_dir+i for i in os.listdir(small_real_train_dir) if 'cat' in i]

synth_dogs = [synth_train_dir+i for i in os.listdir(synth_train_dir) if 'dog' in i]
synth_cats = [synth_train_dir+i for i in os.listdir(synth_train_dir) if 'cat' in i]

test_images = [test_dir+i for i in os.listdir(test_dir)]

# Pasiimamas tam tikras kiekis nuotraukų pagal klasę(class_size) iš realių mokymo bei sintetinių duomenų rinkinių
train_images = train_dogs[:class_size] + train_cats[:class_size]
random.shuffle(train_images)

small_train_images = my_train_dogs + my_train_cats
random.shuffle(small_train_images)

synth_images = synth_dogs[:class_size] + synth_cats[:class_size]
random.shuffle(synth_images)

test_images =  test_images[:1000]
random.shuffle(test_images)

val_images = train_dogs[4000:5000] + train_cats[4000:5000]
random.shuffle(val_images)

def read_image(file_path, rows, cols):
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    b,g,r = cv2.split(image)
    new_image = cv2.merge([r,g,b])
    return cv2.resize(new_image, (rows, cols), interpolation=cv2.INTER_CUBIC)

def prep_data(images, rows=64, cols=64):
    count = len(images)
    data = np.ndarray((count, channels, rows, cols), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file, rows, cols)
        data[i] = image.T
    return data

real = prep_data(train_images)
small_real = prep_data(small_train_images)
synth = prep_data(synth_images)
test = prep_data(test_images)
show_test = prep_data(test_images, 256, 256)
validation = prep_data(val_images)
# Susižymimi duomenys(pagal nuotraukų pavadinimą): katė - 0, šuo - 1
def get_labels(labels, images):
    for i in images:
        if 'cat' in i:
            labels.append(0)
        else:
            labels.append(1)
labels = []
small_labels = []
synth_labels = []
val_labels = []

get_labels(labels, train_images)
get_labels(small_labels, small_train_images)
get_labels(synth_labels, synth_images)
get_labels(val_labels, val_images)
import keras.backend.tensorflow_backend as tfback
import tensorflow as tf

def _get_available_gpus():  
    if tfback._LOCAL_DEVICES is None:  
        devices = tf.config.list_logical_devices()  
        tfback._LOCAL_DEVICES = [x.name for x in devices]  
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

optimizer = RMSprop(lr=1e-4)
optimizer2 = Adam(lr=1e-4, decay=1e-6)
optimizer3 = SGD(lr=1e-4, momentum=0.9)

def catdognet(): 
    catdog_input = Input(shape=(3, 64, 64))
    
    x = Conv2D(32, 3, padding='same', activation='relu')(catdog_input)
    x = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.7)(x)
    catdog_output = Dense(1, activation='sigmoid')(x)
    
    catdog = Model(inputs=catdog_input, outputs=catdog_output, name='Klasifikatorius')
    catdog.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    return catdog
model = catdognet()
small_model = catdognet()
synth_model = catdognet()
model.summary()
class LossAccHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
def run_catdognet(model, real, labels, batch_size=32):
    history = LossAccHistory()
    model.fit(real, labels, batch_size=batch_size, epochs=epochs, validation_data=(validation, val_labels), verbose=2, shuffle=True, callbacks=[history])
    predictions = model.predict(test, verbose=0)
    return predictions, history
predictions, history = run_catdognet(model, real, labels, 128)
loss = history.losses
val_loss = history.val_losses
acc = history.acc
val_acc = history.val_acc

print('Aukščiausias pasiektas klasifikavimo tikslumas:', max(val_acc))

plt.xlabel('Epocha')
plt.ylabel('Tikslumas')
plt.title('Tikslumo raida su realiais duomenimis')
plt.plot(acc, 'mediumvioletred', label='Mokymosi')
plt.plot(val_acc, 'lightseagreen', label='Validacijos')
plt.xticks(range(0,epochs)[1::20])
plt.legend()
plt.show()

plt.xlabel('Epocha')
plt.ylabel('Nuostolis')
plt.title('Nuostolių raida su realiais duomenimis')
plt.plot(loss, 'mediumvioletred', label='Mokymosi')
plt.plot(val_loss, 'lightseagreen', label='Validacijos')
plt.xticks(range(0,epochs)[1::20])
plt.legend()
plt.ylim(0.6, 1.05)
plt.show()
for i in range(0,10):
    if predictions[i, 0] >= 0.5: 
        print('{:.2%} jog tai yra šuo'.format(predictions[i][0]))
    else: 
        print('{:.2%} jog tai yra katė'.format(1-predictions[i][0]))
        
    plt.imshow(show_test[i].T)
    plt.axis('off')
    plt.show()
s_predictions, s_history = run_catdognet(synth_model, synth, synth_labels, 128)
loss = s_history.losses
val_loss = s_history.val_losses
acc = s_history.acc
val_acc = s_history.val_acc

print('Aukščiausias pasiektas klasifikavimo tikslumas:', max(val_acc))

plt.xlabel('Epocha')
plt.ylabel('Tikslumas')
plt.title('Tikslumo raida su sintetiniais duomenimis')
plt.plot(acc, 'mediumvioletred', label='Mokymosi')
plt.plot(val_acc, 'lightseagreen', label='Validacijos')
plt.xticks(range(0,epochs)[1::20])
plt.legend()
plt.show()

plt.xlabel('Epocha')
plt.ylabel('Nuostolis')
plt.title('Nuostolių raida su sintetiniais duomenimis')
plt.plot(loss, 'mediumvioletred', label='Mokymosi')
plt.plot(val_loss, 'lightseagreen', label='Validacijos')
plt.xticks(range(0,epochs)[1::20])
plt.legend()
plt.ylim(0.1, 1.05)
plt.show()
for i in range(0,10):
    if s_predictions[i, 0] >= 0.5: 
        print('{:.2%} jog tai yra šuo'.format(s_predictions[i][0]))
    else: 
        print('{:.2%} jog tai yra katė'.format(1-s_predictions[i][0]))
        
    plt.imshow(show_test[i].T)
    plt.axis('off')
    plt.show()
r_predictions, r_history = run_catdognet(small_model, small_real, small_labels, 16)
loss = r_history.losses
val_loss = r_history.val_losses
acc = r_history.acc
val_acc = r_history.val_acc

print('Aukščiausias pasiektas klasifikavimo tikslumas:', max(val_acc))

plt.xlabel('Epocha')
plt.ylabel('Tikslumas')
plt.title('Tikslumo raida su pradiniu realių duomenų rinkiniu')
plt.plot(acc, 'mediumvioletred', label='Mokymosi')
plt.plot(val_acc, 'lightseagreen', label='Validacijos')
plt.xticks(range(0,epochs)[1::20])
plt.legend()
plt.ylim(0, 1.05)
plt.show()

plt.xlabel('Epocha')
plt.ylabel('Nuostolis')
plt.title('Nuostolių raida su pradiniu realių duomenų rinkiniu')
plt.plot(loss, 'mediumvioletred', label='Mokymosi')
plt.plot(val_loss, 'lightseagreen', label='Validacijos')
plt.xticks(range(0,epochs)[1::20])
plt.legend()
plt.show()
for i in range(0,10):
    if r_predictions[i, 0] >= 0.5: 
        print('{:.2%} jog tai yra šuo'.format(r_predictions[i][0]))
    else: 
        print('{:.2%} jog tai yra katė'.format(1-r_predictions[i][0]))
        
    plt.imshow(show_test[i].T)
    plt.axis('off')
    plt.show()