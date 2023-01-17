#!pip install tensorflow-gpu
import numpy as np
import glob
import numpy as np
import time
import matplotlib.pyplot as plt
#@title
def load_data_from_np(folder_path, classes = None, verbose = 1):
    '''
    Загрузит множество npy файлов и объединит. (X, y)

    Parameters:
    
      folder_path (string): Путь до директории с файлами (Заканчивается именем папки)

      classes (list[string]): Названия классов, которые необходимо загружать
            Если None то загрузит все
  
      verbose (int): Если 1 то выводит логи. 0 - иначе
  '''
    X, y = None, []
    start_time = time.time()

    for file_path in glob.glob(folder_path + '/*.*'):
        class_name = file_path.split('/')[-1].split('.')[0]
        if ((classes == None) or (class_name in classes)):
            if (X is None):
                X = np.load(file_path)
                y = np.array([class_name]*X.shape[0])
            else:
                X_temp = np.load(file_path)
                X = np.concatenate((X, X_temp))
                y = np.concatenate((y, np.array([class_name]*X_temp.shape[0])))
      
        if (verbose == 1):
            #print('{} loaded. Total time {}'.format(class_name, time.time() - start_time))
            print('%-25s Total time: %-4f'%(class_name, time.time() - start_time))
    print('\nDone')
    return (X, np.array(y))
GLOBAL_PATH = '/content/drive/My Drive/Colab Notebooks/hw'
DATA_PATH = '../input/simpsons-train-numpy-my/np_images/train'
INPUT_PATH = '../input'
n_classes = 7
X, y = load_data_from_np(DATA_PATH, classes=['bart_simpson', 'marge_simpson', 'lisa_simpson', 'krusty_the_clown',
                                             'homer', 'abraham_grampa_simpson', 'maggie_simpson'])
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y.reshape(-1, 1))
map_characters = {i : ohe.categories_[0][i] for i in range(n_classes) }


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=True)
plt.figure(figsize=(20,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    n = np.random.choice(X.shape[0])
    
    plt.imshow(X[n])
    #plt.title(ohe.inverse_transform(y[n].reshape(1, -1))[0][0])
    plt.title(map_characters[np.argmax(y[n])])
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam

def create_model_six_conv(input_shape):
    """
    CNN Keras model with 6 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu')) 
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(18, activation='softmax'))
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return model, opt

def load_model_from_checkpoint(weights_path, input_shape=(64,64,3)):
    model, opt = create_model_six_conv(input_shape)
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    return model
%time pretrained_model = load_model_from_checkpoint(INPUT_PATH + '/simpsons-train-numpy-my/weights.best.hdf5')
model = Sequential()

for l in pretrained_model.layers[:-3]:
    model.add(l)

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=opt,
          metrics=['accuracy'])
model.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_gen_train = ImageDataGenerator(
                    rotation_range=20,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.3
                    )

data_train_gen = image_gen_train.flow(X_train, y_train)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
augmented_images = [data_train_gen[5][0][20] for i in range(5)]
plotImages(augmented_images)
try:
    history = model.fit(x=data_train_gen, epochs=20, verbose=1, shuffle=True, validation_data=(X_val, y_val))
except KeyboardInterrupt:
    print('\n\nStopped')

import sklearn
from sklearn.metrics import classification_report

print('\n', sklearn.metrics.classification_report(np.argmax(y_val, axis=1), 
                                                  np.argmax(model.predict(X_val), axis=1), 
                                                  target_names=list(map_characters.values())), sep='')
#model.save('../input/weightsbesthdf5/model_8classes_als.h5')
model.save('model_7classes_als.h5')
#Для загрузки нашей модели
#model = keras.models.load_model('../input/simpsons-train-numpy-my/model_7classes_als.h5')
X_test, y_test = load_data_from_np('../input/simpsons-train-numpy-my/np_images/test',
                                  ['bart_simpson', 'marge_simpson', 'lisa_simpson', 'krusty_the_clown',
                                             'homer', 'abraham_grampa_simpson', 'maggie_simpson'])
y_test = ohe.transform(y_test.reshape(-1, 1))
print('\n', sklearn.metrics.classification_report(np.argmax(y_test, axis=1), 
                                                  np.argmax(model.predict(X_test), axis=1), 
                                                  target_names=list(map_characters.values())), sep='')
from mpl_toolkits.axes_grid1 import AxesGrid
import cv2

F = plt.figure(1, (15,20))
grid = AxesGrid(F, 111, nrows_ncols=(4, 4), axes_pad=0, label_mode="1")

for i in range(16):
    n = np.random.choice(X_test.shape[0])

    img = X_test[n]
    a = model.predict(img.reshape(1, 64, 64,3))[0]
    
    actual = map_characters[np.argmax(y_test[n])].split('_')[0]

    text = sorted(['{:s} : {:.1f}%'.format(map_characters[k].split('_')[0].title(), 100*v) for k,v in enumerate(a)], 
       key=lambda x:float(x.split(':')[1].split('%')[0]), reverse=True)[:3]
    
    img = cv2.resize(img, (352, 352))
    cv2.rectangle(img, (0,260),(215,352),(255,255,255), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Actual : %s' % actual, (10, 280), font, 0.7,(0,0,0),2,cv2.LINE_AA)
    for k, t in enumerate(text):
        cv2.putText(img, t,(10, 300+k*18), font, 0.65,(0,0,0),2,cv2.LINE_AA)
    grid[i].imshow(img)
TEST_PATH = '../input/simpsons-train-numpy-my/simpsons_test_set_8_classes'
F = plt.figure(1, (15,20))
grid = AxesGrid(F, 111, nrows_ncols=(5, 5), axes_pad=0, label_mode="1")

for i in range(25):
    class_path = glob.glob(TEST_PATH + '/*')[i%n_classes]
    img = cv2.imread(np.random.choice(glob.glob(class_path + '/*')))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    temp = cv2.resize(img,(64,64)).astype('float32') / 255.
    img = cv2.resize(img, (352, 352))
    
    a = model.predict(temp.reshape(1, 64, 64,3))[0]
    actual = class_path.split('/')[-1].split('_')[0]
    
    text = sorted(['{:s} : {:.1f}%'.format(map_characters[k].split('_')[0].title(), 100*v) for k,v in enumerate(a)], 
       key=lambda x:float(x.split(':')[1].split('%')[0]), reverse=True)[:3]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (0,260),(215,352),(255,255,255), -1)
    cv2.putText(img, 'Actual : %s' % actual, (10, 280), font, 0.7,(0,0,0),2,cv2.LINE_AA)
    for k, t in enumerate(text):
        cv2.putText(img, t,(10, 300+k*18), font, 0.65,(0,0,0),2,cv2.LINE_AA)

    grid[i].imshow(img)