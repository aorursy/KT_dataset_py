import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
def add_status(train):
    
    # It has been done this way in order to avoid unnecessary warnings
    train['status'] = 0
    train['label'] = 0
    
    train.loc[train.healthy == 1, 'status'] = 'healthy'
    train.loc[train.healthy == 1, 'label'] = 0
    train.loc[train.multiple_diseases == 1, 'status'] = 'multiple_diseases'
    train.loc[train.multiple_diseases == 1, 'label'] = 1
    train.loc[train.rust == 1, 'status'] = 'rust'
    train.loc[train.rust == 1, 'label'] = 2
    train.loc[train.scab == 1, 'status'] = 'scab'
    train.loc[train.scab == 1, 'label'] = 3
    
    return train
def load_images(train, directory):
    
    # This function loads the images, resizes them and puts them into an array
    
    img_size = 900
    train_image = []
    for name in train['image_id']:
        path = directory + 'images/' + name + '.jpg'
        img = cv2.imread(path)
        image = cv2.resize(img, (img_size, img_size))
        train_image.append(image)
    train_image_array = np.array(train_image)
    
    return train_image_array
def save_images(folder_name, x, y):
    healthy_count = 0
    multiple_diseases_count = 0
    rust_count = 0
    scab_count = 0

    for i in range(0, len(x)):
        if y[i] == 0:
            healthy_count += 1
            name = 'healthy_' + str(healthy_count) + '.jpg'
            cv2.imwrite(folder_name + 'healthy/' + name, x[i])
        elif y[i] == 1:
            multiple_diseases_count +=1
            name = 'multiple_diseases_' + str(multiple_diseases_count) + '.jpg'
            cv2.imwrite(folder_name + 'multiple_diseases/' + name, x[i])
        elif y[i] == 2:
            rust_count +=1
            name = 'rust_' + str(rust_count) + '.jpg'
            cv2.imwrite(folder_name + 'rust/' + name, x[i])
        elif y[i] == 3:
            scab_count +=1
            name = 'scab_' + str(scab_count) + '.jpg'
            cv2.imwrite(folder_name + 'scab/' + name, x[i])
def make_folders(directory):
    import os
    classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
    os.mkdir(directory + 'image_generator')
    os.mkdir(directory + 'image_generator/train')
    for cls in classes:
        os.mkdir(directory + 'image_generator/train/' + cls )
    os.mkdir(directory + 'image_generator/validation')
    for cls in classes:
        os.mkdir(directory + 'image_generator/validation/' + cls )
directory = '../input/plant-pathology-2020-fgvc7/'
df_train = pd.read_csv(directory + 'train.csv')
df_train = add_status(df_train)
train_img = load_images(df_train, directory)
directory_output = '/kaggle/working/'
make_folders(directory_output)
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(train_img, df_train['label'].to_numpy(), 
                                                  stratify = df_train['label'].to_numpy(), test_size = 0.2)
save_images(directory_output + 'image_generator/train/', x_train, y_train)
save_images(directory_output + 'image_generator/validation/', x_val, y_val)
def show_distribution():
    n_healthy = 413
    n_multiple = 73
    n_rust = 497
    n_scab = 473

    print('Number of Healthy images:', n_healthy)
    print('Number of Multiple_diseases images:', n_multiple)
    print('Number of Rust images:', n_rust)
    print('Number of Scab images:', n_scab)
    
    distribution_dictionary = {'healthy' : n_healthy,
                               'multiple_diseases' : n_multiple,
                               'rust' : n_rust,
                               'scab' : n_scab}
    
    return distribution_dictionary
def load_images_new(dir_name, n_img):
    
    # This function loads the images, resizes them and puts them into an array
    
    img_size = 900
    train_image = []
    counter = 0
    for i in range(1, n_img+1):
        counter += 1
        if (counter % 100 == 0):
            print('we have loaded', counter , 'images')
        path = directory + dir_name + '/' + dir_name + '_' + str(i) + '.jpg'
#         print(path)
        img = cv2.imread(path)
        image = cv2.resize(img, (img_size, img_size))
        train_image.append(image)
    train_image_array = np.array(train_image)
    
    return train_image_array
def append_images(name, add_images):
    
    train_add = train_img.copy()
    
    if name == 'rust': # Don't add anything
        print('rust')

    elif name == 'multiple_diseases': # Add 4 times the amount of images and more
        for i in range(0,4):
            train_add = np.concatenate((train_add, train_img))
        train_add = np.concatenate((train_add, train_img[0:59]))
        
    else: # Add the needed amount.
        train_add = train_img[0:add_images]
        
    return train_add
def write_images(directory, train_img, len_previous, name):
    
    for i in range(0, len(train_img)):
        img_name =  name + '_' + str(len_previous + i + 1) + '.jpg'
        cv2.imwrite(directory + name + '/' + img_name, train_img[i])
directory = '/kaggle/working/image_generator/train/'
names = ['healthy', 'multiple_diseases', 'rust', 'scab']
distr_dict = show_distribution()
print()

for name in names:
    if name == 'rust':
        print('Russt!')
        
    else:
        df_len_previous = distr_dict[name] # The length before adding more images

        train_img = load_images_new(name, df_len_previous)
        print()
#         train_img = train_img[df_train[name] == 1] # Takes the images of the label
        n_samples = distr_dict[name] # Takes the number of samples of that images

        add_images = distr_dict['rust'] - n_samples # The amount of images we have to add
        print('images we have to add:', add_images)

        train_add = append_images(name, add_images)

#         df_train = append_dataframe(df_train, name, train_add)

        write_images(directory, train_add, df_len_previous, name)
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import keras
from keras import models, Sequential
from keras.layers import Dense
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import losses
from keras.models import model_from_json
def predict_and_save(model, x_test, name):
    x_pred = model.predict(x_test, verbose = 1)
    df_test['healthy'] = x_pred[:,0]
    df_test['multiple_diseases'] = x_pred[:,1]
    df_test['rust'] = x_pred[:,2]
    df_test['scab'] = x_pred[:,3]
    df_test.to_csv(name, index = None)
def model_plot(history):

    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'validation'], loc='upper left')
    plt.show()
def print_score(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose = 0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
from keras import backend as K
from keras.layers import Layer, InputSpec
from keras.legacy import interfaces

class GlobalKMaxPooling2D(Layer): #Inherits the properties of Layer class    
    

    def __init__(self, data_format=None, k = 10, **kwargs):
        super(GlobalKMaxPooling2D, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        self.k = k

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], input_shape[3])
        else:
            return (input_shape[0], input_shape[1])

    def get_config(self):
        config = {'data_format': self.data_format, 'k' : self.k}
        base_config = super(GlobalKMaxPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
        
    def call(self, inputs):
        if self.data_format == 'channels_last':
            # Here first sort
            # Then take K maximum values
            # Then average them
            k = self.k

            input_reshaped = tf.reshape(inputs, [tf.shape(inputs)[0], -1, tf.shape(inputs)[3]])
            input_reshaped = tf.reshape(input_reshaped, [tf.shape(input_reshaped)[0], tf.shape(input_reshaped)[2], tf.shape(input_reshaped)[1]])
            top_k = tf.math.top_k(input_reshaped, k=k, sorted = True, name = None)[0]
            mean = tf.keras.backend.mean(top_k, axis = 2)
            #assert ((input_reshaped.get_shape()[0], input_reshaped.get_shape()[-1]) == mean.get_shape())
        
        return mean
def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
               lr_min=0.00001, lr_rampup_epochs=8, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max #* strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    return lrfn

lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
def load_generators_flow_from_directory():

    from sklearn.model_selection import train_test_split

    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 40,
                                       width_shift_range = 0.25, height_shift_range = 0.25, 
                                       shear_range = 0.25, zoom_range = 0.25, 
                                       horizontal_flip = True, vertical_flip = True,
                                       fill_mode = 'nearest')

    directory_train = '/kaggle/working/image_generator/train'
    directory_val = '/kaggle/working/image_generator/validation'
    validation_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(directory_train, 
                                                        target_size = (448,448),
                                                        batch_size = 4, 
                                                        shuffle = True, 
                                                        class_mode = 'categorical')
    
    validation_generator = validation_datagen.flow_from_directory(directory_val,
                                                                  target_size = (448,448),
                                                                  batch_size = 4,
                                                                  shuffle = True,
                                                                  class_mode = 'categorical')    
    return (train_generator, validation_generator)
def build_model():
    
    from keras.applications.resnet import ResNet50
    import tensorflow as tf
    from tensorflow.keras.metrics import AUC
    import keras
    from keras.layers import Dense
    from keras.models import Sequential
    import matplotlib.pyplot as plt
    # import efficientnet.keras as efn 
    # from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.applications.xception import Xception
        
    input_shape = (448, 448, 3)
    # model_efficientnet = efn.EfficientNetB5(weights='imagenet', include_top = False, input_shape = input_shape)
    # model_resnet = ResNet50(include_top = False, weights = 'imagenet', input_shape = input_shape)
    # model_inception = InceptionResNetV2(include_top = False, weights = 'imagenet', input_shape = input_shape)
    model_xception = Xception(include_top = False, weights = 'imagenet', input_shape = input_shape)
    model = Sequential()
    # base_model =  efn.EfficientNetB7(weights='imagenet', include_top=False, pooling='avg', input_shape= input_shape)
#         base_model = base_model.output

    model.add(model_xception)
    model.add(GlobalKMaxPooling2D(data_format = 'channels_last' , k = 20))
    model.add(Dense(4, activation = 'softmax'))
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                              metrics = ['categorical_accuracy', AUC()]) 
    print('Model Compiled!')
    model.summary()

    return model
def save_model(name, model):
        """ Saves the model as a Json file"""
        # serialize model to JSON
        model_json = model.to_json()
        with open( str(name) + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(name + ".h5")
        # print("Saved model to disk")  
model = build_model()
train_generator, validation_generator = load_generators_flow_from_directory()
history = model.fit_generator(train_generator, epochs = 40, verbose = 1, validation_data = validation_generator,
                              callbacks = [lr_schedule])
name = 'Plants_balanced_xception_img448_V2'
save_model(name, model)