import os



import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



import xml.etree.ElementTree as ET



import numpy as np

import pandas as pd



from skimage.io import imread

from skimage.transform import resize

from PIL import Image

from imgaug import augmenters as iaa



from sklearn.model_selection import train_test_split



from keras.models import *

from keras.layers import *

from keras.optimizers import *

from keras.utils import *

from keras.callbacks import *



from keras.applications.densenet import DenseNet121, preprocess_input
breed_list = os.listdir("../input/stanford-dogs-dataset/images/Images/")



num_classes = len(breed_list)

print("{} breeds".format(num_classes))



n_total_images = 0

for breed in breed_list:

    n_total_images += len(os.listdir("../input/stanford-dogs-dataset/images/Images/{}".format(breed)))

print("{} images".format(n_total_images))
label_maps = {}

label_maps_rev = {}

for i, v in enumerate(breed_list):

    label_maps.update({v: i})

    label_maps_rev.update({i : v})
def show_dir_images(breed, n_to_show):

    plt.figure(figsize=(16,16))

    img_dir = "../input/stanford-dogs-dataset/images/Images/{}/".format(breed)

    images = os.listdir(img_dir)[:n_to_show]

    for i in range(n_to_show):

        img = mpimg.imread(img_dir + images[i])

        plt.subplot(n_to_show/4+1, 4, i+1)

        plt.imshow(img)

        plt.axis('off')
print(breed_list[0])

show_dir_images(breed_list[0], 16)
%%time



# copy from https://www.kaggle.com/gabrielloye/dogs-inception-pytorch-implementation

# reduce the background noise



os.mkdir('data')

for breed in breed_list:

    os.mkdir('data/' + breed)

print('Created {} folders to store cropped images of the different breeds.'.format(len(os.listdir('data'))))



for breed in os.listdir('data'):

    for file in os.listdir('../input/stanford-dogs-dataset/annotations/Annotation/{}'.format(breed)):

        img = Image.open('../input/stanford-dogs-dataset/images/Images/{}/{}.jpg'.format(breed, file))

        tree = ET.parse('../input/stanford-dogs-dataset/annotations/Annotation/{}/{}'.format(breed, file))

        xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)

        xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)

        ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)

        ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)

        img = img.crop((xmin, ymin, xmax, ymax))

        img = img.convert('RGB')

        img = img.resize((224, 224))

        img.save('data/' + breed + '/' + file + '.jpg')
def paths_and_labels():

    paths = list()

    labels = list()

    targets = list()

    for breed in breed_list:

        base_name = "./data/{}/".format(breed)

        for img_name in os.listdir(base_name):

            paths.append(base_name + img_name)

            labels.append(breed)

            targets.append(label_maps[breed])

    return paths, labels, targets



paths, labels, targets = paths_and_labels()



assert len(paths) == len(labels)

assert len(paths) == len(targets)



targets = np_utils.to_categorical(targets, num_classes=num_classes)
batch_size = 64



class ImageGenerator(Sequence):

    

    def __init__(self, paths, targets, batch_size, shape, augment=False):

        self.paths = paths

        self.targets = targets

        self.batch_size = batch_size

        self.shape = shape

        self.augment = augment

        

    def __len__(self):

        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    

    def __getitem__(self, idx):

        batch_paths = self.paths[idx * self.batch_size : (idx + 1) * self.batch_size]

        x = np.zeros((len(batch_paths), self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32)

        y = np.zeros((self.batch_size, num_classes, 1))

        for i, path in enumerate(batch_paths):

            x[i] = self.__load_image(path)

        y = self.targets[idx * self.batch_size : (idx + 1) * self.batch_size]

        return x, y

    

    def __iter__(self):

        for item in (self[i] for i in range(len(self))):

            yield item

            

    def __load_image(self, path):

        image = imread(path)

        image = preprocess_input(image)

        if self.augment:

            seq = iaa.Sequential([

                iaa.OneOf([

                    iaa.Fliplr(0.5),

                    iaa.Flipud(0.5),

                    iaa.CropAndPad(percent=(-0.25, 0.25)),

                    iaa.Crop(percent=(0, 0.1)),

                    iaa.Sometimes(0.5,

                        iaa.GaussianBlur(sigma=(0, 0.5))

                    ),

                    iaa.Affine(

                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},

                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},

                        rotate=(-180, 180),

                        shear=(-8, 8)

                    )

                ])

            ], random_order=True)

            image = seq.augment_image(image)

        return image
train_paths, val_paths, train_targets, val_targets = train_test_split(paths, 

                                                  targets,

                                                  test_size=0.15, 

                                                  random_state=1029)



train_gen = ImageGenerator(train_paths, train_targets, batch_size=32, shape=(224,224,3), augment=True)

val_gen = ImageGenerator(val_paths, val_targets, batch_size=32, shape=(224,224,3), augment=False)
inp = Input((224, 224, 3))

backbone = DenseNet121(input_tensor=inp,

                       weights="../input/densenet-keras/DenseNet-BC-121-32-no-top.h5",

                       include_top=False)

x = backbone.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

x = Dense(512, activation="relu")(x)

x = Dropout(0.5)(x)

outp = Dense(num_classes, activation="softmax")(x)



model = Model(inp, outp)
for layer in model.layers[:-6]:

    layer.trainable = False
model.compile(optimizer="adam",

              loss="categorical_crossentropy",

              metrics=["acc"])
history = model.fit_generator(generator=train_gen, 

                              steps_per_epoch=len(train_gen), 

                              validation_data=val_gen, 

                              validation_steps=len(val_gen),

                              epochs=20)
plt.rcParams['figure.figsize'] = (6,6)



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)



plt.title('Training and validation accuracy')

plt.plot(epochs, acc, 'red', label='Training acc')

plt.plot(epochs, val_acc, 'blue', label='Validation acc')

plt.legend()



plt.figure()

plt.title('Training and validation loss')

plt.plot(epochs, loss, 'red', label='Training loss')

plt.plot(epochs, val_loss, 'blue', label='Validation loss')



plt.legend()



plt.show()
for layer in model.layers[:]:

    layer.trainable = True
# a check point callback to save our best weights

checkpoint = ModelCheckpoint('dog_breed_classifier_model.h5', 

                             monitor='val_acc', 

                             verbose=1, 

                             save_best_only=True, 

                             mode='max', 

                             save_weights_only=True)



# a reducing lr callback to reduce lr when val_loss doesn't increase

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                                   patience=1, verbose=1, mode='min',

                                   min_delta=0.0001, cooldown=2, min_lr=1e-7)



# for early stop

early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
history = model.fit_generator(generator=train_gen, 

                              steps_per_epoch=len(train_gen), 

                              validation_data=val_gen, 

                              validation_steps=len(val_gen),

                              epochs=20,

                              callbacks=[checkpoint, reduce_lr, early_stop])
plt.rcParams['figure.figsize'] = (6,6)



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)



plt.title('Training and validation accuracy')

plt.plot(epochs, acc, 'red', label='Training acc')

plt.plot(epochs, val_acc, 'blue', label='Validation acc')

plt.legend()



plt.figure()

plt.title('Training and validation loss')

plt.plot(epochs, loss, 'red', label='Training loss')

plt.plot(epochs, val_loss, 'blue', label='Validation loss')



plt.legend()



plt.show()
print(max(val_acc))
def download_and_predict(url, filename):

    # download and save

    os.system("curl -s {} -o {}".format(url, filename))

    img = Image.open(filename)

    img = img.convert('RGB')

    img = img.resize((224, 224))

    img.save(filename)

    # show image

    plt.figure(figsize=(4, 4))

    plt.imshow(img)

    plt.axis('off')

    # predict

    img = imread(filename)

    img = preprocess_input(img)

    probs = model.predict(np.expand_dims(img, axis=0))

    for idx in probs.argsort()[0][::-1][:5]:

        print("{:.2f}%".format(probs[0][idx]*100), "\t", label_maps_rev[idx].split("-")[-1])
download_and_predict("https://cdn.pixabay.com/photo/2018/08/12/02/52/belgian-mallinois-3599991_1280.jpg",

                     "test_1.jpg")
download_and_predict("https://cdn.pixabay.com/photo/2016/07/25/00/06/corgi-1539598_1280.jpg",

                     "test_2.jpg")
download_and_predict("https://cdn.pixabay.com/photo/2019/02/24/20/15/chihuahua-4018429_1280.jpg",

                     "test_3.jpg")
download_and_predict("https://cdn.pixabay.com/photo/2018/03/31/06/31/dog-3277416_1280.jpg",

                     "test_4.jpg")
download_and_predict("https://cdn.pixabay.com/photo/2016/02/19/15/46/dog-1210559_1280.jpg",

                     "test_5.jpg")
!rm -rf data/* 
!rm -f test_*.jpg