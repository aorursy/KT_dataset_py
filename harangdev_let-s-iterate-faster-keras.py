import sys

sys.path.append('../input/autoaugment')

from autoaugment import ImageNetPolicy



import PIL



import pandas as pd

import numpy as np



import keras

from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from keras.models import Model

from keras.layers import GlobalAveragePooling2D, Dense

from keras.optimizers import SGD



from sklearn.metrics import f1_score

from sklearn.model_selection import StratifiedKFold



from tqdm import tqdm
IMG_SIZE = 224

BATCH_SIZE = 128
train = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/train.csv')

test = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/test.csv')
folds = list(StratifiedKFold(5, random_state=0, shuffle=False).split(train, train['class']))
def make_dataset(df, augment=True, test=False):

    x = []

    for i, image_path in tqdm(enumerate(df['img_file'].values)):

        if test:

            image_path = '../input/3rd-ml-month-car-image-cropping-dataset/test_crop/' + image_path

        else:

            image_path = '../input/3rd-ml-month-car-image-cropping-dataset/train_crop/' + image_path

        image = PIL.Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

        x.append(image)

    if test:

        y = None

    else:

        y = keras.utils.to_categorical(df['class'].astype(int)-1, num_classes=196)

    return x, y



x_train, y_train = make_dataset(train.iloc[folds[0][0]])

x_val, y_val = make_dataset(train.iloc[folds[0][1]])
print(len(x_train))

x_train[:5]
print(y_train.shape)

y_train
class SlowGenerator(keras.utils.Sequence):

    

    def __init__(self, x, y=None, augment=True):

        self.x = x

        self.y = y

        self.augment = augment

        self.policy = ImageNetPolicy()

        

    def __len__(self):

        return int(np.ceil(len(self.x)/BATCH_SIZE))

    

    def _transform(self, image):

        image = self.policy(image)

        image = preprocess_input(np.array(image))

        return image

    

    def __getitem__(self, index):

        if self.y is None:

            images = [PIL.Image.open('../input/3rd-ml-month-car-image-cropping-dataset/test_crop/'+_x).convert('RGB').resize((IMG_SIZE, IMG_SIZE)) for _x in self.x[index*BATCH_SIZE:(index+1)*BATCH_SIZE]]

            if self.augment:

                x = list(map(self._transform, images))

            else:

                x = [preprocess_input(np.array(x_)) for x_ in images]

            return np.array(x)

        

        images = [PIL.Image.open('../input/3rd-ml-month-car-image-cropping-dataset/train_crop/'+_x).convert('RGB').resize((IMG_SIZE, IMG_SIZE)) for _x in self.x[index*BATCH_SIZE:(index+1)*BATCH_SIZE]]

        if self.augment:

            x = np.array(list(map(self._transform, images)))

            y = self.y[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

            y = keras.utils.to_categorical(y-1, num_classes=196)

            return x, y

        

        x = np.array([preprocess_input(np.array(x_)) for x_ in images])

        y = self.y[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

        y = keras.utils.to_categorical(y-1, num_classes=196)

        return x, y



class FastGenerator(keras.utils.Sequence):

    

    def __init__(self, x, y=None, augment=True):

        self.x = x

        self.y = y

        self.augment = augment

        self.policy = ImageNetPolicy()

        

    def __len__(self):

        return int(np.ceil(len(self.x)/BATCH_SIZE))

    

    def _transform(self, image):

        image = self.policy(image)

        image = preprocess_input(np.array(image))

        return image

    

    def __getitem__(self, index):

        

        if self.y is None:

            if self.augment:

                x = list(map(self._transform, self.x[index*BATCH_SIZE:(index+1)*BATCH_SIZE]))

            else:

                x = [preprocess_input(np.array(x_)) for x_ in self.x[index*BATCH_SIZE:(index+1)*BATCH_SIZE]]

            return np.array(x)

        

        if self.augment:

            x = np.array([preprocess_input(np.array(x_)) for x_ in self.x[index*BATCH_SIZE:(index+1)*BATCH_SIZE]])

            y = self.y[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

            return x, y

        

        x = np.array([preprocess_input(np.array(x_)) for x_ in self.x[index*BATCH_SIZE:(index+1)*BATCH_SIZE]])

        y = self.y[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

        return x, y
fast_train_generator = FastGenerator(x_train, y_train)

fast_val_generator = FastGenerator(x_val, y_val, augment=False)

slow_train_generator = SlowGenerator(train.iloc[folds[0][0]]['img_file'].values, train.iloc[folds[0][0]]['class'].values)

slow_val_generator = SlowGenerator(train.iloc[folds[0][1]]['img_file'].values, train.iloc[folds[0][1]]['class'].values, augment=False)
def get_model():

    

    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')

    base_model.trainable = False

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    output = Dense(196, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer=SGD(momentum=0.9))



    return model
model = get_model()

history = model.fit_generator(

    slow_train_generator,

    validation_data=slow_val_generator,

    epochs=2,

    verbose=1

)
model = get_model()

history = model.fit_generator(

    fast_train_generator,

    validation_data=fast_val_generator,

    epochs=2,

    verbose=1

)
model = get_model()

history = model.fit_generator(

    fast_train_generator,

    validation_data=fast_val_generator,

    epochs=2,

    verbose=1,

    use_multiprocessing=True,

    workers=2

)