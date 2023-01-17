import os

import re



import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



import cv2



import numpy as np

import pandas as pd



from keras.models import *

from keras.layers import *

from keras.optimizers import *

from keras.utils import *

from keras.callbacks import *



from keras import backend as K

from keras.applications.densenet import DenseNet121, preprocess_input

from keras.preprocessing.image import ImageDataGenerator



from sklearn.model_selection import train_test_split

from sklearn.metrics import fbeta_score



from tqdm import tqdm
train_images = os.listdir("../input/imet-2019-fgvc6/train/")

test_images = os.listdir("../input/imet-2019-fgvc6/test/")



print("number of train images: ", len(train_images))

print("number of test  images: ", len(test_images))
train = pd.read_csv("../input/imet-2019-fgvc6/train.csv")

train.head()
labels = pd.read_csv("../input/imet-2019-fgvc6/labels.csv")

labels.head()
labels_map = {v:i for i, v in zip(labels.attribute_id.values, labels.attribute_name.values)}

labels_map_rev = {i:v for i, v in zip(labels.attribute_id.values, labels.attribute_name.values)}



num_classes = len(labels_map)

print("{} categories".format(num_classes))
submission = pd.read_csv("../input/imet-2019-fgvc6/sample_submission.csv")

submission.head()
def ids_to_lables(attribute_id):

    return "\n".join([labels_map_rev[int(i)] for i in attribute_id.split(" ")])
train["labels"] = train.attribute_ids.apply(lambda x: ids_to_lables(x))

train.head()
train["n_cate"] = train.attribute_ids.apply(lambda x: len(x.split(" ")))

train.head()
# TODO: maybe multi cultures here.



def get_culture(x):

    try: 

        return re.search(r"culture::(\w+)", x).group(1)

    except:

        return "none"



train["culture"] = train.labels.apply(lambda x: get_culture(x))

train.head()
def get_num_tag(x):

    return len(re.findall(r"tag::(\w+)", x))



train["n_tag"] = train.labels.apply(lambda x: get_num_tag(x))

train.head()
num_not_culture = train[train.culture == "none"].shape[0]



print("{} ({:.2f}%) not have a culture categroy".format(num_not_culture, 

                                                        num_not_culture *100 / train.shape[0]))
num_not_tag = train[train.n_tag == 0].shape[0]



print("{} ({:.2f}%) not have a tag categroy".format(num_not_tag, 

                                                    num_not_tag *100 / train.shape[0]))
_ = train.n_cate.value_counts().sort_index().plot.bar()
_ = train.n_tag.value_counts().sort_index().plot.bar()
_ = train.culture.value_counts()[:10].sort_index().plot.bar()
def show_images(n_to_show, is_train=True):

    img_dir = "../input/imet-2019-fgvc6/train/" if is_train else "../input/imet-2019-fgvc6/test/"

    plt.figure(figsize=(16,16))

    images = os.listdir(img_dir)[:n_to_show]

    for i in range(n_to_show):

        img = mpimg.imread(img_dir + images[i])

        plt.subplot(n_to_show/2+1, 2, i+1)

        if is_train:

            plt.title(train[train.id == images[i].split(".")[0]].labels.values[0])

        plt.imshow(img)

        plt.axis('off')
show_images(6)
show_images(6, is_train=False)
def obtain_y(ids):

    y = np.zeros(num_classes)

    for idx in ids.split(" "):

        y[int(idx)] = 1

    return y
paths = ["../input/imet-2019-fgvc6/train/{}.png".format(x) for x in train.id.values]

targets = np.array([obtain_y(y) for y in train.attribute_ids.values])
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

        image = cv2.imread(path)

        image = cv2.resize(image, (self.shape[0], self.shape[1]))

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
batch_size = 64



train_paths, val_paths, train_targets, val_targets = train_test_split(paths, 

                                                                      targets,

                                                                      test_size=0.1, 

                                                                      random_state=1029)



train_gen = ImageGenerator(train_paths, train_targets, batch_size=batch_size, shape=(224,224,3), augment=False)

val_gen = ImageGenerator(val_paths, val_targets, batch_size=batch_size, shape=(224,224,3), augment=False)
inp = Input((224, 224, 3))

backbone = DenseNet121(input_tensor=inp,

                       weights="../input/densenet-keras/DenseNet-BC-121-32-no-top.h5",

                       include_top=False)

x = backbone.output

x = GlobalAveragePooling2D()(x)

x = Dense(2048, activation="relu")(x)

x = Dropout(0.5)(x)

outp = Dense(num_classes, activation="sigmoid")(x)



model = Model(inp, outp)
def f_score(y_true, y_pred, threshold=0.1, beta=2):

    tp = tp_score(y_true, y_pred, threshold)

    fp = fp_score(y_true, y_pred, threshold)

    fn = fn_score(y_true, y_pred, threshold)

    precision = tp / (tp + fp)

    recall = tp / (tp + fn)

    return (1+beta**2) * ((precision * recall) / ((beta**2)*precision + recall))





def tp_score(y_true, y_pred, threshold=0.1):

    tp_3d = K.concatenate(

        [

            K.cast(K.expand_dims(K.flatten(y_true)), 'bool'),

            K.cast(K.expand_dims(K.flatten(K.greater(y_pred, K.constant(threshold)))), 'bool'),

            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')

        ], axis=1

    )

    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))

    return tp





def fp_score(y_true, y_pred, threshold=0.1):

    fp_3d = K.concatenate(

        [

            K.cast(K.expand_dims(K.flatten(K.abs(y_true - K.ones_like(y_true)))), 'bool'),

            K.cast(K.expand_dims(K.flatten(K.greater(y_pred, K.constant(threshold)))), 'bool'),

            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')

        ], axis=-1

    )

    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))

    return fp





def fn_score(y_true, y_pred, threshold=0.1):

    fn_3d = K.concatenate(

        [

            K.cast(K.expand_dims(K.flatten(y_true)), 'bool'),

            K.cast(K.expand_dims(K.flatten(K.abs(K.cast(K.greater(y_pred, K.constant(threshold)), 'float') - K.ones_like(y_pred)))), 'bool'),

            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')

        ], axis=1

    )

    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    return fn





def precision_score(y_true, y_pred, threshold=0.1):

    tp = tp_score(y_true, y_pred, threshold)

    fp = fp_score(y_true, y_pred, threshold)

    return tp / (tp + fp)





def recall_score(y_true, y_pred, threshold=0.1):

    tp = tp_score(y_true, y_pred, threshold)

    fn = fn_score(y_true, y_pred, threshold)

    return tp / (tp + fn)
checkpoint = ModelCheckpoint('model.h5', 

                             monitor='val_f_score', 

                             verbose=1, 

                             save_best_only=True, 

                             mode='max', 

                             save_weights_only=True)



reduce_lr = ReduceLROnPlateau(monitor='val_f_score', factor=0.2,

                              patience=1, verbose=1, mode='max',

                              min_delta=0.0001, cooldown=2, min_lr=1e-7)



early_stop = EarlyStopping(monitor="val_f_score", mode="max", patience=5)
model.compile(

    loss='binary_crossentropy',

    optimizer=Adam(1e-03),

    metrics=['acc', f_score])
history = model.fit_generator(generator=train_gen, 

                              steps_per_epoch=len(train_gen), 

                              validation_data=val_gen, 

                              validation_steps=len(val_gen),

                              epochs=15,

                              callbacks=[checkpoint, reduce_lr, early_stop])
plt.rcParams['figure.figsize'] = (6,6)



fscore = history.history['f_score']

val_fscore = history.history['val_f_score']

epochs = range(1, len(fscore) + 1)



plt.title('Training and validation accuracy')

plt.plot(epochs, fscore, 'red', label='Training f_score')

plt.plot(epochs, val_fscore, 'blue', label='Validation f_score')

plt.legend()



plt.show()
model.load_weights("./model.h5")
class TestImageGenerator(Sequence):

    

    def __init__(self, paths, batch_size, shape):

        self.paths = paths

        self.targets = targets

        self.batch_size = batch_size

        self.shape = shape

        

    def __len__(self):

        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    

    def __getitem__(self, idx):

        batch_paths = self.paths[idx * self.batch_size : (idx + 1) * self.batch_size]

        x = np.zeros((len(batch_paths), self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32)

        for i, path in enumerate(batch_paths):

            x[i] = self.__load_image(path)

        return x

    

    def __iter__(self):

        for item in (self[i] for i in range(len(self))):

            yield item

            

    def __load_image(self, path):

        image = cv2.imread(path)

        image = cv2.resize(image, (self.shape[0], self.shape[1]))

        image = preprocess_input(image)

        return image
test_paths = ["../input/imet-2019-fgvc6/test/{}.png".format(x) for x in submission.id.values]

test_gen = TestImageGenerator(test_paths, batch_size=batch_size, shape=(224,224,3))



predicts = model.predict_generator(test_gen, verbose=1)
n = 6

threshold = 0.15



img = cv2.imread(test_paths[n])

plt.imshow(img)



a = np.array(predicts[n]>threshold, dtype=np.int8)

b = np.where(a==1)[0]

for idx in b.tolist():

    print(labels_map_rev[idx])
train.n_tag.describe()
def classifier(probs):

    

    culture = None

    tags = None

    arr = probs.argsort()

    

    culture_threshold = 0.1

    tag_max_threshold = 0.55

    

    n_min_tag = 1

    n_max_tag = 3

    

    # first: find culture category by sorting probs

    

    for idx in arr[::-1]:

        if labels_map_rev[idx].startswith("culture") and probs[idx] > culture_threshold:

            culture = str(idx)

            break           # TODO: maybe multi culture here.

    

    # second: find tags by different threshold

    for threshold in np.arange(0.05, tag_max_threshold, 0.05):

        n = 0                # stores len(tags)

        tags_list = list()   # stores tags

        

        a = np.array(probs > threshold, dtype=np.int8)

        b = np.where(a == 1)[0]

        for idx in b.tolist():

            if labels_map_rev[idx].startswith("tag"):

                n += 1

                tags_list.append(str(idx))

        if n >= n_min_tag and n <= n_max_tag:

            tags = tags_list

            break

    

    # finally packs our answer

    answer = list()

    if culture:

        answer.append(culture)

    if tags:

        for t in tags:

            answer.append(t)

            

    return " ".join(answer)
predictions = list()



for probs in tqdm(predicts):

    predictions.append(classifier(probs))
submission["attribute_ids"] = np.array(predictions)

submission.head()
submission_df = submission.copy()

submission_df.n_cate = submission.attribute_ids.apply(lambda x: len(x.split(" ")))

_ = submission_df.n_cate.value_counts().sort_index().plot.bar()
submission.to_csv('submission.csv', index=False)
submission.shape
!head submission.csv