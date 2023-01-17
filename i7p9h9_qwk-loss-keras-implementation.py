import os

import cv2

import numpy as np

import pandas as pd



from tqdm import tqdm_notebook as tqdm

from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import train_test_split



import imgaug as ia

import imgaug.augmenters as iaa



import keras.backend as K

from keras import models

from keras import layers

from keras import callbacks

from keras import optimizers

from keras.applications import DenseNet121
class Eye(object):

    def __init__(self, path, target, memory_allocate=False, size=None):

        self.path = path

        self.size = size

        self.target = target



        if memory_allocate:

            self.__image = self.__get_image()

        else:

            self.__image = None



    def __get_image(self):

        im = cv2.cvtColor(cv2.imread(self.path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        if self.size is not None:

            return cv2.resize(im, self.size).astype("uint8")

        else:

            return im.astype("uint8")



    @property

    def image(self):

        if self.__image is not None:

            return self.__image

        else:

            return self.__get_image()
def get_data(data_path, memory_allocate=False, image_size=None, extension="png"):

    def _get_data(pd_set, folder):

        eyes = []

        for im in tqdm(pd_set.iterrows(), total=pd_set.shape[0]):

            im_path = os.path.join(data_path, folder, "{}.{}".format(im[1].id_code, extension))

            if "diagnosis" in im[1].keys().values:

                target = im[1].diagnosis

            else:

                target = None

            eyes.append(Eye(path=im_path, memory_allocate=memory_allocate, size=image_size, target=target))

        return eyes

    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))



    test_eyes = None

    train_eyes = _get_data(train_df, "train_images")

    if os.path.exists(os.path.join(data_path, "test.csv")):

        test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

        test_eyes = _get_data(test_df, "test_images")



    return train_eyes, test_eyes
class SimpleExtractor(object):

    def __init__(self, image_size: tuple=(224, 224), is_normalized=True, **kwargs):

        self.size = image_size

        self.is_normalized = is_normalized



    @property

    def shape(self):

        return self.size + (3, )



    def get_image(self, eye):

        resized_image = cv2.resize(eye.image, self.size)

        return resized_image



    def normalize(self, image):

        image = image.astype("float32")

        image -= np.mean(image, axis=(0, 1), keepdims=True)

        image /= (np.std(image, axis=(0, 1), keepdims=True) + 1e-7)



        return image



    def __call__(self, eye):

        im = self.get_image(eye=eye)

        if self.is_normalized:

            im = self.normalize(image=im)

        return im[None, ...]





class AugmentExtractor(SimpleExtractor):

    def __init__(self, augmentation=None, **kwargs):

        super(AugmentExtractor, self).__init__(**kwargs)

        self.augmentation = augmentation



    def __call__(self, eye):

        im = self.get_image(eye=eye)

        if self.augmentation is not None:

            im = self.augmentation.augment_image(im)



        if self.is_normalized:

            im = self.normalize(image=im)



        return im[None, ...]
from keras.utils.data_utils import Sequence

class SimpleGenerator(Sequence):

    TRAIN = 0

    TEST = 1



    def __init__(self, eyes, extractor, mode, is_balances_batch=True, batch_size=32, one_hot=True):

        assert mode in [self.TRAIN, self.TEST]



        self.eyes = np.asarray(eyes)

        self.extractor = extractor

        self.mode = mode

        self.one_hot = one_hot

        self.is_balances_batch = is_balances_batch

        self.batch_size = batch_size if self.mode == self.TRAIN else 1



        self.targets = np.asarray([x.target for x in self.eyes])

        self.unique_target = np.asarray(list(set(self.targets)))



    @property

    def n_classes(self):

        return self.unique_target.shape[0]



    def __len__(self):

        return self.eyes.shape[0] // self.batch_size



    def __getitem__(self, item):

        if self.mode == self.TRAIN:

            return self.get_train_batch()

        else:

            return self.get_test_batch(item)



    def get_test_batch(self, item):

        return self.extractor(self.eyes[item])



    def get_train_batch(self):

        X = np.empty((self.batch_size, ) + self.extractor.shape)

        targets = np.random.choice(self.unique_target, self.batch_size)

        Y = []

        for n, t in enumerate(targets):

            if self.is_balances_batch:

                n_eye = np.random.choice(np.where(self.targets == t)[0])

            else:

                n_eye = np.random.randint(self.eyes.shape[0])

            eye = self.eyes[n_eye]

            X[n] = self.extractor(eye)

            if self.one_hot:

                Y.append(self.unique_target == eye.target)

            else:

                Y.append(eye.target)



        Y = np.vstack(Y).astype("float64")



        return X, Y
class QWK(callbacks.Callback):

    def __init__(self, generator, save_path, monitor_length=5, net=None, one_hot=False):

        self._not_improvement_epochs = 0

        self.generator = generator

        self.save_path = save_path

        self.net = net

        self.monitor_length = monitor_length

        self.one_hot = one_hot

        self.kappa_score = -1.0 * np.inf

        super(QWK, self).__init__()



    def on_epoch_end(self, epoch, logs={}):

        if self.net is not None:

            model = self.net

        else:

            model = self.model



        y_val = self.generator.targets



        y_pred = model.predict_generator(self.generator, verbose=1)

        if self.one_hot:

            y_pred = np.argmax(y_pred, axis=1)

        else:

            y_pred = np.round(y_pred)



        qwk = cohen_kappa_score(y_val, y_pred, weights='quadratic')

        if qwk > self.kappa_score:

            self.model.save(os.path.join(self.save_path, "final.h5"), overwrite=True, include_optimizer=False)

            self._not_improvement_epochs = 0

        else:

            self._not_improvement_epochs += 1

        

        if self._not_improvement_epochs > self.monitor_length:

            self.model.stop_training = True

        print("val_kappa: {:.4f}".format(qwk))

def build_densenet121(input_shape=(224, 224, 3), classes=5, one_hot=True):

    net = DenseNet121(

                weights=None,

                include_top=False,

                input_shape=(224, 224, 3)

                )

    model = models.Sequential()

    model.add(net)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dropout(0.2))

    if one_hot:

        model.add(layers.Dense(classes, activation="softmax"))

    else:

        model.add(layers.Dense(1, activation="relu"))

        model.add(layers.Lambda(lambda a: 4 - K.relu(-a + 4)))



    return model
def quadratic_kappa_coefficient(y_true, y_pred):

    y_true = K.cast(y_true, "float32")

    n_classes = K.cast(y_pred.shape[-1], "float32")

    weights = K.arange(0, n_classes, dtype="float32") / (n_classes - 1)

    weights = (weights - K.expand_dims(weights, -1)) ** 2



    hist_true = K.sum(y_true, axis=0)

    hist_pred = K.sum(y_pred, axis=0)



    E = K.expand_dims(hist_true, axis=-1) * hist_pred

    E = E / K.sum(E, keepdims=False)



    O = K.transpose(K.transpose(y_true) @ y_pred)  # confusion matrix

    O = O / K.sum(O)



    num = weights * O

    den = weights * E



    QWK = (1 - K.sum(num) / K.sum(den))

    return QWK



def quadratic_kappa_loss(scale=2.0):

    def _quadratic_kappa_loss(y_true, y_pred):

        QWK = quadratic_kappa_coefficient(y_true, y_pred)

        loss = -K.log(K.sigmoid(scale * QWK))

        return loss

        

    return _quadratic_kappa_loss
y_true   = np.asarray([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

y_pred_1 = np.asarray([[0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1]])

y_pred_2 = np.asarray([[0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0]])



qwk_loss = quadratic_kappa_loss(scale=2.0)

print("for pred_1")

print("sklearn QWK: {}".format(cohen_kappa_score(np.argmax(y_true, axis=1), np.argmax(y_pred_1, axis=1), weights='quadratic')))

print("keras QWK: {}".format(K.eval(quadratic_kappa_coefficient(K.variable(y_true), K.variable(y_pred_1)))))

print("keras QWK loss: {}\n".format(K.eval(qwk_loss(K.variable(y_true), K.variable(y_pred_1)))))



print("for pred_2")

print("sklearn QWK for pred 2: {}".format(cohen_kappa_score(np.argmax(y_true, axis=1), np.argmax(y_pred_2, axis=1), weights='quadratic')))

print("keras QWK for pred 2: {}".format(K.eval(quadratic_kappa_coefficient(K.variable(y_true), K.variable(y_pred_2)))))

print("keras QWK loss for pred 2: {}\n".format(K.eval(qwk_loss(K.variable(y_true), K.variable(y_pred_2)))))



print("for fully correct output")

print("sklearn QWK: {}".format(cohen_kappa_score(np.argmax(y_true, axis=1), np.argmax(y_true, axis=1), weights='quadratic')))

print("keras QWK: {}".format(K.eval(quadratic_kappa_coefficient(K.variable(y_true), K.variable(y_true)))))

print("keras QWK: {}\n".format(K.eval(qwk_loss(K.variable(y_true), K.variable(y_true)))))
sometimes05 = lambda aug: iaa.Sometimes(0.5, aug)

sometimes01 = lambda aug: iaa.Sometimes(0.1, aug)



augment0 = iaa.Sequential(

    [

        # apply the following augmenters to most images

        iaa.Fliplr(0.5), # horizontally flip 50% of all images

        iaa.Flipud(0.5), # vertically flip 20% of all images

        # crop images by -5% to 10% of their height/width

        sometimes05(iaa.CropAndPad(

            percent=(-0.05, 0.1),

            pad_mode=ia.ALL,

            pad_cval=(0, 255)

        )),

        sometimes01(iaa.Affine(

            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis

            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)

            rotate=(-45, 45), # rotate by -45 to +45 degrees

            shear=(-16, 16), # shear by -16 to +16 degrees

            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)

            cval=(0, 255), # if mode is constant, use a cval between 0 and 255

            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)

        )),

        sometimes01(iaa.Grayscale(alpha=(0.0, 1.0))),

        iaa.OneOf([

            iaa.Multiply((0.5, 1.5), per_channel=0.5),

            iaa.FrequencyNoiseAlpha(

                exponent=(-4, 0),

                first=iaa.Multiply((0.5, 1.5), per_channel=True),

                second=iaa.ContrastNormalization((0.5, 2.0))

            )

        ]),





    ],

    random_order=True

)
def train(data_path, save_path, weights=None, initial_epoch=0, one_hot=True, is_balances_batch=True):

    image_shape = (224, 224, 3)

    

    train_data, _ = get_data(data_path=data_path, memory_allocate=False)

    train_eyes, validation_eyes = train_test_split(train_data, test_size=0.1)



    train_extractor = AugmentExtractor(augmentation=augment0, image_size=image_shape[:-1], is_normalized=True)

    test_extractor = AugmentExtractor(augmentation=None, image_size=image_shape[:-1], is_normalized=True)



    train_generator = SimpleGenerator(eyes=train_eyes,

                                      extractor=train_extractor,

                                      batch_size=32,

                                      is_balances_batch=is_balances_batch,

                                      one_hot=one_hot,

                                      mode=SimpleGenerator.TRAIN

                                      )

    validate_generator = SimpleGenerator(eyes=validation_eyes,

                                         extractor=test_extractor,

                                         batch_size=1,

                                         one_hot=one_hot,

                                         mode=SimpleGenerator.TEST

                                         )

    test_batch = train_generator[0]



    net = build_densenet121(input_shape=image_shape, classes=5)



    if one_hot:

        loss = quadratic_kappa_loss(scale=2.0)

        metrics=["accuracy"]

    else:

        loss = "MSE"

        metrics=[]

        

    optimizer = optimizers.Adam(lr=0.0001)

    net.compile(loss=loss, optimizer=optimizer, metrics=metrics)



    if weights is not None:

        net.load_weights(weights, by_name=True, skip_mismatch=True)



    callbacks = [QWK(validate_generator, one_hot=one_hot, save_path=save_path, monitor_length=4)]



    net.fit_generator(train_generator,

                      epochs=20,

                      callbacks=callbacks,

                      steps_per_epoch=None,

                      initial_epoch=initial_epoch,

                      use_multiprocessing=False,

                      workers=4)
def submit(data_path, save_path, model_path, one_hot=True):

    _, test_data = get_data(data_path=data_path, memory_allocate=False)

    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))



    extractor = SimpleExtractor(image_size=(224, 224), is_normalized=True)

    generator = SimpleGenerator(extractor=extractor,

                                eyes=test_data,

                                mode=SimpleGenerator.TEST)



    test_batch = generator[0]

    model = models.load_model(model_path)

    pred = model.predict_generator(generator=generator, verbose=1)

    if one_hot:

        pred = np.argmax(pred, axis=-1)

    else:

        pred = np.round(pred).astype("int")



    test_df['diagnosis'] = pred

    test_df.to_csv(os.path.join(save_path, 'submission.csv'), index=False)
os.listdir("../input")
data_path = os.path.join("../input", "aptos2019-blindness-detection")

weights = os.path.join("../input", "densenet121weights", "densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5")

save_path = os.path.join("result", "densenet121", "qwk_loss")

model_path = os.path.join(save_path, "final.h5")

os.makedirs(save_path, exist_ok=True)



is_balances_batch = False

one_hot = True  # for QWK loss

# one_hot = False  # for MSE loss



train(data_path=data_path,

      save_path=save_path,

      weights=weights,

      initial_epoch=0,

      one_hot=one_hot, 

      is_balances_batch=is_balances_batch)



submit(data_path=data_path, 

       save_path=save_path, 

       model_path=model_path, 

       one_hot=one_hot)