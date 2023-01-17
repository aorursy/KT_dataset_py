from keras import backend as K
from keras.models import load_model
import numpy as np
# statistics.py
import numpy as np
import math
from sklearn.metrics import confusion_matrix

goes_classes = ['quiet','A','B','C','M','X']


def flux_to_class(f: float, only_main=False):
    'maps the peak_flux of a flare to one of the following descriptors: \
    *quiet* = 1e-9, *B* >= 1e-7, *C* >= 1e-6, *M* >= 1e-5, and *X* >= 1e-4\
    See also: https://en.wikipedia.org/wiki/Solar_flare#Classification'
    decade = int(min(math.floor(math.log10(f)), -4))
    sub = round(10 ** -decade * f)
    if decade < -4: # avoiding class 10
        decade += sub // 10
        sub = max(sub % 10, 1)
    main_class = goes_classes[decade + 9] if decade >= -8 else 'quiet'
    sub_class = str(sub) if main_class != 'quiet' and only_main != True else ''
    return main_class + sub_class

def class_to_flux(c: str):
    'Inverse of flux_to_class \
    Maps a flare class (e.g. B6, M, X9) to a GOES flux value'
    if c == 'quiet':
        return 1e-9
    decade = goes_classes.index(c[0])-9
    sub = float(c[1:]) if len(c) > 1 else 1
    return round(10 ** decade * sub, 10)


#
#   See https://arxiv.org/pdf/1608.06319.pdf for details about scores and statistics
#

def true_skill_statistic(y_true, y_pred, threshold='M'):
    'Calculates the True Skill Statistic (TSS) on binarized predictions\
    It is not sensitive to the balance of the samples\
    This statistic is often used in weather forecasts (including solar weather)\
    1 = perfect prediction, 0 = random prediction, -1 = worst prediction'
    separator = class_to_flux(threshold)
    y_true = [1 if yt >= separator else 0 for yt in y_true]
    y_pred = [1 if yp >= separator else 0 for yp in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) - fp / (fp + tn)

def heidke_skill_score(y_true, y_pred, threshold='M'):
    'Claculates the Heidke skill score'
    separator = class_to_flux(threshold)
    y_true = [1 if yt >= separator else 0 for yt in y_true]
    y_pred = [1 if yp >= separator else 0 for yp in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tp + fn) / len(y_pred) * (tp + fp) / len(y_pred) + (tn + fn) / len(y_pred) * (tn + fp) / len(y_pred)



# keras_generator.py
import keras.utils.data_utils
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
import os
import datetime as dt

class SDOBenchmarkGenerator(keras.utils.data_utils.Sequence):
    'Generates data for keras \
    Inspired by https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html'
    def __init__(self, base_path, batch_size=32, dim=(4, 256, 256), channels=['magnetogram'], shuffle=True, augment=True, label_func=None, data_format="channels_last"):
        'Initialization'
        self.batch_size = batch_size
        self.base_path = base_path
        self.data_format = data_format
        self.label_func = label_func
        self.dim = dim if len(dim) == 4 else (dim + (len(channels),) if data_format=='channels_last' else (len(channels),) + dim)
        self.channels = channels
        self.time_steps = [0, 7*60, 10*60+30, 11*60+50]
        self.data = self.loadCSV(augment)
        self.shuffle = shuffle
        self.on_epoch_end()

    def loadCSV(self, augment=True):
        data = pd.read_csv(os.path.join(self.base_path, 'meta_data.csv'), sep=",", parse_dates=["start", "end"], index_col="id")

        # augment by doubling the data and flagging them to be flipped horizontally
        data['flip'] = False
        if augment:
            new_data = data.copy()
            new_data.index += '_copy'
            new_data['flip'] = True
            data = pd.concat([data, new_data])
        return data

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = [
            np.empty((self.batch_size, 1)),
            np.empty((self.batch_size, *self.dim))
        ]

        # Generate data
        data = self.data.iloc[indexes]
        X[0] = np.asarray(list(map(self.loadImg, data.index)))
        ind = np.where(data['flip'])
        X[0][ind] = X[0][ind, ::-1, ...]
        X[1] = (data['start'] - pd.Timestamp('2012-01-01 00:00:00')).view('int64')
        X[1] /= (pd.Timestamp('2018-01-01 00:00:00') - pd.Timestamp('2012-01-01 00:00:00')).view('int64')
        y = np.array(data['peak_flux'])
        if self.label_func is not None:
            y = self.label_func(y)
        return X, y

    def loadImg(self, sample_id):
        'Load the images of each timestep as channels'
        ar_nr, p = sample_id.replace('_copy','').split("_", 1)
        path = os.path.join(self.base_path, ar_nr, p)

        slices = np.zeros(self.dim)

        sample_date = dt.datetime.strptime(p[:p.rfind('_')], "%Y_%m_%d_%H_%M_%S")
        time_steps = [sample_date + dt.timedelta(minutes=offset) for offset in self.time_steps]
        for img in [name for name in os.listdir(path) if name.endswith('.jpg')]:
            img_datetime_raw, img_wavelength = os.path.splitext(img)[0].split("__")
            img_datetime = dt.datetime.strptime(img_datetime_raw, "%Y-%m-%dT%H%M%S")

            # calc wavelength and datetime index
            datetime_index = [di[0] for di in enumerate(time_steps) if abs(di[1] - img_datetime) < dt.timedelta(minutes=15)]
            if img_wavelength in self.channels and len(datetime_index) > 0:
                val = np.squeeze(img_to_array(load_img(os.path.join(path, img), grayscale=True)), 2)
                if self.data_format == 'channels_first':
                    slices[datetime_index[0],:,:,self.channels.index(img_wavelength)] = val
                else:
                    slices[self.channels.index(img_wavelength),:,:,datetime_index[0]] = val



        return slices

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

base_path = '../input/sdobenchmark/sdobenchmark_full/'

# Parameters
params = {'dim': (4, 256, 256, 4),
          'batch_size': 4,
          'channels': ['magnetogram', '304', '131', '1700'],
          'shuffle': False}

validation_generator = SDOBenchmarkGenerator(os.path.join(base_path, 'test'), **params)

# This is a custom metric I used during training. It simply maps my classes to a value, on which
# I can calculate the mean absolute error.
def converted_mae(y_true, y_pred):
    conv_y_true = 10.**(K.cast(K.argmax(y_true, axis=-1), K.floatx())-9.) * 5.
    conv_y_pred = 10.**(K.cast(K.argmax(y_pred, axis=-1), K.floatx())-9.) * 5.
    return K.mean(K.abs(conv_y_pred - conv_y_true), axis=-1)

model = load_model('../input/trainedmodelromanbolzern/best_classes', custom_objects={'converted_mae': converted_mae})
# model.summary()

Y_pred = model.predict_generator(validation_generator,
                        steps = len(validation_generator),
                        use_multiprocessing=True,
                        max_queue_size=params['batch_size'],
                        workers=params['batch_size'],
                        verbose=2
                        )
Y_pred_flux = 10. ** (np.argmax(Y_pred, axis=-1)-9.)*5
Y_val = np.asarray(validation_generator.data.iloc[validation_generator.indexes]['peak_flux'])

print(f'Mean absolute error:  {np.mean(np.abs(Y_val-Y_pred_flux))}')
print('TSS: ' + str(true_skill_statistic(Y_val, Y_pred_flux)))
print(f'(categorical_accuracy: {np.mean(np.equal(np.floor(np.log10(Y_val)+9).astype(int), np.argmax(Y_pred, axis=-1)).astype(np.float32))})')
X_index = np.where(Y_val >= 1e-4)[0]
print(f'{np.sum((Y_pred_flux[X_index]>=1e-4).astype(int))} of {len(X_index)} were correctly classified as X-flares')
print(f'{len(np.where(np.logical_and(Y_val < 1e-4, Y_pred_flux >= 1e-4))[0])} were wrongly classified as X-flares')
print(f'(categorical_accuracy: {np.mean(np.equal(np.floor(np.log10(Y_val[X_index])+9).astype(int), np.argmax(Y_pred[X_index], axis=-1)).astype(np.float32))})')
september_index = np.where(np.logical_and(validation_generator.data.iloc[validation_generator.indexes].index.str.startswith('12673'), Y_val >= 1e-4))
validation_generator.data['peak_class'] = [flux_to_class(f) for f in validation_generator.data['peak_flux']]
validation_generator.data['predicted_class'] = [flux_to_class(float(f), only_main=True) for f in Y_pred_flux]
validation_generator.data.iloc[september_index]