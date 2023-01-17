import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from keras.models import Sequential, Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import img_to_array

import librosa
import os
import multiprocessing.pool
from functools import partial

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train_path = '../input/train/audio/'
test_path = '../input/test.7z'

#
# The classes correspond to directory names under ../train/audio
classnames = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
def spect_loader(path, window_size, window_stride, window, normalize, max_len=101, logit=True):
    y, sr = librosa.load(path, sr=None)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    if logit==True:
        #S = log(S+1)
        spect = np.log1p(spect)
    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    #spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = np.mean(np.ravel(spect))
        std = np.std(np.ravel(spect))
        if std != 0:
            spect = spect -mean
            spect = spect / std

    return spect

def _count_valid_files_in_directory(directory, white_list_formats, follow_links):
    """Count files with extension in `white_list_formats` contained in a directory.
    # Arguments
        directory: absolute path to the directory containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    samples = 0
    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                samples += 1
    return samples

def _list_valid_filenames_in_directory(directory, white_list_formats,
                                       class_indices, follow_links):
    """List paths of files in `subdir` relative from `directory` whose extensions are in `white_list_formats`.
    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        class_indices: dictionary mapping a class name to its index.
    # Returns
        classes: a list of class indices
        filenames: the path of valid files in `directory`, relative from
            `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be ["class1/file1.jpg", "class1/file2.jpg", ...]).
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    classes = []
    filenames = []
    subdir = os.path.basename(directory)
    basedir = os.path.dirname(directory)
    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                classes.append(class_indices[subdir])
                # add filename relative to directory
                absolute_path = os.path.join(root, fname)
                filenames.append(os.path.relpath(absolute_path, basedir))
    return classes, filenames

class SpeechDirectoryIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.
    # Arguments
       
    """

    def __init__(self, directory, window_size, window_stride, 
                 window_type, normalize, max_len=101, logit=True,
                 target_size=(256, 256), color_mode='grayscale',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None,
                 save_prefix='', save_format='png',
                 follow_links=False, interpolation='nearest'):
        if data_format is None:
            data_format = K.image_data_format()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len
        self.directory = directory
        self.logit = logit
#        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'wav'}

        # first, count the number of samples and classes
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_files_in_directory,
                                   white_list_formats=white_list_formats,
                                   follow_links=follow_links)
        self.samples = sum(pool.map(function_partial,
                                    (os.path.join(directory, subdir)
                                     for subdir in classes)))

        print('Found %d images belonging to %d classes.' % (self.samples, self.num_classes))

        # second, build an index of the images in the different class subfolders
        results = []

        self.filenames = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(pool.apply_async(_list_valid_filenames_in_directory,
                                            (dirpath, white_list_formats,
                                             self.class_indices, follow_links)))
            
        
        for res in results:
            classes, filenames = res.get()
            self.classes[i:i + len(classes)] = classes
            self.filenames += filenames
            if i==0:
                img = spect_loader(os.path.join(self.directory, filenames[0]), 
                               self.window_size, 
                               self.window_stride, 
                               self.window_type, 
                               self.normalize, 
                               self.max_len, 
                               self.logit) 
                img=np.swapaxes(img, 0, 2)
                self.target_size = tuple((img.shape[0], img.shape[1]))
                print(self.target_size)
                if self.color_mode == 'rgb':
                    if self.data_format == 'channels_last':
                        self.image_shape = self.target_size + (3,)
                    else:
                        self.image_shape = (3,) + self.target_size
                else:
                    if self.data_format == 'channels_last':
                        self.image_shape = self.target_size + (1,)
                    else:
                        self.image_shape = (1,) + self.target_size
                        
            i += len(classes)
        pool.close()
        pool.join()
        super(SpeechDirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)
    

    
    
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_f = []
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        #print(index_array)
        for i, j in enumerate(index_array):
            #print(i, j, self.filenames[j])
            fname = self.filenames[j]
            #img = load_img(os.path.join(self.directory, fname),
            #               grayscale=grayscale,
            #               target_size=self.target_size,
            #               interpolation=self.interpolation)
            img = spect_loader(os.path.join(self.directory, fname), 
                               self.window_size, 
                               self.window_stride, 
                               self.window_type, 
                               self.normalize, 
                               self.max_len)
            img=np.swapaxes(img, 0, 2)
            
            x = img_to_array(img, data_format=self.data_format)
            #x = self.image_data_generator.random_transform(x)
            #x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_f.append(fname)
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e7),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)[0]
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
    
window_size=.02
window_stride=.01
window_type='hamming'
normalize=True
max_len=101
batch_size = 64
train_iterator = SpeechDirectoryIterator(directory=train_path, 
                                   batch_size=batch_size, 
                                   window_size=window_size, 
                                   window_stride=window_stride, 
                                   window_type=window_type,
                                   normalize=normalize, 
                                   max_len=max_len, 
                                   classes=classnames, 
                                   shuffle=True, 
                                   seed=123)

train_iterator.reset()
X, y = next(train_iterator)
print(X.shape)
f, axarr = plt.subplots(3, 3)
f.set_figheight(8)
f.set_figwidth(15)
for i in range(9):
    axarr[int(i/3), i%3].imshow(X[i, ..., 0], cmap='gray')
    axarr[int(i/3), i%3].set_title(classnames[np.argmax(y[i])])
plt.show()
#plt.imshow(X[0, ..., 0], cmap='gray')
#plt.title(classnames[np.argmax(y[0])])
#plt.show()
train_iterator_2 = SpeechDirectoryIterator(directory=train_path, 
                                   batch_size=batch_size, 
                                   window_size=window_size, 
                                   window_stride=window_stride, 
                                   window_type=window_type,
                                   normalize=normalize, 
                                   logit = False, 
                                   max_len=61, 
                                   classes=classnames, 
                                   shuffle=True, 
                                   seed=123)
train_iterator_2.reset()
X, y = next(train_iterator_2)
print(X.shape)
f, axarr = plt.subplots(3, 3)
f.set_figheight(8)
f.set_figwidth(15)
for i in range(9):
    axarr[int(i/3), i%3].imshow(X[i, ..., 0], cmap='gray')
    axarr[int(i/3), i%3].set_title(classnames[np.argmax(y[i])])
plt.show()
model = Sequential()
model.add(Conv2D(12, (5, 5), activation = 'relu', input_shape=train_iterator.image_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(25, (5, 5), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(180, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classnames), activation = 'softmax')) #Last layer with one output per class

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
model.summary()
# Make the model learn
model.fit_generator(train_iterator,
        steps_per_epoch=np.ceil(train_iterator.n / batch_size),
        epochs=3,
        verbose=1)
predict_iterator = SpeechDirectoryIterator(directory=test_path, 
                                   batch_size=batch_size, 
                                   window_size=window_size, 
                                   window_stride=window_stride, 
                                   window_type=window_type,
                                   normalize=normalize, 
                                   max_len=max_len, 
                                   classes=None,
                                   shuffle=False)
preds = model.predict_generator(generator=predict_iterator, 
                        steps=int(np.ceil(predict_iterator.n)/batch_size), 
                        verbose=1)
