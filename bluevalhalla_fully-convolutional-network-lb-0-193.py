import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pdb

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Lambda, Activation

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras import backend as K

from matplotlib import pyplot as plt

%matplotlib inline

import scipy

from scipy import misc, ndimage

from scipy.ndimage.interpolation import zoom

from scipy.ndimage import imread
train = pd.read_json('../input/train.json')
def get_images(df):

    '''Create 3-channel 'images'. Return rescale-normalised images.'''

    images = []

    for i, row in df.iterrows():

        # Formulate the bands as 75x75 arrays

        band_1 = np.array(row['band_1']).reshape(75, 75)

        band_2 = np.array(row['band_2']).reshape(75, 75)

        band_3 = band_1 / band_2



        # Rescale

        r = (band_1 - band_1.min()) / (band_1.max() - band_1.min())

        g = (band_2 - band_2.min()) / (band_2.max() - band_2.min())

        b = (band_3 - band_3.min()) / (band_3.max() - band_3.min())



        rgb = np.dstack((r, g, b))

        images.append(rgb)

    return np.array(images)
X = get_images(train)
y = to_categorical(train.is_iceberg.values,num_classes=2)
Xtr, Xv, ytr, yv = train_test_split(X, y, shuffle=False, test_size=0.20)
def ConvBlock(model, layers, filters):

    '''Create [layers] layers consisting of zero padding, a convolution with [filters] 3x3 filters and batch normalization. Perform max pooling after the last layer.'''

    for i in range(layers):

        model.add(ZeroPadding2D((1, 1)))

        model.add(Conv2D(filters, (3, 3), activation='relu'))

        model.add(BatchNormalization(axis=3))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
def create_model():

    '''Create the FCN and return a keras model.'''



    model = Sequential()



    # Input image: 75x75x3

    model.add(Lambda(lambda x: x, input_shape=(75, 75, 3)))

    ConvBlock(model, 1, 32)

    # 37x37x32

    ConvBlock(model, 1, 64)

    # 18x18x64

    ConvBlock(model, 1, 128)

    # 9x9x128

    ConvBlock(model, 1, 128)

    # 4x4x128

    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(2, (3, 3), activation='relu'))

    model.add(GlobalAveragePooling2D())

    # 4x4x2

    model.add(Activation('softmax'))

    

    return model
# Create the model and compile

model = create_model()

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.summary()
init_epo = 0

num_epo = 30

end_epo = init_epo + num_epo
print ('lr = {}'.format(K.get_value(model.optimizer.lr)))

history = model.fit(Xtr, ytr, validation_data=(Xv, yv), batch_size=32, epochs=end_epo, initial_epoch=init_epo)

init_epo += num_epo

end_epo = init_epo + num_epo
l = model.layers

conv_fn = K.function([l[0].input, K.learning_phase()], [l[-4].output])
def get_cm(inp, label):

    '''Convert the 4x4 layer data to a 75x75 image.'''

    conv = np.rollaxis(conv_fn([inp,0])[0][0],2,0)[label]

    return scipy.misc.imresize(conv, (75,75), interp='nearest')
def info_img (im_idx):

    '''Generate heat maps for the boat (boatness) and iceberg (bergness) for image im_idx.'''

    if (yv[im_idx][1] == 1.0):

        img_type = 'iceberg'

    else:

        img_type = 'boat'

    inp = np.expand_dims(Xv[im_idx], 0)

    img_guess = np.round(model.predict(inp)[0],2)

    if (img_guess[1] > 0.5):

        guess_type = 'iceberg'

    else:

        guess_type = 'boat'

    cm0 = get_cm(inp, 0)

    cm1 = get_cm(inp, 1)

    print ('truth: {}'.format(img_type))

    print ('guess: {}, prob: {}'.format(guess_type, img_guess))

    plt.figure(1,figsize=(10,10))

    plt.subplot(121)

    plt.title('Boatness')

    plt.imshow(Xv[im_idx])

    plt.imshow(cm0, cmap="cool", alpha=0.5)

    plt.subplot(122)

    plt.title('Bergness')

    plt.imshow(Xv[im_idx])

    plt.imshow(cm1, cmap="cool", alpha=0.5)
info_img(13)
test = pd.read_json('../input/test.json')

Xtest = get_images(test)

test_predictions = model.predict_proba(Xtest)

submission = pd.DataFrame({'id': test['id'], 'is_iceberg': test_predictions[:, 1]})

submission.to_csv('sub_fcn.csv', index=False)
submission.head(5)