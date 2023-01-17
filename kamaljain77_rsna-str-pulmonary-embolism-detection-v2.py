import numpy as np

import pandas as pd



import os

import gc

import time

from IPython.display import clear_output



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D



from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import ModelCheckpoint as MC

from tensorflow.keras import backend as K

root = '/kaggle/input/rsna-str-pulmonary-embolism-detection'

for item in os.listdir(root):

    path = os.path.join(root, item)

    if os.path.isfile(path):

        print(path)
print("Loading Training Data ...")

train = pd.read_csv("/kaggle/input/rsna-str-pulmonary-embolism-detection/train.csv")

train.head()
train.shape
train.describe()
print("Loading Testing Data...")

test = pd.read_csv("/kaggle/input/rsna-str-pulmonary-embolism-detection/test.csv")

print(test.shape)
test.head()
print(" Loading Sample Output ...")

ss = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/sample_submission.csv")

print(ss.shape)

ss.head()
ids = ss.id

counter = [1 for _ in range(10)]

mapper = []

for i in ids:

    n = '_'.join(i.split('_')[1:])

    if n not in mapper:

        mapper.append(n)

    else:

        counter[mapper.index(n)] +=1

print("List of keys: ")

print(mapper, sep = '\n')

print()

print("Count of items per key: ")

print(counter)

    
import vtk

from vtk.util import numpy_support

import cv2



reader = vtk.vtkDICOMImageReader()

def get_img(path):

    reader.SetFileName(path)

    reader.Update()

    _extent = reader.GetDataExtent()

    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

    ConstPixelSpacing = reader.GetPixelSpacing()

    imageData = reader.GetOutput()

    pointData = imageData.GetPointData()

    arrayData = pointData.GetArray(0)

    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)

    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order = 'F')

    ArrayDicom = cv2.resize(ArrayDicom, (512, 512))

    return ArrayDicom
# Let's load a dcom file and view it



fpath = "../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf/00ac73cfc372.dcm"

ds = get_img(fpath)



import matplotlib.pyplot as plt



# Convert dcom file to 8 bit color



func = lambda x: int((2**15 + x)*(255/2**16))

int16_to_uint8 = np.vectorize(func)



def show_dicom_images(dcom):

    f, ax = plt.subplots(1, 2, figsize=(16, 20))

    data_row_img = int16_to_uint8(ds)

    ax[0].imshow(data_row_img, cmap=plt.cm.bone)

    ax[1].imshow(ds, cmap=plt.cm.bone)

    

    ax[0].axis("off")

    ax[0].set_title('8-bit DICOM Image')

    ax[1].axis('off')

    ax[1].set_title('16-bit DICOM Image')

    plt.show()



show_dicom_images(ds)
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D



inputs = Input((512, 512, 3))

#x = Conv2D(3, (1, 1), activation='relu')(inputs)

base_model = keras.applications.Xception(

    include_top=False,

    weights="imagenet"

)



base_model.trainable = False



outputs = base_model(inputs, training=False)

outputs = keras.layers.GlobalAveragePooling2D()(outputs)

outputs = Dropout(0.25)(outputs)

outputs = Dense(1024, activation='relu')(outputs)

outputs = Dense(256, activation='relu')(outputs)

outputs = Dense(64, activation='relu')(outputs)

ppoi = Dense(1, activation='sigmoid', name='pe_present_on_image')(outputs)

rlrg1 = Dense(1, activation='sigmoid', name='rv_lv_ratio_gte_1')(outputs)

rlrl1 = Dense(1, activation='sigmoid', name='rv_lv_ratio_lt_1')(outputs) 

lspe = Dense(1, activation='sigmoid', name='leftsided_pe')(outputs)

cpe = Dense(1, activation='sigmoid', name='chronic_pe')(outputs)

rspe = Dense(1, activation='sigmoid', name='rightsided_pe')(outputs)

aacpe = Dense(1, activation='sigmoid', name='acute_and_chronic_pe')(outputs)

cnpe = Dense(1, activation='sigmoid', name='central_pe')(outputs)

indt = Dense(1, activation='sigmoid', name='indeterminate')(outputs)



model = Model(inputs=inputs, outputs={'pe_present_on_image':ppoi,

                                      'rv_lv_ratio_gte_1':rlrg1,

                                      'rv_lv_ratio_lt_1':rlrl1,

                                      'leftsided_pe':lspe,

                                      'chronic_pe':cpe,

                                      'rightsided_pe':rspe,

                                      'acute_and_chronic_pe':aacpe,

                                      'central_pe':cnpe,

                                      'indeterminate':indt})



opt = keras.optimizers.Adam(lr=0.001)



model.compile(optimizer=opt,

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.summary()

model.save('pe_detection_model.h5')

del model

K.clear_session()

gc.collect()
# Image Generator



def convert_to_rgb(array):

    array = array.reshape((512, 512, 1))

    return np.stack([array, array, array], axis=2).reshape((512, 512, 3))

    

def custom_dcom_image_generator(batch_size, dataset, test=False, debug=False):

    

    fnames = dataset[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']]

    

    if not test:

        Y = dataset[['pe_present_on_image', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1', 'leftsided_pe',

                     'chronic_pe', 'rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate'

                    ]]

        prefix = 'input/rsna-str-pulmonary-embolism-detection/train'

        

    else:

        prefix = 'input/rsna-str-pulmonary-embolism-detection/test'

    

    X = []

    batch = 0

    for st, sr, so in fnames.values:

        if debug:

            print(f"Current file: ../{prefix}/{st}/{sr}/{so}.dcm")



        dicom = get_img(f"../{prefix}/{st}/{sr}/{so}.dcm")

        image = convert_to_rgb(dicom)

        X.append(image)

        

        del st, sr, so

        

        if len(X) == batch_size:

            if test:

                yield np.array(X)

                del X

            else:

                yield np.array(X), Y[batch*batch_size:(batch+1)*batch_size].values

                del X

                

            gc.collect()

            X = []

            batch += 1

        

    if test:

        yield np.array(X)

    else:

        yield np.array(X), Y[batch*batch_size:(batch+1)*batch_size].values

        del Y

    del X

    gc.collect()

    return
# Training the model with train data



history = {}

start = time.time()

debug = 0

batch_size = 1000

train_size = int(batch_size*0.9)



max_train_time = 3600 * 4 #hours to seconds of training



checkpoint = MC(filepath='../working/pe_detection_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

#Train loop

for n, (x, y) in enumerate(custom_dcom_image_generator(batch_size, train.sample(frac=1), False, debug)):

    

    if len(x) < 10: #Tries to filter out empty or short data

        break

        

    clear_output(wait=True)

    print("Training batch: %i - %i" %(batch_size*n, batch_size*(n+1)))

    

    model = load_model('../working/pe_detection_model.h5')

    hist = model.fit(

        x[:train_size], #Y values are in a dict as there's more than one target for training output

        {'pe_present_on_image':y[:train_size, 0],

         'rv_lv_ratio_gte_1':y[:train_size, 1],

         'rv_lv_ratio_lt_1':y[:train_size, 2],

         'leftsided_pe':y[:train_size, 3],

         'chronic_pe':y[:train_size, 4],

         'rightsided_pe':y[:train_size, 5],

         'acute_and_chronic_pe':y[:train_size, 6],

         'central_pe':y[:train_size, 7],

         'indeterminate':y[:train_size, 8]},



        callbacks = checkpoint,



        validation_split=0.2,

        epochs=3,

        batch_size=8,

        verbose=debug

    )

    

    print("Metrics for batch validation:")

    model.evaluate(x[train_size:],

                   {'pe_present_on_image':y[train_size:, 0],

                    'rv_lv_ratio_gte_1':y[train_size:, 1],

                    'rv_lv_ratio_lt_1':y[train_size:, 2],

                    'leftsided_pe':y[train_size:, 3],

                    'chronic_pe':y[train_size:, 4],

                    'rightsided_pe':y[train_size:, 5],

                    'acute_and_chronic_pe':y[train_size:, 6],

                    'central_pe':y[train_size:, 7],

                    'indeterminate':y[train_size:, 8]

                   }

                  )

    

    try:

        for key in hist.history.keys():

            history[key] = np.concatenate([history[key], hist.history[key]], axis=0)

    except:

        for key in hist.history.keys():

            history[key] = hist.history[key]

            

    #To make sure that our model don't train overtime

    if time.time() - start >= max_train_time:

        print("Time's up!")

        break

        

    model.save('pe_detection_model.h5')

    del model, x, y, hist

    K.clear_session()

    gc.collect()
for key in history.keys():

    if key.startswith('val'):

        continue

    else:

        epoch = range(len(history[key]))

        plt.plot(epoch, history[key]) #X=epoch, Y=value

        plt.plot(epoch, history['val_'+key])

        plt.title(key)

        if 'accuracy' in key:

            plt.axis([0, len(history[key]), -0.1, 1.1]) #Xmin, Xmax, Ymin, Ymax

        plt.legend(['train', 'validation'], loc='upper right')

        plt.show()