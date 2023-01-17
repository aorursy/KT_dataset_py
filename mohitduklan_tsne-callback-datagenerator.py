# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import tensorflow as tf

import tensorflow.keras.backend as k

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten



from keras.preprocessing.image import ImageDataGenerator



from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

csvData = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

csvTest = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

csvSample = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
csvFeat = csvData.drop("label", axis=1)

csvLabel = csvData["label"]

csvFeat /= 255.

csvTest = csvTest/255.
csvData.head()
csvData.keys()
print(f'Total Pixels: {len(csvFeat.keys())}')

print(f'Labels: {sorted(pd.unique(csvLabel))}')

print(f'Min and Max value of pixel: {csvFeat.values.min()}, {csvFeat.values.max()}')
csvFeatImg = csvFeat.values.reshape([-1, 28, 28, 1])

csvTestImg = csvTest.values.reshape([-1, 28, 28, 1])
sns.countplot(csvData["label"])

plt.title("Label Count")
tsne = TSNE(n_components=2, n_iter=500, n_iter_without_progress=50, verbose=20)

samp = csvData.sample(frac=.05, replace=False)

sampY, sampX = samp.iloc[:,:1], samp.iloc[:,1:]

components = tsne.fit_transform(sampX)
plt.figure(figsize=(16,10))

sns.scatterplot(

    x=components[:,0], y=components[:,1],

    hue=np.ravel(sampY),

    palette=sns.color_palette("hls", 10),

    legend="full",

    alpha=0.7

)
fig, ax = plt.subplots(2, 5, figsize=(14,8))

ax = np.ravel(ax)

for i in range(10):

    ax[i].imshow(csvFeatImg[i].squeeze(),cmap='gray')

    plt.axis("off")

plt.show()
csvTrainX, csvValX, csvTrainY, csvValY = train_test_split(csvFeatImg, csvLabel, test_size=.1, stratify=csvLabel)

csvTrainX.shape, csvTrainY.shape, csvValX.shape, csvValY.shape
trainGen = ImageDataGenerator(

        rotation_range=15,

        zoom_range = 0.15,

        width_shift_range=0.15,

        height_shift_range=0.15)

batchSize = 100

hidddenN = 300

lastN = 10

checkpoint = ModelCheckpoint("/kaggle/working/best_model.hdf5", monitor='val_sparse_categorical_accuracy', verbose=1,

    save_best_only=True, mode='auto', period=1)
k.clear_session()

inputData = Input(shape = [28, 28, 1], dtype = tf.float32)

hiddenLayer_ = Flatten()(inputData)

hiddenLayer = Dense(hidddenN, activation = 'relu')(hiddenLayer_)

output = tf.keras.layers.Dense(lastN, activation='softmax')(hiddenLayer)

model = Model(inputs = inputData, outputs = output)

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics='sparse_categorical_accuracy'

)

model.summary()
gen = trainGen.flow(csvTrainX, csvTrainY, batch_size=150, shuffle=False)

hist = model.fit_generator(gen, epochs = 60, validation_data=(csvValX, csvValY), callbacks=[checkpoint])
def report(model, x, y):

    p = np.argmax(model.predict(x), axis = 1)

    classReport = classification_report(p, y)

    accReport = accuracy_score(p, y)

    print(classReport, accReport)
report(model, csvTrainX, csvTrainY)
report(model, csvValX, csvValY)
model.load_weights("/kaggle/working/best_model.hdf5")
report(model, csvTrainX, csvTrainY)
report(model, csvValX, csvValY)
testPrediction = np.argmax(model.predict(csvTest), axis=1)

d = {

        "ImageId": list(range(1, len(testPrediction)+1)),

        "Label": testPrediction.tolist()

    }

submission = pd.DataFrame(d)

submission.to_csv("/kaggle/working/submit.csv", index = False)