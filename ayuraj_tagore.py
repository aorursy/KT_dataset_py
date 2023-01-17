import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline



import cv2

import os



from sklearn.preprocessing import MinMaxScaler



import keras.backend as K

from keras.utils import Sequence

from keras.utils import to_categorical

from keras.applications import VGG19

from keras.models import Model

from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization

from spatialpp import SpatialPyramidPooling



print(os.listdir("../input"))
artist_df = pd.read_csv("../input/best-artworks-of-all-time/artists.csv")
artist_df.head()
trainDirPath = "../input/best-artworks-of-all-time/resized/resized"

trainImgs = os.listdir(trainDirPath)
samples = np.random.choice(len(trainImgs), 4)



def show_images(images, cols = 1, titles = None):

    assert((titles is None)or (len(images) == len(titles)))

    n_images = len(images)

    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]

    fig = plt.figure()

    for n, (image, title) in enumerate(zip(images, titles)):

        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)

        if image.ndim == 2:

            plt.gray()

        plt.imshow(image)

        a.set_title(title, fontsize=15)

        a.grid(False)

        a.axis("off")

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

plt.show()
sampleImages = []

titles = []

for sample in samples:

    img = cv2.imread(trainDirPath+'/'+trainImgs[sample])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(img.shape)

    img = cv2.resize(img, (400,400))

    sampleImages.append(img)

    titles.append(trainImgs[sample][:-4])

    
show_images(sampleImages, cols=2, titles=titles)
model = VGG19()
model.summary()
## Model defined

x = Input(shape=(None,None,3))

c = model.get_layer('block1_conv1')(x)

for i in range(2, 22):

    c = model.get_layer(index=i)(c)

    

c = SpatialPyramidPooling([2, 4, 6])(c)

c = Dense(4096, activation='relu')(c)



for i in range(24, 26):

    c = model.get_layer(index=i)(c)



c = Dense(50, activation='softmax')(c)



tagoreModel = Model(inputs=x, outputs=c)
tagoreModel.summary()
print("Total number of images: ", len(trainImgs))
labelList = artist_df['name'].values

labelList = [label.replace(" ", "_") for label in labelList]
partitions = {'train': [],

             'validation': []}

labels = {}



for ID in range(7000):

    partitions['train'].append(trainImgs[ID])

    try:

        labels[trainImgs[ID]] = labelList.index(''.join(x for x in trainImgs[ID][:-4] if not x.isdigit())[:-1])

    except (IndexError, ValueError):

        partitions['train'].pop()



for ID in range(7000, len(trainImgs)):

    partitions['validation'].append(trainImgs[ID])

    try:

        labels[trainImgs[ID]] = labelList.index(''.join(x for x in trainImgs[ID][:-4] if not x.isdigit())[:-1])

    except (IndexError, ValueError):

        partitions['validation'].pop()
print("Data to be trained on:", len(partitions['train'])+len(partitions['validation']))

print("data lost: ", len(trainImgs)- len(partitions['train'])-len(partitions['validation']))
class DataGenerator(Sequence):

    def __init__(self, Data, labels, dataDir, num_classes, batch_size = 32, shuffle=True, resize=False, resized_dim=(None,None)):

        self.Data = Data

        self.labels = labels

        self.dataDir = dataDir

        self.num_classes = num_classes

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.resize = resize

        self.resized_dim = resized_dim

        self.on_epoch_end()

        

    def on_epoch_end(self):

        ## update indexes after epoch end

        self.indexes = np.arange(len(self.Data))

        if self.shuffle is True:

            np.random.shuffle(self.indexes)

        

    def __len__(self):

        ## Number of data in batch per epoch

        return int(np.floor(len(self.Data)/self.batch_size))

    

    def __getitem__(self, index):

        ## Return X,y for each batch index

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        Data_temp = [self.Data[i] for i in indexes]

        X, y = self._DataGeneration(Data_temp)

        

        return (X,y)

    

    def _DataGeneration(self, Data_temp):

        X = []

        y = []

        sizes = []

        

#         for data in Data_temp:

#             img = cv2.imread(self.dataDir+'/'+data)

#             sizes.append(img.shape)

            

#         H,W,C = min(sizes)

        H,W,C = 400,400,3

        

        for data in Data_temp:

            img = cv2.imread(self.dataDir+'/'+data)

            img = cv2.resize(img, (H,W)) 

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.resize is True:

                img = cv2.resize(img, self.resized_dim)

            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            X.append(img)

            y.append(self.labels[data])

            

        return np.array(X).reshape(self.batch_size,H,W,3), to_categorical(np.array(y), self.num_classes, dtype='float32')
training_generator = DataGenerator(partitions['train'], labels, trainDirPath, num_classes=50)

validation_generator = DataGenerator(partitions['validation'], labels, trainDirPath, num_classes=50)
tagoreModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = tagoreModel.fit_generator(generator=training_generator, steps_per_epoch=10, epochs=10, validation_data=validation_generator, use_multiprocessing=True,

                         workers=10, max_queue_size=15)