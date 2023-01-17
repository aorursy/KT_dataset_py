import pandas as pd

import numpy as np

import random

import os



import cv2 as cv

from skimage import filters

from skimage import morphology



import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.figure_factory as ff

import matplotlib.pyplot as plt



from kaggle_datasets import KaggleDatasets

import tensorflow as tf

print(f"Tensorflow version: {tf.__version__}")

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score



SEED = 5
INPUT_PATH = '/kaggle/input/plant-pathology-2020-fgvc7/'

IMG_PATH = INPUT_PATH + 'images/'

TRAIN_DATA = INPUT_PATH + 'train.csv'

TEST_DATA = INPUT_PATH + 'test.csv'

SAMPLE_SUB = INPUT_PATH + 'sample_submission.csv'
train_df = pd.read_csv(TRAIN_DATA)

test_df = pd.read_csv(TEST_DATA)

sampleSubmission_df = pd.read_csv(SAMPLE_SUB)
EDA_IMG_SHAPE = (512,256)



def getImage(image_id,SHAPE=EDA_IMG_SHAPE):

    img = cv.imread(IMG_PATH + image_id + '.jpg')

    img = cv.resize(img,SHAPE)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    

    return img
%%time



healthy = [getImage(image_id) for image_id in train_df[train_df['healthy']==1].iloc[:,0]]



multiple_diseases = [getImage(image_id) for image_id in train_df[train_df['multiple_diseases']==1].iloc[:,0]]



rust = [getImage(image_id) for image_id in train_df[train_df['rust']==1].iloc[:,0]]



scab = [getImage(image_id) for image_id in train_df[train_df['scab']==1].iloc[:,0]]
classes = {'healthy':healthy, 'multiple_diseases':multiple_diseases, 'rust':rust, 'scab': scab} 
def plotlyDataFrame(df,title):

    

    fig = go.Figure(data=[go.Table(

    header = dict(values = df.columns),

    cells = dict(values = [df[col] for col in df.columns]))])

    

    fig.update_layout(

        title = title)

    

    fig.show()
plotlyDataFrame(train_df.iloc[:15,:],'Train Data')
plotlyDataFrame(test_df.iloc[:15,:],'Test Data')
plotlyDataFrame(sampleSubmission_df.iloc[:15,:], 'Sample Submission')
fig = go.Figure(data=[go.Pie(labels=train_df.columns[1:],

                             values=[np.sum(train_df[col]) for col in train_df.columns[1:]])])



fig.update_traces(hoverinfo='label+percent',

                  textinfo='value',

                  textfont_size=20,

                  marker=dict(line=dict(color='#000000', width=2)))



fig.update_layout(title_text="Target Distribution of Training-Data ")



fig.show()
fig = go.Figure(go.Parcats(dimensions=[dict(values=train_df[col],label=col) for col in train_df.columns[1:]],

                          line={'color':train_df.healthy, 'colorscale':[[0,'red'],[1,'green']]}))



fig.update_layout(title='Parallel Categorical Plot')



fig.show()
img = cv.imread(IMG_PATH + random.choice(train_df.iloc[:,0]) + '.jpg')

img = cv.resize(img,(256,128))

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)





fig = make_subplots(2,1)



fig.add_trace(go.Image(z=img),1,1)



for channel, color in enumerate(['red','green','blue']):

    fig.add_trace(go.Histogram(x=img[:,:,channel].ravel(),

                               opacity=0.5,

                               marker_color=color,

                               name=f'{color} channel'),2,1)



fig.update_layout(title='Image & its Channel Distribution')

    

fig.show()
def displayImages(condition='healthy'):



#     fig = make_subplots(3,3,horizontal_spacing=0.01,vertical_spacing=0.05)



#     for i in range(9):

#         image = random.choice(classes[condition])

#         fig.add_trace(go.Image(z=image),i//3 + 1,i%3 + 1)



#     fig.update_layout(title = f'{condition.capitalize()} Leaves Images',height=128*3 + 50,width=256*3 + 50)

    

#     fig.update_xaxes(showticklabels=False)

#     fig.update_yaxes(showticklabels=False)

    

#     fig.show()



    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 10))



    for i in range(9):

        

        image = random.choice(classes[condition])

        

        ax[i//3,i%3].imshow(image)

        

    fig.suptitle(f'{condition.capitalize()} Class Leaves',fontsize=20)

    

    plt.show()
displayImages('healthy')
colors = ['rgb(200, 0, 0)', 'rgb(0, 200, 0)', 'rgb(0,0,200)']



def plotChannelDistribution(condition):

    

    distributions = []

        

    for channel in range(3):

        distributions.append([np.mean(img[:,:,channel]) for img in classes[condition]])

    

    fig = ff.create_distplot(distributions,

                            group_labels=['red','green','blue'],

                            colors=colors)

    

    fig.update_layout(title=f'{condition.capitalize()} leaves channel distribution')

    

    fig.show()
plotChannelDistribution('healthy')
displayImages('rust')
plotChannelDistribution('rust')
displayImages('scab')
plotChannelDistribution('scab')
displayImages('multiple_diseases')
plotChannelDistribution('multiple_diseases')
channelDict = {'red':0,'green':1,'blue':2}



group_labels=[train_df.columns[i] for i in range(1,5)]



colors_cw = {'red':['rgb(250,0,0)','rgb(190,0,0)','rgb(130,0,0)','rgb(50,0,0)'],

         'green':['rgb(0,250,0)','rgb(0,190,0)','rgb(0,130,0)','rgb(0,50,0)'],

         'blue':['rgb(0,0,250)','rgb(0,0,190)','rgb(0,0,130)','rgb(0,0,50)']}



def plotChannelWiseDistribution(channel):

    

    distributions = []

    

    for c in [healthy, multiple_diseases, rust, scab]:

        distributions.append([np.mean(img[:,:,channelDict[channel]]) for img in c])

    

    fig = ff.create_distplot(distributions,

                            group_labels=group_labels,

                            colors=colors_cw[channel],

                            show_hist=False)

    

    fig.update_layout(title=f'{channel.capitalize()} channel distribution for all Classes')

    

    fig.show()
plotChannelWiseDistribution('red')
plotChannelWiseDistribution('green')
plotChannelWiseDistribution('blue')
def getRandomImage():

    return random.choice(classes[random.choice(train_df.columns[2:])])
img = getRandomImage()
fig = go.Figure(go.Image(z=img))



fig.update_layout(title_text="Smaple Image")



fig.show()
img = np.expand_dims(img,axis=0)
%%time



generator = tf.keras.preprocessing.image.ImageDataGenerator(vertical_flip=True,

                                                            horizontal_flip=True,

                                                            brightness_range=[0.5,1.5],

                                                            zoom_range=[0.5,1.1])



iterator = generator.flow(img,batch_size=1)



# fig = make_subplots(10,3,horizontal_spacing=0.01,vertical_spacing=0.01)



# for i in range(30):

#     image = iterator.next()[0].astype('uint8')

    

#     fig.add_trace(go.Image(z=image),i//3 + 1,i%3 + 1)



# fig.update_layout(title_text="Augmented Images of the sample image",

#                  height=128*10 + 20,

#                  width=256*3 + 20)



# fig.update_xaxes(showticklabels=False)

# fig.update_yaxes(showticklabels=False)



# fig.show()



fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(15,10))



ax[0,0].imshow(img[0])

ax[0,0].set_title("Sample Image",fontsize=10)

ax[0,0].set_xticks([])

ax[0,0].set_yticks([])





for i in range(1,15):

    

    image = iterator.next()[0].astype('uint8')

    

    ax[i//3,i%3].imshow(image)

    ax[i//3,i%3].set_xticks([])

    ax[i//3,i%3].set_yticks([])



fig.suptitle("Augmented Images of the sample image",fontsize=20)



plt.show()
sampleImg = getRandomImage()
def convertToHSV(img):

    return cv.cvtColor(img,cv.COLOR_RGB2HSV_FULL)
# mask = np.zeros(sampleImg.shape[:2],np.uint8)



# bgdModel = np.zeros((1,65),np.float64)

# fgdModel = np.zeros((1,65),np.float64)



# rect = (0,0,520,255)

# cv.grabCut(sampleImg,mask,rect,bgdModel,fgdModel,20,cv.GC_INIT_WITH_RECT)



# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

# segImg = sampleImg*mask2[:,:,np.newaxis]



# px.imshow(segImg)
def getROI(img):

    # convert the image to the gray-scale image

    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)

    

    # Detect the edges in the image using canny edge detection

    edged = cv.Canny(gray,150,200)

    

    xm = img.shape[1]//2    # Middle coordinate of the x-axis (width of the image)

    ym = img.shape[0]//2    # Middle coordinate of the y-axis (height of the image)

    

    # to find the bottom-y coordinate to the Rectangle-ROI

    for i in range(img.shape[0]-1,-1,-1):

        if np.sum(edged[i,xm-5:xm+5])!=0:

            y_bottom = np.where(i+10<img.shape[0]-1,i+10,img.shape[0]-2)

            break

            

    # to find the top-y coordinate to the Rectangle-ROI

    for i in range(img.shape[0]):

        if np.sum(edged[i,xm-5:xm+5])!=0:

            y_top = np.where(i-10>1,i-10,2)

            break

    

    # to find the top-x coordinate to the Rectangle-ROI

    for i in range(img.shape[1]):

        if np.sum(edged[ym-5:ym+5,i])!=0:

            x_top = np.where(i-10>1,i-10,2)

            break

            

    # to find the bottom-x coordinate to the Rectangle-ROI

    for i in range(img.shape[1]-1,-1,-1):

        if np.sum(edged[ym-5:ym+5,i])!=0:

            x_bottom = np.where(i+10<img.shape[1]-1,i+10,img.shape[1]-2)

            break



    return edged,(x_top,y_top,x_bottom,y_bottom)
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 13))



for i in range(3):

    orignal = getRandomImage()

    edged, coordinates = getROI(orignal)

    

    roi = orignal.copy()

    

    (x_top,y_top,x_bottom,y_bottom) = coordinates

    

    roi[y_top-2:y_top,x_top:x_bottom+1] = [255,0,0]        # Top-edge

    roi[y_bottom:y_bottom+2,x_top:x_bottom+1] = [255,0,0]  # Bottom-edge

    roi[y_top:y_bottom+1,x_top-2:x_top] = [255,0,0]        # Left-edge

    roi[y_top:y_bottom+1,x_bottom:x_bottom+2] = [255,0,0]  # Right-edge

    

    ax[i,0].imshow(orignal)

    ax[i,0].set_title('Original Image', fontsize=15)

    ax[i,1].imshow(edged, cmap='gray')

    ax[i,1].set_title('Detected Edges', fontsize=15)

    ax[i,2].imshow(roi)

    ax[i,2].set_title('ROI', fontsize=15)

    

fig.suptitle("ROI selection using Canny Edge Detection",fontsize=20)

    

plt.show()
%%time



fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(15,15))



for i in range(3):

    orignal = getRandomImage()

#     blur = cv.bilateralFilter(orignal,9,75,75)

    

    gray = cv.cvtColor(orignal,cv.COLOR_RGB2GRAY)

    sobel = filters.sobel(gray)

    

#     sobel = cv.morphologyEx(sobel, cv.MORPH_OPEN, kernel)

#     blurred = cv.bilateralFilter(sobel.astype('float32'),9,75,75)

    blurred = filters.gaussian(sobel, sigma=2.0)

    

    ym = blurred.shape[0]//2

    xm = blurred.shape[1]//2

    

    markers = np.zeros(blurred.shape,dtype=np.int)

    # using corners of the image as background

    markers[0,0:2*xm] = 1

    markers[2*ym-1,0:2*xm] = 1

    markers[0:2*ym,0] = 1

    markers[0:2*ym,2*xm-1] = 1

    

    # using middle part of the image as foreground

    markers[ym-50:ym+50,xm-20:xm+20] = 2

    

    mask = morphology.watershed(blurred, markers)

    

    ax[0,i].imshow(orignal)

    ax[0,i].set_title('Original Image', fontsize=12)

    

    ax[1,i].imshow(gray, cmap='gray')

    ax[1,i].set_title('Gray Image', fontsize=12)

    

    ax[2,i].imshow(sobel, cmap='gray')

    ax[2,i].set_title('After Sobel Filter', fontsize=12)

    

    ax[3,i].imshow(blurred, cmap='gray')

    ax[3,i].set_title('Blurred Image', fontsize=12)

    

    ax[4,i].imshow(mask, cmap='gray')

    ax[4,i].set_title('Mask', fontsize=12)

    

    orignal[mask==1,:] = [0,0,0]

    

    ax[5,i].imshow(orignal)

    ax[5,i].set_title('Segmented Image', fontsize=12)

    



for i in range(6):

    for j in range(3):

        ax[i,j].set_xticks([])

        ax[i,j].set_yticks([])

    

fig.suptitle("Image Segmentation (ROI selection) using Watershed Transformation",fontsize=20)

    

plt.show()
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))



for i in range(3):

    orignal = getRandomImage()

    hsv = convertToHSV(orignal)

    

    ax[i,0].imshow(orignal)

    ax[i,0].set_title('Original Image', fontsize=15)

    ax[i,1].imshow(hsv, cmap='gray')

    ax[i,1].set_title('HSV Image', fontsize=15)

    

fig.suptitle("RGB to HSV Conversion",fontsize=20)

    

plt.show()
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))



for i in range(3):

    orignal = getRandomImage()

    gray = cv.cvtColor(orignal,cv.COLOR_RGB2GRAY)

    

    ax[i,0].imshow(orignal)

    ax[i,0].set_title('Original Image', fontsize=15)

    ax[i,1].imshow(gray, cmap='gray')

    ax[i,1].set_title('Gray Image', fontsize=15)

    

fig.suptitle("RGB to Gray Scale Conversion",fontsize=20)



plt.show()
# TPU detection  

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print("TPU Detected")

    

except ValueError:

    print("TPU not Detected")

    tpu = None



# TPUStrategy for distributed training

if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)



else: # default strategy that works on CPU and single GPU

    strategy = tf.distribute.get_strategy()
image_count = train_df.shape[0]



if tpu:

    BATCH_SIZE = 16 * strategy.num_replicas_in_sync

else:

    BATCH_SIZE = 64



print("Setting Batch size to: ",BATCH_SIZE)

    

IMG_HEIGHT = 512

IMG_WIDTH = 512

STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)



GCS_PATH = KaggleDatasets().get_gcs_path()

print("GCS Path: ",GCS_PATH)
AUTO = tf.data.experimental.AUTOTUNE

ignore_order = tf.data.Options()

ignore_order.experimental_deterministic = False



# read train & test filenames

train_filenames = train_df['image_id'].apply(lambda x: GCS_PATH + '/images/' + x + '.jpg')

test_filenames = test_df['image_id'].apply(lambda x: GCS_PATH + '/images/' + x + '.jpg')



train_labels = train_df.iloc[:,1:].values.astype('float32')
def decodeImage(filename,label=None,image_size=(IMG_HEIGHT,IMG_WIDTH)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    

    if label is not None:

        return image, label

    

    return image

    

def imageAugmentation(image,label=None):

    

    aug = [True,False]

    

    if random.choice(aug):

        image = tf.image.random_brightness(image,max_delta=0.5,seed=SEED)

    if random.choice(aug):

        image = tf.image.random_flip_left_right(image,seed=SEED)

    if random.choice(aug):

        image = tf.image.random_flip_up_down(image,seed=SEED)

    

    if label is not None:

        return image,label

    

    return image





train_dataset = (

    tf.data.Dataset.from_tensor_slices((train_filenames,train_labels))

    .map(decodeImage, num_parallel_calls=AUTO)

    .shuffle(500)

    .cache()

    .repeat()

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

    .map(imageAugmentation, num_parallel_calls=AUTO))



test_dataset = (

    tf.data.Dataset.from_tensor_slices(test_filenames)

    .map(decodeImage, num_parallel_calls=AUTO)

    .cache()

    .batch(BATCH_SIZE)

    .prefetch(AUTO))
def scratchModel():

    model = tf.keras.models.Sequential()

    

    model.add(tf.keras.layers.Conv2D(64,kernel_size=5,padding='same',activation='relu',input_shape=[IMG_HEIGHT,IMG_WIDTH,3]))

#     model.add(tf.keras.layers.Conv2D(64,kernel_size=5,padding='same',activation='relu'))

#     model.add(tf.keras.layers.MaxPool2D())

    

    model.add(tf.keras.layers.Conv2D(64,kernel_size=5,strides=2,activation='relu'))

#     model.add(tf.keras.layers.Conv2D(64,kernel_size=5,padding='same'))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.ReLU())

#     model.add(tf.keras.layers.MaxPool2D())

    

    model.add(tf.keras.layers.Conv2D(128,kernel_size=5,strides=2,activation='relu'))

#     model.add(tf.keras.layers.Conv2D(128,kernel_size=5,padding='same'))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.ReLU())

#     model.add(tf.keras.layers.MaxPool2D())

    

    model.add(tf.keras.layers.Conv2D(256,kernel_size=5,strides=2,activation='relu'))

#     model.add(tf.keras.layers.Conv2D(256,kernel_size=5,padding='same'))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.ReLU())

#     model.add(tf.keras.layers.MaxPool2D())

    

#     model.add(tf.keras.layers.Conv2D(256,kernel_size=3,strides=2,activation='relu'))

# #     model.add(tf.keras.layers.Conv2D(512,kernel_size=3,padding='same'))

#     model.add(tf.keras.layers.BatchNormalization())

#     model.add(tf.keras.layers.ReLU())

# #     model.add(tf.keras.layers.MaxPool2D())

    

#     model.add(tf.keras.layers.Conv2D(256,kernel_size=3,strides=2,activation='relu'))

# #     model.add(tf.keras.layers.Conv2D(512,kernel_size=3,padding='same'))

#     model.add(tf.keras.layers.BatchNormalization())

#     model.add(tf.keras.layers.ReLU())

# #     model.add(tf.keras.layers.MaxPool2D())



#     model.add(tf.keras.layers.Conv2D(256,kernel_size=3,strides=2,activation='relu'))

# #     model.add(tf.keras.layers.Conv2D(512,kernel_size=3,padding='same'))

#     model.add(tf.keras.layers.BatchNormalization())

#     model.add(tf.keras.layers.ReLU())

# #     model.add(tf.keras.layers.MaxPool2D())

    

    model.add(tf.keras.layers.GlobalAveragePooling2D())

    model.add(tf.keras.layers.Dense(521,activation='relu'))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(521))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Dense(4,activation='softmax'))

    

    return model
def getDenseNets(index):

    

    if index==0:

        model = tf.keras.applications.DenseNet121(input_shape=[IMG_HEIGHT,IMG_WIDTH,3],

                                                 include_top=False,

                                                 weights='imagenet')

    elif index==1:

        model = tf.keras.applications.DenseNet169(input_shape=[IMG_HEIGHT,IMG_WIDTH,3],

                                                 include_top=False,

                                                 weights='imagenet')

    else:

        model = tf.keras.applications.DenseNet201(input_shape=[IMG_HEIGHT,IMG_WIDTH,3],

                                                 include_top=False,

                                                 weights='imagenet')

    

    return model
class CyclicLR(tf.keras.callbacks.Callback):

    

    def __init__(self,base_lr=1e-5,max_lr=1e-3,stepsize=10):

        super().__init__()

        

        self.base_lr = base_lr

        self.max_lr = max_lr

        self.stepsize = stepsize

        self.iterations = 0

        self.history = {}

        

    def clr(self):

        cycle = np.floor((1+self.iterations)/(2*self.stepsize))

        x = np.abs(self.iterations/self.stepsize - 2*cycle + 1)

        

        return self.base_lr + (self.max_lr - self.base_lr)*(np.maximum(0,1-x))

    

    def on_train_begin(self,logs={}):

        tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)

    

    def on_batch_end(self,batch,logs=None):

        logs = logs or {}

        

        self.iterations += 1

        

        self.history.setdefault('lr', []).append(tf.keras.backend.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.iterations)



        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        

        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())
class LRFinder(tf.keras.callbacks.Callback):

    

    def __init__(self,min_lr=1e-5,max_lr=1e-2,steps_per_epoch=None,epochs=None):

        super().__init__()

        

        self.min_lr = min_lr

        self.max_lr = max_lr

        self.total_iterations = steps_per_epoch*epochs

        self.iteration = 0

        self.history = {}

        

    def lr(self):

        

        x = self.iteration/self.total_iterations

        

        return self.min_lr + (self.max_lr - self.min_lr)*x

    

    def on_train_begin(self,logs={}):

        

        tf.keras.backend.set_value(self.model.optimizer.lr,self.min_lr)

        

    def on_batch_end(self,batch,logs=None):

        logs = logs or {}

        

        self.iteration += 1

        

        self.history.setdefault('lr', []).append(tf.keras.backend.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.iteration)



        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

            

        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr())

    

    def plot_lr(self):

        

        plt.plot(self.history['iterations'], self.history['lr'])

        plt.yscale('log')

        plt.xlabel('Iteration')

        plt.ylabel('Learning rate')

        plt.show()

        

    def plot_loss(self):

        

        plt.plot(self.history['lr'], self.history['loss'])

        plt.xscale('log')

        plt.xlabel('Learning rate')

        plt.ylabel('Loss')

        plt.show()
# tempModel = tf.keras.applications.Xception(input_shape=[512,512,3],

#                                                  include_top=False,

#                                                  weights='imagenet')



# tempModel.summary()



# # Let's take a look to see how many layers are in the base model

# print("Number of layers in the base model: ", len(tempModel.layers))
# (input_shape=[IMG_HEIGHT,IMG_WIDTH,3],

#  include_top=False,

#  weights='imagenet')



with strategy.scope():

            

    base_model = tf.keras.applications.Xception(input_shape=[IMG_HEIGHT,IMG_WIDTH,3],

                                                 include_top=False,

                                                 weights='imagenet')



#     base_model.trainable = False

    

    # Let's take a look to see how many layers are in the base model

    print("Number of layers in the base model: ", len(base_model.layers))



    # Fine-tune from this layer onwards

    fine_tune_at = np.floor(len(base_model.layers)*0.9)



    # Freeze all the layers before the `fine_tune_at` layer

#     for layer in base_model.layers[:int(fine_tune_at)]:

#         layer.trainable =  False



    

    model = tf.keras.models.Sequential([

        base_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(4,activation='softmax')

    ])



#     model = scratchModel()



    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),

             loss=tf.keras.losses.CategoricalCrossentropy(),

             metrics=['categorical_accuracy'])

    

    model.summary()

    

LRF_EPOCHS = 4    



lrfinder = LRFinder(steps_per_epoch=STEPS_PER_EPOCH,epochs=LRF_EPOCHS)

    

history = model.fit(train_dataset,epochs=LRF_EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,callbacks=[lrfinder])
lrfinder.plot_lr()
lrfinder.plot_loss()
FOLDS = 5



skf = StratifiedKFold(n_splits=FOLDS,shuffle=True,random_state=SEED)





def crossValidation(train_filenames, train_labels, fold=skf, callbacks=[], epochs=20,base_trainable=True,top_layers_trainable=False):

        

    val_scores = []

    history = []



    for i, (train_idx, test_idx) in enumerate(fold.split(train_filenames,[x.argmax() for x in train_labels])):



        X_train, y_train = train_filenames[train_idx], train_labels[train_idx]

        X_val, y_val = train_filenames[test_idx], train_labels[test_idx]

        

        

        # Load Dataset 

        train_dataset = (

        tf.data.Dataset.from_tensor_slices((X_train,y_train))

        .map(decodeImage, num_parallel_calls=AUTO)

        .shuffle(500)

        .cache()

        .repeat()

        .batch(BATCH_SIZE)

        .prefetch(AUTO)

        .map(imageAugmentation, num_parallel_calls=AUTO))



        validation_dataset = (

        tf.data.Dataset.from_tensor_slices((X_val,y_val))

        .map(decodeImage, num_parallel_calls=AUTO)

        .cache()

        .batch(BATCH_SIZE)

        .prefetch(AUTO))

        

        

        # load model 

        with strategy.scope():

            

            base_model = tf.keras.applications.Xception(input_shape=[IMG_HEIGHT,IMG_WIDTH,3],

                                                 include_top=False,

                                                 weights='imagenet')

            

            if not base_trainable:

                base_model.trainable = False

            

            if top_layers_trainable:

                fine_tune_at = np.floor(len(base_model.layers)*0.9)



                # Freeze all the layers before the `fine_tune_at` layer

                for layer in base_model.layers[:int(fine_tune_at)]:

                    layer.trainable =  False

            

            model = tf.keras.models.Sequential([

                base_model,

                tf.keras.layers.GlobalAveragePooling2D(),

                tf.keras.layers.Dense(4,activation='softmax')

            ])



#             model = scratchModel()

            

            model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),

                     loss=tf.keras.losses.CategoricalCrossentropy(),

                     metrics=['categorical_accuracy'])

            

#         print(model.summary())

        

        

        # Model training

        print(f"\n\nFold {i+1} Training: ")

        print("---------------------------------------------------------------------------\n\n")

        history0 = model.fit(train_dataset,epochs=epochs,steps_per_epoch=len(X_train)//BATCH_SIZE,validation_data=validation_dataset,callbacks=callbacks)

        print("\n\n---------------------------------------------------------------------------\n\n")



        history.append(history0)

        

        

        # Valdiation Average F1-Score Calculation

        val_predictions_prob = model.predict(validation_dataset)



        val_predictions = val_predictions_prob.copy()

        val_predictions[:,:] = 0

        

        for j, pred in enumerate(val_predictions_prob):

            val_predictions[j,pred.argmax()] = 1

        

        val_score = 0

        

        for j in range(4):

            val_score += f1_score(y_val[:,j],val_predictions[:,j])



        val_score /= 4



        print(f"\nFold {i+1} F1_Scores: \nValidation: {val_score}")

        print("------------------------------------------------------------------------\n\n")



        val_scores.append(val_score)

        

        # Debugging

#         break



    return history, val_scores
%%time



cyclicLR = CyclicLR(base_lr=1e-4,max_lr=1e-3)

# earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',baseline=0.9600,min_delta=0.001,patience=4)



VAL_EPOCHS = 20



cross_val_history, val_scores = crossValidation(train_filenames,

                                                train_labels,

                                                callbacks=[cyclicLR],

                                                epochs=VAL_EPOCHS,

                                                base_trainable=True,

                                                top_layers_trainable=False)
# Mean Validation F1-Scores



val_score = np.mean(val_scores)



print("Mean F1 Score: ",np.mean(val_scores))

print("Std F1 Score: ",np.std(val_scores))
fig = go.Figure()



train_loss = np.zeros(VAL_EPOCHS)

val_loss = np.zeros(VAL_EPOCHS)



for i, hist in enumerate(cross_val_history):

    

    train_loss = np.add(train_loss,hist.history['loss'])

    val_loss = np.add(val_loss,hist.history['val_loss'])



    

fig.add_trace(go.Scatter(x=np.arange(1,VAL_EPOCHS+1),

                        y=train_loss/FOLDS,

                        mode='lines+markers',

                        name='Train'))



fig.add_trace(go.Scatter(x=np.arange(1,VAL_EPOCHS+1),

                    y=val_loss/FOLDS,

                        mode='lines+markers',

                        name='Validation'))

    

fig.update_layout(title_text="Average Cross-Validation Losses")



fig.show()
fig = go.Figure()



train_accuracy = np.zeros(VAL_EPOCHS)

val_accuracy = np.zeros(VAL_EPOCHS)



for i, hist in enumerate(cross_val_history):

    

    train_accuracy = np.add(train_accuracy,hist.history['categorical_accuracy'])

    val_accuracy = np.add(val_accuracy,hist.history['val_categorical_accuracy'])

    



fig.add_trace(go.Scatter(x=np.arange(1,VAL_EPOCHS+1),

                        y=train_accuracy/FOLDS,

                        mode='lines+markers',

                        name='Train'))



fig.add_trace(go.Scatter(x=np.arange(1,VAL_EPOCHS+1),

                    y=val_accuracy/FOLDS,

                        mode='lines+markers',

                        name='Validation'))

    

fig.update_layout(title_text="Average Cross-Validation Accuracies")



fig.show()
%%time 

with strategy.scope():

    base_model = tf.keras.applications.Xception(input_shape=[IMG_HEIGHT,IMG_WIDTH,3],

                                                     include_top=False,

                                                     weights='imagenet')



    model = tf.keras.models.Sequential([

            base_model,

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(4,activation='softmax')

        ])



    #     model = scratchModel()



    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),

             loss=tf.keras.losses.CategoricalCrossentropy(),

             metrics=['categorical_accuracy'])   



EPOCHS = 20



history = model.fit(train_dataset,steps_per_epoch=STEPS_PER_EPOCH,epochs=EPOCHS,callbacks=[cyclicLR])
fig = make_subplots(1,2)



fig.add_trace(go.Scatter(x=np.arange(1,EPOCHS+1),

                        y=history.history['loss'],

                        mode='lines+markers',

                        name=f'Training Loss'),1,1)



fig.add_trace(go.Scatter(x=np.arange(1,EPOCHS+1),

                        y=history.history['categorical_accuracy'],

                        mode='lines+markers',

                        name=f'Training Accuracy'),1,2)



fig.update_layout(title_text="Training Loss & Accuracy")



fig.show()
test_predictions = model.predict(test_dataset)
fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(15, 18))



for i in range(5):

    img_id = random.choice(np.arange(0,test_df.shape[0]))

    test_image = getImage(test_df.image_id[img_id],(IMG_HEIGHT,IMG_WIDTH))

    

    ax[i,0].imshow(test_image)

    ax[i,0].set_title(f'{test_df.image_id[img_id]}', fontsize=12)

    ax[i,1].barh(y=train_df.columns[1:],width=test_predictions[img_id])

    ax[i,1].set_title('Predictions', fontsize=12)

    

fig.suptitle("Test set Predictions",fontsize=20)

    

plt.show()
sampleSubmission_df.iloc[:,1:] = test_predictions



model_name = 'Xception'



sampleSubmission_df.to_csv(f'/kaggle/working/{model_name}{val_score}.csv',index=False)