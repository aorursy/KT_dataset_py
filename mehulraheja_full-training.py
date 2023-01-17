import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image
import keras
import tensorflow as tf
import time
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
!mkdir Data
os.system('gsutil ls gs://quickdraw_dataset/sketchrnn/ >> names.txt')
paths = []
with open('names.txt','r') as f:
    for line in f.readlines():
        if "full" in line:
            continue
        paths.append(line.replace("\n",""))
paths = np.array(paths)

#TESTING

num_classes = 7

paths = paths[:num_classes]

names = [path.split('/')[-1].replace('.npz','') for path in paths]
#Utils 

download = lambda file, dataPath: os.system('gsutil -m cp "' + file + '" ' + dataPath)

def getImg(fig):
    canvas = FigureCanvas(fig)
    canvas.draw() 

    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)

    return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
def getData(path, numSamples = 20, generator = 'train',verbose = True):
    
    dataPath = 'Data'
    fileName = path.split('/')[-1]
    
    localPath = os.path.join(dataPath,fileName)
    
    if not os.path.exists(localPath):
        if verbose:
            print('DOWNLOADING',path)
        download(path,dataPath)
    
    #Caches data
    data = np.load(localPath, encoding='latin1', allow_pickle=True)
        
    test,train,valid = data['test'],data['train'],data['valid']
    
    cur = train if generator=='train' else valid
    
    assert numSamples < len(cur)
    
    idx = np.concatenate((np.ones(numSamples),np.zeros(len(cur) - numSamples)))
    np.random.shuffle(idx)
    
    imgs = cur[idx == 1]
    
    ret = []
    
    
    for i,img in enumerate(imgs):

        fig, ax = plt.subplots()
        curx, cury = 0, 0
        
        for stroke in img:
            if stroke[2] == 0:
                x,y  = [curx, curx + stroke[0]], [cury, cury - stroke[1]]
                ax.plot(x,y)
            curx += stroke[0]
            cury -= stroke[1]
        
        
        ax.axis('off')
        fig.patch.set_facecolor('black')
        fig.savefig('temp.jpg',facecolor=fig.get_facecolor())
        temp = image.imread("temp.jpg")
        #temp = getImg(fig)
        ret.append(temp)
        fig.clear()
        plt.close(fig)
        
    return np.array(ret)
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    
    def __init__(self, generator='train'):
        self.epochNumber = 0
        self.generator = generator

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.generator=='valid':
            return 5
        elif self.generator=='train':
            return 50

    def __getitem__(self, index):
        'Generate one batch of data'
        num = 5
        idx = np.concatenate((np.ones(num),np.zeros(len(paths) - num)))
        np.random.shuffle(idx)
        
        
        
        pos = np.where(idx == 1)[0]
        curPaths = paths[idx == 1]
        
        
        X = []
        y = []
        
        for i, path in enumerate(curPaths):
            fileName = path.split('/')[-1]
            
            NUM_SAMPLES = 5
            
            X.append(np.array(getData(path, numSamples = NUM_SAMPLES, generator = self.generator)))
            for j in range(NUM_SAMPLES):
                Y = np.zeros(num_classes)
                Y[pos[i]] = 1
                y.append(np.array([Y]))
            
       
        
        X = np.vstack(X)
        y = np.vstack(y)
        
        
        return X, y

    def on_epoch_end(self):
        
        self.epochNumber += 1
        
        if not self.epochNumber%2:
            name = str(self.epochNumber) + ".h5"
            model.save(name)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D



model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
train = DataGenerator()
valid = DataGenerator(generator='valid')
model.fit_generator(generator=train, validation_data = valid, epochs = 20)
model = keras.models.load_model('4.h5')
def saveGif(save_path, data_class):
    dataPath = 'Data'
    fileName = paths[data_class].split('/')[-1]

    localPath = os.path.join(dataPath,fileName)

    if not os.path.exists(localPath):
        if verbose:
            print('DOWNLOADING',path)
        download(path,dataPath)

    data = np.load(localPath, encoding='latin1', allow_pickle=True)
    test,train,valid = data['test'],data['train'],data['valid']

    cur = train

    img = np.random.choice(cur,1)[0]

    #
    fig, ax = plt.subplots()

    ax.axis('off')
    fig.patch.set_facecolor('black')
    progress = []

    curx, cury = 0, 0 
    for stroke in img:
        if stroke[2] == 0:
            x,y  = [curx, curx + stroke[0]], [cury, cury - stroke[1]]
            ax.plot(x,y)
        curx += stroke[0]
        cury -= stroke[1]

        fig.savefig('temp.jpg',facecolor=fig.get_facecolor())
        temp = image.imread("temp.jpg")
        progress.append(temp)

    plt.close()

    #Normal Drawings
    fig, ax = plt.subplots()

    progress2 = []

    ax.axis('off')
    ax.set_xlim(-800,800)
    ax.set_ylim(-800,800)
    progress2 = []

    curx, cury = 0, 0 
    for stroke in img:
        if stroke[2] == 0:
            x,y  = [curx, curx + stroke[0]], [cury, cury - stroke[1]]
            ax.plot(x,y,color='black')
        curx += stroke[0]
        cury -= stroke[1]

        fig.savefig('temp.jpg',facecolor=fig.get_facecolor())
        temp = image.imread("temp.jpg")
        progress2.append(temp)

    plt.close()

    GIF = []
    for i in range(len(progress)):
        fig, ax = plt.subplots(1,2,figsize = (15, 10))
        fig.suptitle('ACTUAL:' + names[data_class] +' CURRENT PREDICTION: ' + names[np.argmax(model.predict(np.array([progress[i]])))])

        ax[1].set_title('Representation')
        ax[1].axis('off')
        ax[1].imshow(progress[i])

        ax[0].set_title('Drawing')
        ax[0].axis('off')
        ax[0].imshow(progress2[i])

        fig.savefig('temp.jpg',facecolor=fig.get_facecolor())
        temp = image.imread("temp.jpg")
        GIF.append(temp)

        plt.close()

    imageio.mimsave(save_path, GIF)
saveGif('out.gif',4)
for i, name in enumerate(names):
    saveGif(name + '.gif',i)
data = np.load(localPath, encoding='latin1', allow_pickle=True)