import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
import multiprocessing as mp
#from dataProcessing import loadImage
import matplotlib.pyplot as plt
import time
BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
SEED = 777
SHAPE = (512, 512, 4)
CORES = mp.cpu_count() #4
DIR = '../input'
OUTPUT_DIR = '.'
DEBUG = True
def getTrainDataset():
    
    path_to_train = DIR + '/train/'
    data = pd.read_csv(DIR + '/train.csv')

    paths = []
    labels = []
    
    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        paths.append(os.path.join(path_to_train, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


def getTestDataset():
    
    path_to_test = DIR + '/test/'
    data = pd.read_csv(DIR + '/sample_submission.csv')

    paths = []
    labels = []
    
    for name in data['Id']:
        y = np.ones(28)
        paths.append(os.path.join(path_to_test, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


def prepareData(paths, labels, shuffle = True, shape = SHAPE, seed = SEED, batch_size = BATCH_SIZE, debug = False):
    
    keys = np.arange(paths.shape[0], dtype=np.int)
    if(shuffle):
        np.random.seed(seed)
        np.random.shuffle(keys)

    if(paths.shape[0] % batch_size != 0):
        remaining = (paths.shape[0] // batch_size + 1) * batch_size - paths.shape[0]
        keys = np.append(keys, np.zeros(remaining, dtype=np.int32))
        
    keys = keys.reshape(-1,batch_size)
    
    if debug == True:
        keys = keys[0:8]
    
    paths = paths[keys]
    labels = labels[keys]
    
    processImages(paths, labels, shape)
    
    return paths, labels


def loadImage(path):

        R = Image.open(path + '_red.png')
        G = Image.open(path + '_green.png')
        B = Image.open(path + '_blue.png')
        Y = Image.open(path + '_yellow.png')

        im = np.stack((
            np.array(R), 
            np.array(G), 
            np.array(B),
            np.array(Y)), -1)
        
        im = np.divide(im, 255)
        return im

    
def processImages(paths, labels, shape):
    
    p = Pool(CORES)
    
    for batch in tqdm(range(paths.shape[0])):

        batch_size = paths[batch].shape[0]
        n_labels = labels[batch].shape[1]
        
        batch_labels = np.zeros((batch_size, n_labels))
        
        batch_images = np.array(p.map(loadImage, paths[batch]))
        batch_labels = labels[batch]
        np.savez(os.path.dirname(paths[batch][0]).replace(DIR, OUTPUT_DIR) + '-memory'+str(batch), images=batch_images, labels=batch_labels)
paths, labels = getTrainDataset()
pathsTrain, labelsTrain = prepareData(paths, labels, debug = DEBUG)
paths, labels = getTestDataset()
pathsTest, labelsTest = prepareData(paths, labels, shuffle=False, batch_size = TEST_BATCH_SIZE, debug = DEBUG)
