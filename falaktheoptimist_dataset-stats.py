import sys
sys.path.append('/usr/local/lib/python2.7/site-packages') # For cv2 finding
import os, glob, math, cv2, time
import numpy as np
from joblib import Parallel, delayed
train_folder = '../input/train'
test_folder = '../input/test'
'''
Get information about the images in the dataset
1. How many images in the training set
2. How many belonging to each category in the train set
3. How many images in the test set
4. Stats about sizes of images in train and test set: This was 

'''
def image_stats_train(train_folder, num_classes):
    numfiles = []
    filesizevector = [480,640]
    for j in range(num_classes):
        path = os.path.join(train_folder, 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        numfiles.append(np.size(files))
        for eachfile in files:
            a = cv2.imread(eachfile)
            filesize = [np.size(a,0),np.size(a,1)]
            filesizevector = np.vstack((filesizevector,filesize))
    return (numfiles,filesizevector)
retval   = image_stats_train(train_folder,10)
fsvector = retval[1]
numimg   = retval[0]
a = [('class ' + str(i)+ ' : '+ str(numimg[i])) for i in range(10)] 
print(a)
print(np.mean(fsvector,0))
print(np.std(fsvector,0))
def image_stats_test(test_folder):
    numfiles = []
    filesizevector = [480,640]
    path = os.path.join(test_folder, '*.jpg')
    files = glob.glob(path)
    numfiles.append(np.size(files))
    for eachfile in files:
        a = cv2.imread(eachfile)
        filesize = [np.size(a,0),np.size(a,1)]
        filesizevector = np.vstack((filesizevector,filesize))
    return (numfiles,filesizevector)
retval   = image_stats_test(test_folder)
fsvector = retval[1]
print('Number of test images: ' +str(retval[0]))
