import os, glob, math, cv2, time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as plb
from joblib import Parallel, delayed

img_size = 50
sz = (img_size, img_size)
nprocs = 2

def process_image(img_file):
    img = cv2.imread(img_file)
    img = cv2.resize(img, sz).transpose((2,0,1)).astype('float32') / 255.0
    return img

#Training
start = time.time()

X_train = []
Y_train = []

for j in range(10):
    print('Load folder c{}'.format(j))
    path = os.path.join('../input/train', 'c' + str(j), '*.jpg')
    files = glob.glob(path)
    X_train.extend(Parallel(n_jobs=nprocs)(delayed(process_image)(im_file) for im_file in files))
    Y_train.extend([j]*len(files))
    
end = time.time() - start
print("Time: %.2f seconds" % end)
Y_train[1]
