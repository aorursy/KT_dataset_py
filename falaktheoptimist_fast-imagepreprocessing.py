import os, glob, math, cv2, time

import numpy as np
from joblib import Parallel, delayed
img_size = 50
sz = (img_size, img_size)

nprocs = 2
def process_image(img_file):
    img = cv2.imread(img_file)
    img = cv2.resize(img, sz).transpose((2,0,1)).astype('float32') / 255.0
    
    return img
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
a = cv2.imread('../input/train/c9/img_48026.jpg')
print(np.size(X_train,0))
print(np.size(cv2.resize(a, sz).transpose((2,0,1)),1))
print(np.size(cv2.resize(a, sz).transpose((2,0,1)),2))
def process_test_image(img_file):
    return process_image(img_file), os.path.basename(img_file)
start = time.time()

X_test    = []
X_test_id = []


path  = os.path.join('../input/test', '*.jpg')
files = glob.glob(path)

results = Parallel(n_jobs=nprocs)(delayed(process_test_image)(im_file) for im_file in files)
X_test, X_test_id = zip(*results)

end = time.time() - start
print("Time: %.2f seconds" % end)

print(len(X_test))