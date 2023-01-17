import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import glob

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
%matplotlib inline
plt.rcParams['figure.figsize'] = (19.0, 17.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2
data_dir = r'/kaggle/input/dataset/dataset/'
classes = ['broadleaf', 'grass', 'soil', 'soybean'] 

num_file = 1100 
all_files = [] 
num_data =num_file*len(classes)
Y = np.zeros(num_data)


for i, cls in enumerate(classes):
    all_files += [f for f in glob.glob(data_dir+cls+'/*.tif')][:num_file]
    Y[i*num_file:(i+1)*num_file] = i # label all classes with int [0.. len(classes)]

    
# Image dimension
im_width = 230
im_height = 230 
im_channel = 3
dim = im_width * im_height * im_channel

X = np.ndarray(shape=(num_data, im_width, im_height, im_channel), dtype=np.uint8)

for idx, file in enumerate(all_files):
    X[idx] = cv2.resize(cv2.imread(file), (im_width, im_height))

X_train = np.empty(shape=(4000,im_width, im_height, im_channel), dtype=np.uint8)
X_val = np.empty(shape=(200,im_width, im_height, im_channel), dtype=np.uint8)
X_test = np.empty(shape=(200,im_width, im_height, im_channel), dtype=np.uint8)

y_train = np.empty(4000)
y_val = np.empty(200)
y_test = np.empty(200) 

for i, cls in enumerate(classes): 
    X_test[50*i:50*(i+1)] = X[np.where(Y == i)[0][:50]]
    X_val[50*i:50*(i+1)] = X[np.where(Y == i)[0][50:100]]
    X_train[1000*i:1000*(i+1)] = X[np.where(Y == i)[0][100:]]
    
    y_test[50*i:50*(i+1)] = i
    y_val[50*i:50*(i+1)] = i
    y_train[1000*i:1000*(i+1)] = i
    
del Y 
del X

def color_space_transform(imgs, space=['hsv','cive','exg','exr']):
    '''
    imgs: N inputs image shape (N, D0, D1, D2,..)
    space: list of transformation that will be compute and return. 
    return result = {} 
    '''
    available_space = ['hsv','cive','exg','exr']
    for s in space: 
        if(s not in available_space): 
            print(s+'is not available')
            quit()
    N = imgs.shape[0] if len(imgs.shape) == 4 else 1
    if(N == 1): imgs = np.reshape(imgs, (1,imgs.shape[0],imgs.shape[1], imgs.shape[2]))
        
    imgs = imgs.astype('float32')
    result = {} 
    #create  memory
    for s in space:
        if(s == "hsv"): result[s] = np.zeros(imgs.shape,dtype='float32')
        else: result[s] = np.zeros((N, imgs.shape[1], imgs.shape[2]),dtype='float32')
    
    for i in range(N):
        for s in space:
            if(s == 'hsv'):
                result[s][i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2HSV)
            elif(s == 'cive'):
                p_blue, p_green, p_red= cv2.split(imgs[i]) # For BGR image # For RGB image
                result[s][i] = 0.881*p_green + 0.441*p_red + 0.385*p_blue - 18.78745
            elif(s == 'exg'):
                p_blue, p_green, p_red= cv2.split(imgs[i]) # For BGR image # For RGB image
                result[s][i] = 2 * p_green - p_red - p_blue
            elif(s == 'exr'):
                 p_blue, p_green, p_red= cv2.split(imgs[i]) # For BGR image # For RGB image
                 result[s][i] = 1.4 * p_red - p_green
    return result
# Convert to Hsv and cive for example 
X_transform = color_space_transform(X_train[0:4000], space=['hsv','cive','exg','exr'])

#And plot to visualize some 
# Visualize some images 
# Make sure that everything when OK
classes = ['broadleaf', 'grass', 'soil', 'soybean']
n_class = len(classes)
samples_per_class = 4

print("HSV color space")
for y, cls in enumerate(classes):
    idxes = np.flatnonzero(y == y_train[0:4000])
    idxes = np.random.choice(idxes, samples_per_class, replace = False)
    for i, idx in enumerate(idxes):
        plt_idx = i * n_class + y + 1
        plt.subplot(samples_per_class,n_class, plt_idx)
        plt.imshow(X_transform['hsv'][idx])
        if(i==0): plt.title(cls)

plt.show()
print("cive color space")
for y, cls in enumerate(classes):
    idxes = np.flatnonzero(y == y_train[0:4000])
    idxes = np.random.choice(idxes, samples_per_class, replace = False)
    for i, idx in enumerate(idxes):
        plt_idx = i * n_class + y + 1
        plt.subplot(samples_per_class,n_class, plt_idx)
        plt.imshow(X_transform['cive'][idx])
        if(i==0): plt.title(cls)

plt.show()
#Shuffle training index
train_idxs = np.random.permutation(X_train.shape[0])
y_train  = y_train[train_idxs]
X_train = X_train[train_idxs]

X_train = np.reshape(X_train, (X_train.shape[0], -1)).astype('float64')
X_test = np.reshape(X_test, (X_test.shape[0], -1)).astype('float64')
X_val = np.reshape(X_val, (X_val.shape[0], -1)).astype('float64')
X_dev = X_train[0:100].astype('float64')
X_tiny = X_train[100:110].astype('float64')
y_dev = y_train[0:100] 
y_tiny = y_train[100:110] 

print("X_train shape", X_train.shape, "| y_train shape:", y_train.shape)
print("X_test shape", X_test.shape, "| y_test shape:", y_test.shape)
print("X_val shape", X_val.shape, "| y_val shape:", y_val.shape)
print("X_dev shape", X_dev.shape, "| y_dev shape:", y_dev.shape)
print("X_tiny shape", X_tiny.shape, "| y_tiny shape:", y_tiny.shape)
# Visualize some images 
# Make sure that everything when OK
classes = ['broadleaf', 'grass', 'soil', 'soybean']
n_class = len(classes)
samples_per_class = 4


for y, cls in enumerate(classes):
    idxes = np.flatnonzero(y == y_train)
    idxes = np.random.choice(idxes, samples_per_class, replace = False)
    for i, idx in enumerate(idxes):
        plt_idx = i * n_class + y + 1
        plt.subplot(samples_per_class,n_class, plt_idx)
        plt.imshow(X_train[idx].reshape(im_width, im_height, im_channel).astype('uint8'))
        if(i==0): plt.title(cls)

plt.show()
# Visualize some images 
# Make sure that everything when OK
classes = ['broadleaf', 'grass', 'soil', 'soybean']
n_class = len(classes)
samples_per_class = 4


for y, cls in enumerate(classes):
    idxes = np.flatnonzero(y == y_val)
    idxes = np.random.choice(idxes, samples_per_class, replace = False)
    for i, idx in enumerate(idxes):
        plt_idx = i * n_class + y + 1
        plt.subplot(samples_per_class,n_class, plt_idx)
        plt.imshow(X_val[idx].reshape(im_width, im_height, im_channel).astype('uint8'))
        if(i==0): plt.title(cls)

plt.show()
#first: compute the mean image
mean_image = np.mean(X_train, axis=0) #axis=0. stack horizontally
plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((im_width, im_height, im_channel)).astype('uint8'))
plt.show()
#Second subtract the mean image from train and test data 
X_train -= mean_image
X_val -= mean_image 
X_test -= mean_image
X_dev -= mean_image
X_tiny -= mean_image
#Third append the bias dimension using linear algebra trick
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
X_tiny = np.hstack([X_tiny, np.ones((X_tiny.shape[0], 1))])

print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)
print("X_val shape", X_val.shape)
print("X_dev shape", X_dev.shape)
print("X_tiny shape", X_tiny.shape)