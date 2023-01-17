#Here are some standard libraries that are loaded when you 
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualize satellite images
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from skimage.io import imshow # visualize satellite images

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout # components of network
from keras.models import Sequential # type of model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from skimage.filters import gaussian
from scipy.interpolate import interp1d
import skimage
import pylab as pl
import scipy.signal
from scipy import misc
x_train_set_fpath = '../input/X_test_sat4.csv'
y_train_set_fpath = '../input/y_test_sat4.csv'
print ('Loading Training Data')
X_train = pd.read_csv(x_train_set_fpath)
print ('Loaded 28 x 28 x 4 images')

Y_train = pd.read_csv(y_train_set_fpath)
print ('Loaded labels')
target_class_dict = {
    0: 'Barren Land',
    1: 'Trees',
    2: 'Grasslands',
    3: 'Urban'
}
plt.figure(figsize=(10, 10))
p = pd.Series(Y_train.argmax(axis=1)).value_counts().plot(kind='pie',
                    labels=['Barren Land', 'Trees', 'Grasslands', 'Urban'],
                    autopct='%1.1f%%')
p.add_artist(plt.Circle((0,0), 0.7, color='white'))
plt.title('Class Distribution of Satellite Images')
plt.legend()
plt.savefig('Class distribution.PNG')
X_train = X_train.as_matrix()
Y_train = Y_train.as_matrix()
print ('We have',X_train.shape[0],'examples and each example is a list of',X_train.shape[1],'numbers with',Y_train.shape[1],'possible classifications.')
#First we have to reshape each of them from a list of numbers to a 28*28*4 image.
X_train_img = X_train.reshape([99999,28,28,4]).astype(float)
print (X_train_img.shape)
class_idxs = []
y_targets = Y_train.argmax(axis=1)
for c in range(4):
    class_idxs.append(np.where(y_targets==c))
# dimensions
w=28
h=28
c=4
fig=plt.figure(figsize=(12, 12))
columns = 5
rows = 4
i = 0
for classs, c_idxs in enumerate(class_idxs):
    n = c_idxs[0].shape[0]
    s = np.random.randint(0, n-5)
    
    for idx in c_idxs[0][s:s+5]:
        img = np.squeeze(X_train_img[idx,:,:,0:3]).astype(float)
        fig.add_subplot(rows, columns, i+1)
        i+=1
        plt.title(str(idx)+':'+target_class_dict[classs])
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        
plt.savefig('target class examples.PNG')
plt.show()
#Let's take a look at one image. Keep in mind the channels are R,G,B, and I(Infrared)
ix = 5#Type a number between 0 and 99,999 inclusive
imshow(np.squeeze(X_train_img[ix,:,:,0:3]).astype(float)) #Only seeing the RGB channels
plt.show()
#Tells what the image is
if Y_train[ix,0] == 1:
    print ('Barren Land')
elif Y_train[ix,1] == 1:
    print ('Trees')
elif Y_train[ix,2] == 1:
    print ('Grassland')
else:
    print ('Other')
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def smooth_image(img):
    return cv2.GaussianBlur(img, (3,3), 0)

def im2col(A,BLKSZ):   

    # Parameters
    M,N = A.shape
    col_extent = N - BLKSZ[1] + 1
    row_extent = M - BLKSZ[0] + 1

    # Get Starting block indices
    start_idx = np.arange(BLKSZ[0])[:,None]*N + np.arange(BLKSZ[1])

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    return np.take (A,start_idx.ravel()[:,None] + offset_idx.ravel())

def coltfilt(A, size):
    
    original_shape = np.shape(A)
    a,b = 0, 0
    if(size%2==0):
        a, b = int(size/2)-1, int(size/2)
    else:
        a,b = int(size/2), int(size/2)
    A = np.lib.pad(A, (a, b), 'constant')
    Acol = im2col(A, (size, size))
    rc = np.floor((Acol.max(axis=0) - Acol.min(axis=0))/float(size)) + 1
    return np.reshape(rc, original_shape)

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def mat2gray(mat):
    maxI = np.max(mat)
    minI = np.min(mat)
    gray = (mat[:,:] - minI) / (maxI - minI)
    return gray
    
#------- computing the slope using linear regression -------
def fractal_aug(image):
    
    image = smooth_image(image)
    
    n_channels = len(np.shape(image))

    if(n_channels == 3):
        image=rgb2gray(image)
    
    image = smooth_image(image)

    image *= 255.0
    imrows, imcols = np.shape(image)
    
    B = np.zeros((6, imrows, imcols))

    #print("Calculating Differential Box Counting image")

    for r in range(2,8):
        mask = matlab_style_gauss2D((r,r), r/2.0)
        im = scipy.signal.convolve2d(image, mask, mode='same')
        F = (coltfilt(im, r))*(49/(r**2))
        B[r - 2] = np.log(F)

    #print("Calculating FD image")

    i = np.log(range(2,8)) #Normalised scale range vector

    Nxx = np.dot(i,i) - (np.sum(i)**2)/6
    FD = np.zeros((imrows,imcols))

    for m in range(1,imrows):
        for n in range(1,imcols):
            fd = [B[5,m,n], B[4,m,n], B[3,m,n], B[2,m,n], B[1,m,n], B[0,m,n]] #Number of boxes multiscale vector
            Nxy = np.dot(i,fd) - (sum(i)*sum(fd))/6
            FD[m,n] = Nxy/Nxx # slope of the linear regression line

    tmp = np.zeros(np.shape(B))
    for r in range(2,8):
        tmp[r-2, :, :] = FD * np.log(m)

    intercept = np.mean(B - tmp, axis=0)

    FDB = mat2gray(FD);

    intercept_image = mat2gray(intercept)
    
    #plt.imshow(intercept_image, cmap='gray')
    #plt.show()
    intercept_image = ((intercept_image - intercept_image.min()) * (1/(intercept_image.max() - intercept_image.min())) * 255).astype('uint8')
    
    return intercept_image
ix = 23
test_img = np.squeeze(X_train_img[ix,:,:,0:3]).astype(float)
fractal_img = fractal_aug(test_img)
print(fractal_img.shape)

plt.imshow(fractal_img, cmap='gray')
plt.savefig('fractal-test.PNG')
plt.imshow(test_img)
plt.savefig('original-test.PNG')
fig=plt.figure(figsize=(12, 12))
columns = 5
rows = 4
i = 0
for classs, c_idxs in enumerate(class_idxs):
    n = c_idxs[0].shape[0]
    s = np.random.randint(0, n-5)
    
    for idx in c_idxs[0][s:s+5]:
        img = fractal_aug(np.squeeze(X_train_img[idx,:,:,0:3]).astype(float))
        fig.add_subplot(rows, columns, i+1)
        i+=1
        plt.title(str(idx)+':'+target_class_dict[classs])
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap='gray')
        
plt.savefig('target class fractal.PNG')
plt.show()
def fractal_dimension(image, threshold=0.9):
    # finding all the non-zero pixels
    pixels=[]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j]>0:
                pixels.append((i,j))

    Lx=image.shape[1]
    Ly=image.shape[0]
    #print (Lx, Ly)
    pixels=pl.array(pixels)
    #print (pixels.shape)

    # computing the fractal dimension
    #considering only scales in a logarithmic list
    scales=np.logspace(0.001, 1, num=10, endpoint=False, base=2)
    Ns=[]
    # looping over several scales
    for scale in scales:
        # computing the histogram
        H, edges=np.histogramdd(pixels, bins=(np.arange(0,Lx,scale),np.arange(0,Ly,scale)))
        Ns.append(np.sum(H>0))

    # linear fit, polynomial of degree 1
    coeffs=np.polyfit(np.log(scales), np.log(Ns), 1)
    return -coeffs[0]
fig=plt.figure(figsize=(12, 12))
columns = 2
rows = 2
i = 0
for classs, c_idxs in enumerate(class_idxs):
    n = c_idxs[0].shape[0]
    s = np.random.randint(0, n-5)
    
    join_indices = c_idxs[0][s:s+16]
    img = (np.vstack((
        np.hstack((np.vstack((np.hstack((np.squeeze(X_train_img[join_indices[0],:,:,0:3]).astype(float),
                                         np.squeeze(X_train_img[join_indices[1],:,:,0:3]).astype(float))),
                              np.hstack((np.squeeze(X_train_img[join_indices[2],:,:,0:3]).astype(float),
                                         np.squeeze(X_train_img[join_indices[3],:,:,0:3]).astype(float))))),
                   np.vstack((np.hstack((np.squeeze(X_train_img[join_indices[4],:,:,0:3]).astype(float),
                                         np.squeeze(X_train_img[join_indices[5],:,:,0:3]).astype(float))),
                              np.hstack((np.squeeze(X_train_img[join_indices[6],:,:,0:3]).astype(float),
                                         np.squeeze(X_train_img[join_indices[7],:,:,0:3]).astype(float))))))),
        np.hstack((np.vstack((np.hstack((np.squeeze(X_train_img[join_indices[8],:,:,0:3]).astype(float),
                                         np.squeeze(X_train_img[join_indices[9],:,:,0:3]).astype(float))),
                              np.hstack((np.squeeze(X_train_img[join_indices[10],:,:,0:3]).astype(float),
                                         np.squeeze(X_train_img[join_indices[11],:,:,0:3]).astype(float))))),
                   np.vstack((np.hstack((np.squeeze(X_train_img[join_indices[12],:,:,0:3]).astype(float),
                                         np.squeeze(X_train_img[join_indices[13],:,:,0:3]).astype(float))),
                              np.hstack((np.squeeze(X_train_img[join_indices[14],:,:,0:3]).astype(float),
                                         np.squeeze(X_train_img[join_indices[15],:,:,0:3]).astype(float))))))))) )
    
    img = rgb2gray(img)
    
    # perform adaptive thresholding
    t = skimage.filters.threshold_otsu(img)
    mask = img > t
    
    fig.add_subplot(rows, columns, i+1)
    i+=1
    plt.title('FD of '+target_class_dict[classs]+':'+str(fractal_dimension(mask, 0.25)))
    #plt.xticks([])
    #plt.yticks([])
    plt.imshow(img, cmap='gray')
        
plt.savefig('FD of classes 8.PNG')
#plt.show()

# FD simulation of all the classes
i = 0
large_fd_table = {
    0: [],
    1: [],
    2: [],
    3: []
}
for classs, c_idxs in enumerate(class_idxs):
    print('...'*15, classs, '...'*15)
    for _ in range(300):
        n = c_idxs[0].shape[0]
        s = np.random.randint(0, n-16)


        join_indices = c_idxs[0][s:s+16]
        img = (np.vstack((
            np.hstack((np.vstack((np.hstack((np.squeeze(X_train_img[join_indices[0],:,:,0:3]).astype(float),
                                             np.squeeze(X_train_img[join_indices[1],:,:,0:3]).astype(float))),
                                  np.hstack((np.squeeze(X_train_img[join_indices[2],:,:,0:3]).astype(float),
                                             np.squeeze(X_train_img[join_indices[3],:,:,0:3]).astype(float))))),
                       np.vstack((np.hstack((np.squeeze(X_train_img[join_indices[4],:,:,0:3]).astype(float),
                                             np.squeeze(X_train_img[join_indices[5],:,:,0:3]).astype(float))),
                                  np.hstack((np.squeeze(X_train_img[join_indices[6],:,:,0:3]).astype(float),
                                             np.squeeze(X_train_img[join_indices[7],:,:,0:3]).astype(float))))))),
            np.hstack((np.vstack((np.hstack((np.squeeze(X_train_img[join_indices[8],:,:,0:3]).astype(float),
                                             np.squeeze(X_train_img[join_indices[9],:,:,0:3]).astype(float))),
                                  np.hstack((np.squeeze(X_train_img[join_indices[10],:,:,0:3]).astype(float),
                                             np.squeeze(X_train_img[join_indices[11],:,:,0:3]).astype(float))))),
                       np.vstack((np.hstack((np.squeeze(X_train_img[join_indices[12],:,:,0:3]).astype(float),
                                             np.squeeze(X_train_img[join_indices[13],:,:,0:3]).astype(float))),
                                  np.hstack((np.squeeze(X_train_img[join_indices[14],:,:,0:3]).astype(float),
                                             np.squeeze(X_train_img[join_indices[15],:,:,0:3]).astype(float))))))))) )

        img = rgb2gray(img)

        # perform adaptive thresholding
        t = skimage.filters.threshold_otsu(img)
        mask = img > t
        
        # add FD to dict
        large_fd_table[classs].append(fractal_dimension(mask, 0.25))
        
for k in large_fd_table.keys():
    large_fd_table[k] = np.array(large_fd_table[k])
large_fd_table_df = pd.DataFrame(large_fd_table)
large_fd_table_df.plot(figsize=(30, 8), grid=True)
plt.xlabel('Sample Set')
plt.ylabel('Fractal Dimension')
plt.title('FD of 300 Sample set of different classes')
plt.legend(['Barren Lands', 'Trees', 'Grasslands', 'Urban'])
plt.savefig('FD of 300 Sample set of different classes.PNG')
f0 = interp1d(large_fd_table_df.index, large_fd_table_df[0],kind=33)
f1 = interp1d(large_fd_table_df.index, large_fd_table_df[1],kind=33)
f2 = interp1d(large_fd_table_df.index, large_fd_table_df[2],kind=33)
f3 = interp1d(large_fd_table_df.index, large_fd_table_df[3],kind=33)

large_fd_table_df2 = pd.DataFrame()

new_index = np.arange(0, 300)
large_fd_table_df2[0] = f0(new_index)
large_fd_table_df2[1] = f1(new_index)
large_fd_table_df2[2] = f2(new_index)
large_fd_table_df2[3] = f3(new_index)

large_fd_table_df2.index = new_index
large_fd_table_df2.plot(style='--', figsize=(25, 12), grid=True)
plt.xlabel('Sample Set')
plt.ylabel('Fractal Dimension')
plt.title('FD of 300 Sample set of different classes - smoothened using cubic interpolation of degree to the power 33')
plt.legend(['Barren Lands', 'Trees', 'Grasslands', 'Urban'])
plt.savefig('FD of 300 Sample set of different classes - smoothened using cubic interpolation of degree to the power 33.PNG')
large_fd_table_df.to_csv('large_fd_table.csv')
model = Sequential([
    Dense(4, input_shape=(3136,), activation='softmax')
])
model.summary()
X_train = X_train/255
callbacks = [EarlyStopping(monitor='val_loss', patience=8),
                ModelCheckpoint(filepath='best_model.h5',
                                monitor='val_loss',
                                save_best_only=True)]
def evaluate_model(history, X_test, y_test, model):
    scores = model.evaluate((X_test), y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    print(history)
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig('Accuracy graph.PNG')
    plt.show()
    
    fig2, ax_loss = plt.subplots()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model - Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.savefig('Loss graph.PNG')
    plt.show()
    target_names = [str(i) for i in range(5)]
    
    y_true = []
    for element in y_test:
        y_true.append(np.argmax(element))
    prediction_proba = model.predict(X_test)
    prediction = np.argmax(prediction_proba, axis=1)
    cnf_matrix = confusion_matrix(y_true, prediction)
    return cnf_matrix
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train,Y_train,batch_size=32,
          callbacks=callbacks,
          epochs=15, verbose=1,
          validation_split=0.01)
preds = model.predict(X_train[-1000:], verbose=1)
conf_matrix = evaluate_model(history, X_train[-1000:], Y_train[-1000:], model)
conf_matrix
df_cm = pd.DataFrame(conf_matrix, range(4), range(4))
plt.figure(figsize=(10,7))
sns.set(font_scale=0.9) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.savefig('confusion_matrix.png')
plt.show()
df_cm = pd.DataFrame(conf_matrix, range(4), range(4)).corr()
plt.figure(figsize=(10,7))
sns.set(font_scale=0.9) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='summer') # font size

plt.savefig('confusion_matrix corr plot.png')
plt.show()
ix = 20 #Type a number between 0 and 999 inclusive
imshow(np.squeeze(X_train_img[99999-(1000-ix),:,:,0:3]).astype(float)*255) #Only seeing the RGB channels
plt.show()
#Tells what the image is
print ('Prediction:\n{:.1f}% probability barren land,\n{:.1f}% probability trees,\n{:.1f}% probability grassland,\n{:.1f}% probability other\n'.format(preds[ix,0]*100,preds[ix,1]*100,preds[ix,2]*100,preds[ix,3]*100))

print ('Ground Truth: ',end='')
if Y_train[99999-(1000-ix),0] == 1:
    print ('Barren Land')
elif Y_train[99999-(1000-ix),1] == 1:
    print ('Trees')
elif Y_train[99999-(1000-ix),2] == 1:
    print ('Grassland')
else:
    print ('Other')