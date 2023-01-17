#Load the modules

from glob import glob 

import os

import numpy as np

import pandas as pd

import itertools

import shutil

from sklearn.utils import shuffle

import keras,math

!pip install opencv-python==4.0.0.21

import cv2 as cv



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation

from keras.layers import Conv2D, MaxPool2D



import gc #garbage collection, we need to save all the RAM we can



from tqdm import tqdm_notebook,trange

import matplotlib.pyplot as plt



import gc #garbage collection, we need to save all the RAM we can
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print(len(os.listdir('../input/train')))

print(len(os.listdir('../input/test')))
#set paths to training and test data

path = "../input/" #adapt this path, when running locally

train_path = path + 'train/'

test_path = path + 'test/'



df = pd.DataFrame({'path': glob(os.path.join(train_path,'*.tif'))}) # load the filenames

df['id'] = df.path.map(lambda x: x.split('/')[3].split(".")[0]) # keep only the file names in 'id'

labels = pd.read_csv(path+"train_labels.csv") # read the provided labels

df = df.merge(labels, on = "id") # merge labels and filepaths

df.head(3) # print the first three entrys
def load_data(N,df):

    """ This functions loads N images using the data df

    """

    # allocate a numpy array for the images (N, 96x96px, 3 channels, values 0 - 255)

    X = np.zeros([N,96,96,3],dtype=np.uint8) 

    #convert the labels to a numpy array too

    y = np.squeeze(df.as_matrix(columns=['label']))[0:N]

    #read images one by one, tdqm notebook displays a progress bar

    for i, row in tqdm_notebook(df.iterrows(), total=N):

        if i == N:

            break

        X[i] = cv.imread(row['path'])

          

    return X,y

N=10000

X,y = load_data(N=N,df=df) 
fig = plt.figure(figsize=(10, 4), dpi=150)

np.random.seed(100) #we can use the seed to get a different set of random images

for plotNr,idx in enumerate(np.random.randint(0,N,8)):

    ax = fig.add_subplot(2, 8//2, plotNr+1, xticks=[], yticks=[]) #add subplots

    plt.imshow(X[idx]) #plot image

    ax.set_title('Label: ' + str(y[idx])) #show the label corresponding to the image
fig = plt.figure(figsize=(4, 2),dpi=150)

plt.bar([1,0], [(y==0).sum(), (y==1).sum()]); #plot a bar chart of the label frequency

plt.xticks([1,0],["Negative (N={})".format((y==0).sum()),"Positive (N={})".format((y==1).sum())]);

plt.ylabel("# of samples")
img = cv.imread('../input/train/019ce31cc317087ca287f66ad757776952826594.tif')

r, g, b = cv.split(img)

r_avg = cv.mean(r)[0]

g_avg = cv.mean(g)[0]

b_avg = cv.mean(b)[0]

 

k = (r_avg + g_avg + b_avg) / 3

kr = k / r_avg

kg = k / g_avg

kb = k / b_avg

 

r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)

g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)

b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

 

balance_img = cv.merge([b, g, r])



plt.figure(figsize=(25,12))

plt.subplot(121)

plt.imshow(img)

plt.subplot(122)

plt.imshow(balance_img)
train_dir = '../input/train'

train_imgs = ['../input/train/{}'.format(i) for i in os.listdir(train_dir)]



plt.figure(figsize=(25,12))

for idx, train_img in enumerate(train_imgs):

    if idx >= 15:

        break

    

    temp_img = cv.imread(train_img, cv.IMREAD_COLOR)        

    

    

    plt.subplot(3,5, idx + 1)

    plt.imshow(temp_img)
plt.figure(figsize=(25,12))

for idx, train_img in enumerate(train_imgs):

    if idx >= 15:

        break

    temp_img = cv.imread(train_img, cv.IMREAD_COLOR)

    r, g, b = cv.split(temp_img)

    r_avg = cv.mean(r)[0]

    g_avg = cv.mean(g)[0]

    b_avg = cv.mean(b)[0]

    k = (r_avg + g_avg + b_avg) / 3

    kr = k / r_avg

    kg = k / g_avg

    kb = k / b_avg

    r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)

    g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)

    b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

    balance_img = cv.merge([b, g, r])



    plt.subplot(3 ,5, idx + 1)

    plt.imshow(balance_img)
N = 130000

X,y = load_data(N=N,df=df)
plt.imshow(X[632])
for i in tqdm_notebook(range(len(X))):

    r ,g, b = cv.split(X[i])

    r_avg = cv.mean(r)[0]

    g_avg = cv.mean(g)[0]

    b_avg = cv.mean(b)[0]

 

    k = (r_avg + g_avg + b_avg) / 3

    kr = k / r_avg

    kg = k / g_avg

    kb = k / b_avg

 

    r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)

    g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)

    b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

 

    X[i] = cv.merge([b, g, r])
plt.imshow(X[632])
#Collect garbage

positives_samples = None

negative_samples = None

gc.collect();
training_portion = 0.8 # Specify training/validation ratio

split_idx = int(np.round(training_portion * y.shape[0])) #Compute split idx



np.random.seed(42) #set the seed to ensure reproducibility



#shuffle

idx = np.arange(y.shape[0])

np.random.shuffle(idx)

X = X[idx]

y = y[idx]
#just some network parameters, see above link regarding the layers for details

kernel_size = (3,3)

pool_size= (2,2)

first_filters = 32

second_filters = 64

third_filters = 128



#dropout is used for regularization here with a probability of 0.3 for conv layers, 0.5 for the dense layer at the end

dropout_conv = 0.3

dropout_dense = 0.5



#initialize the model

model = Sequential()



#now add layers to it

#conv block 1

model.add(Conv2D(first_filters, kernel_size, input_shape = (96, 96, 3)))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Conv2D(first_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size = pool_size)) 

model.add(Dropout(dropout_conv))



#conv block 2

model.add(Conv2D(second_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Conv2D(second_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size = pool_size))

model.add(Dropout(dropout_conv))



#conv block 3

model.add(Conv2D(third_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Conv2D(third_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size = pool_size))

model.add(Dropout(dropout_conv))



#a fully connected (also called dense) layer at the end

model.add(Flatten())

model.add(Dense(256, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Dropout(dropout_dense))



#finally convert to values of 0 to 1 using the sigmoid activation function

model.add(Dense(1, activation = "sigmoid"))
model.summary()
batch_size = 50



model.compile(loss=keras.losses.binary_crossentropy,

              optimizer=keras.optimizers.Adam(0.001), 

              metrics=['accuracy'])
epochs = 12

for epoch in range(epochs):

    iterations = np.floor(split_idx / batch_size).astype(int)

    loss,acc = 0,0 

    with trange(iterations) as t: 

        for i in t:

            start_idx = i * batch_size 

            x_batch = X[start_idx:start_idx+batch_size] 

            y_batch = y[start_idx:start_idx+batch_size] 



            metrics = model.train_on_batch(x_batch, y_batch) #train the model on a batch



            loss = loss + metrics[0] 

            acc = acc + metrics[1] 

            t.set_description('Running training epoch ' + str(epoch)) 

            t.set_postfix(loss="%.2f" % round(loss / (i+1),2),acc="%.2f" % round(acc / (i+1),2)) 
iterations = np.floor((y.shape[0]-split_idx) / batch_size).astype(int) 

loss,acc = 0,0 

with trange(iterations) as t: 

    for i in t:

        start_idx = i * batch_size 

        x_batch = X[start_idx:start_idx+batch_size] 

        y_batch = y[start_idx:start_idx+batch_size] 

        

        metrics = model.test_on_batch(x_batch, y_batch) 

        

        loss = loss + metrics[0] 

        acc = acc + metrics[1] 

        t.set_description('Running training') 

        t.set_description('Running validation')

        t.set_postfix(loss="%.2f" % round(loss / (i+1),2),acc="%.2f" % round(acc / (i+1),2))

        

print("Validation loss:",loss / iterations)

print("Validation accuracy:",acc / iterations)
X = None

y = None

gc.collect();
base_test_dir = path + 'test/' #specify test data folder

test_files = glob(os.path.join(base_test_dir,'*.tif')) #find the test file names

submission = pd.DataFrame() #create a dataframe to hold results

file_batch = 5000 #we will predict 5000 images at a time

max_idx = len(test_files) #last index to use

for idx in range(0, max_idx, file_batch): #iterate over test image batches

    print("Indexes: %i - %i"%(idx, idx+file_batch))

    test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]}) #add the filenames to the dataframe

    test_df['id'] = test_df.path.map(lambda x: x.split('/')[3].split(".")[0]) #add the ids to the dataframe

    test_df['image'] = test_df['path'].map(cv.imread) #read the batch

    K_test = np.stack(test_df["image"].values) #convert to numpy array

    for i in tqdm_notebook(range(len(K_test))):

        r ,g, b = cv.split(K_test[i])

        r_avg = cv.mean(r)[0]

        g_avg = cv.mean(g)[0]

        b_avg = cv.mean(b)[0]

 

        k = (r_avg + g_avg + b_avg) / 3

        kr = k / r_avg

        kg = k / g_avg

        kb = k / b_avg

 

        r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)

        g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)

        b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

 

        K_test[i] = cv.merge([b, g, r])

    predictions = model.predict(K_test,verbose = 1) #predict the labels for the test data

    test_df['label'] = predictions #store them in the dataframe

    submission = pd.concat([submission, test_df[["id", "label"]]])

submission.head() #display first lines
submission.to_csv("submission.csv", index = False, header = True)