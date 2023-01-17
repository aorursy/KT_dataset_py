import numpy as np 
import pandas as pd 
import glob
import os
import cv2 
import matplotlib.pyplot as plt
import skimage.feature
%matplotlib inline
#With this (inline) backend, the output of plotting commands is displayed inline within frontends 
#like the Jupyter notebook, directly below the code cell that produced it. 
#The resulting plots will then also be stored in the notebook document.

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#To view the files in the input directory
# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/Train/train.csv')
display(train_data.head(5))
print('DETAILS OF TRAIN_DATA')
train_data.info()
train_imgs = sorted(glob.glob('../input/Train/*.jpg'), key=lambda name: int(os.path.basename(name)[:-4]))
train_dot_imgs = sorted(glob.glob('../input/TrainDotted/*.jpg'), key=lambda name: int(os.path.basename(name)[:-4]))
submission = pd.read_csv('../input/sample_submission.csv')
print('TRAIN_DATA.SHAPE:')
print(train_data.shape)
print('Number of Train Images: {:d}'.format(len(train_imgs)))
print('Number of Dotted-Train Images: {:d}'.format(len(train_dot_imgs)))
# Count of each type
hist = train_data.sum(axis=0)
print(hist)
sea_lions_types = hist[1:]
f, ax1 = plt.subplots(1,1,figsize=(5,5))
sea_lions_types.plot(kind='bar', title='Count of Sea Lion Types (Train)', ax=ax1)
plt.show()
index = 1
sl_counts = train_data.iloc[index]
print(sl_counts)
sl_counts = sl_counts[1:]
plt.figure()
sl_counts.plot(kind='bar', title='Count of Sea Lion Types index#5')
plt.show()
print(train_imgs[index])
img = cv2.imread(train_imgs[index])
img_dot = cv2.imread(train_dot_imgs[index])
#img_dot = cv2.cvtColor(cv2.imread(train_dot_imgs[index]), cv2.COLOR_BGR2RGB)
# absolute difference between Train and Train Dotted
img_diff = cv2.absdiff(img_dot,img)
mask_1 = cv2.cvtColor(img_dot, cv2.COLOR_BGR2GRAY)
mask_1[mask_1 < 20] = 0
mask_1[mask_1 > 0] = 255
mask_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask_2[mask_2 < 20] = 0
mask_2[mask_2 > 0] = 255
image_4 = cv2.bitwise_or(img_diff, img_diff, mask=mask_1)
image_5 = cv2.bitwise_or(image_4, image_4, mask=mask_2)
# convert to grayscale to be accepted by skimage.feature.blob_log
image_6 = cv2.cvtColor(image_5, cv2.COLOR_BGR2GRAY)
# detect blobs
blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)
# prepare the image to plot the results on
image_7 = cv2.cvtColor(image_6, cv2.COLOR_GRAY2BGR)

classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups", "error"]
indices=[0,1,2,3]
count_df = pd.DataFrame(index=indices,columns=classes).fillna(0)
for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        b,g,r = img_dot[int(y)][int(x)][:]
        
        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if r > 200 and b < 50 and g < 50: # RED
            count_df["adult_males"][0] += 1
            cv2.circle(image_7, (int(x),int(y)), 8, (0,0,255), 2)   
        elif r > 200 and b > 200 and g < 50: # MAGENTA
            count_df["subadult_males"][0] += 1
            cv2.circle(image_7, (int(x),int(y)), 8, (250,10,250), 2)            
        elif r < 100 and b < 100 and 150 < g < 200: # GREEN
            count_df["pups"][0] += 1
            cv2.circle(image_7, (int(x),int(y)), 8, (20,180,35), 2) 
        elif r < 100 and  100 < b and g < 100: # BLUE
            count_df["juveniles"][0] += 1 
            cv2.circle(image_7, (int(x),int(y)), 8, (180,60,30), 2)
        elif r < 150 and b < 50 and g < 100:  # BROWN
            count_df["adult_females"][0] += 1
            cv2.circle(image_7, (int(x),int(y)), 8, (0,42,84), 2)            
        else:
            count_df["error"][0] += 1
            cv2.circle(image_7, (int(x),int(y)), 8, (255,255,155), 2)
f, ax = plt.subplots(3,2,figsize=(10,16))
(ax1, ax2, ax3, ax4, ax5, ax6) = ax.flatten()
ax1.imshow(cv2.cvtColor(img[700:1200,2130:2639,:], cv2.COLOR_BGR2RGB))
ax1.set_title('Train')
ax2.imshow(cv2.cvtColor(img_dot[700:1200,2130:2639,:], cv2.COLOR_BGR2RGB))
ax2.set_title('Train Dotted')
ax3.imshow(cv2.cvtColor(img_diff[700:1200,2130:2639,:], cv2.COLOR_BGR2RGB))
ax3.set_title('Train_Diff = Train Dotted - Train')
ax4.imshow(cv2.cvtColor(image_5[700:1200,2130:2639,:], cv2.COLOR_BGR2RGB))
ax4.set_title('Mask blackened areas of Train_Diff')
ax5.imshow(image_6[700:1200,2130:2639], cmap='gray')
ax5.set_title('Grayscale for input to blob_log')
ax6.imshow(cv2.cvtColor(image_7[700:1200,2130:2639,:], cv2.COLOR_BGR2RGB))
ax6.set_title('Result')
plt.show()
count_df
#Initialize a dataframe to store coordinates
class_names = ['adult_females', 'adult_males', 'juveniles', 'pups', 'subadult_males']
file_names = os.listdir("../input/Train/")
file_names = sorted(file_names, key=lambda 
                    item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item)) 
# select a subset of files to run on
file_names = file_names[0:1]
# dataframe to store results in
coordinates_df = pd.DataFrame(index=file_names, columns=class_names)
for filename in file_names:
    
    # read the Train and Train Dotted images
    image_1 = cv2.imread("../input/TrainDotted/" + filename)
    image_2 = cv2.imread("../input/Train/" + filename)
    #initializing a 'cut' image, i.e template
    cut = np.copy(image_2)
    
    # absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_1,image_2)
    
    # mask out blackened regions from Train Dotted
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 20] = 0
    mask_1[mask_1 > 0] = 255
    
    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    mask_2[mask_2 < 20] = 0
    mask_2[mask_2 > 0] = 255
    
    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2) 
    
    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)
    
    # detect blobs
    blobs = skimage.feature.blob_log(image_3, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)
    
    adult_males = []
    subadult_males = []
    pups = []
    juveniles = []
    adult_females = [] 
    
    image_circles = image_1
    
    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        g,b,r = image_1[int(y)][int(x)][:]
        
        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if r > 200 and g < 50 and b < 50: # RED
            adult_males.append((int(x),int(y)))
            cv2.circle(image_circles, (int(x),int(y)), 20, (0,0,255), 10) 
        elif r > 200 and g > 200 and b < 50: # MAGENTA
            subadult_males.append((int(x),int(y))) 
            cv2.circle(image_circles, (int(x),int(y)), 20, (250,10,250), 10)
        elif r < 100 and g < 100 and 150 < b < 200: # GREEN
            pups.append((int(x),int(y)))
            cv2.circle(image_circles, (int(x),int(y)), 20, (20,180,35), 10)
        elif r < 100 and  100 < g and b < 100: # BLUE
            juveniles.append((int(x),int(y))) 
            cv2.circle(image_circles, (int(x),int(y)), 20, (180,60,30), 10)
        elif r < 150 and g < 50 and b < 100:  # BROWN
            adult_females.append((int(x),int(y)))
            cv2.circle(image_circles, (int(x),int(y)), 20, (0,42,84), 10)  
        #Cutting a 32x32 frame around each coordinate    
        cv2.rectangle(cut, (int(x)-112,int(y)-112),(int(x)+112,int(y)+112), 0,-1)
            
    coordinates_df["adult_males"][filename] = adult_males
    coordinates_df["subadult_males"][filename] = subadult_males
    coordinates_df["adult_females"][filename] = adult_females
    coordinates_df["juveniles"][filename] = juveniles
    coordinates_df["pups"][filename] = pups
f, ax = plt.subplots(1,1,figsize=(10,16))
ax.imshow(cv2.cvtColor(image_circles, cv2.COLOR_BGR2RGB))
plt.show()
x = []
y = []

for filename in file_names:    
    image = cv2.imread("../input/Train/" + filename) #in each training image,
    for lion_class in class_names: # for each class,
        for coordinates in coordinates_df[lion_class][filename]:#create a thumbfile for each coordinate
            thumb = image[coordinates[1]-32:coordinates[1]+32,coordinates[0]-32:coordinates[0]+32,:]
            if np.shape(thumb) == (64, 64, 3):
                x.append(thumb)
                y.append(lion_class)
for i in range(0,np.shape(cut)[0],224):
    for j in range(0,np.shape(cut)[1],224):                
        thumb = cut[i:i+64,j:j+64,:]
        if np.amin(cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)) != 0:
            if np.shape(thumb) == (64,64,3):
                x.append(thumb)
                y.append("negative")     
class_names.append("negative")
x = np.array(x)
y = np.array(y)
for lion_class in class_names:
    f, ax = plt.subplots(1,10,figsize=(12,1.5))
    f.suptitle(lion_class)
    axes = ax.flatten()
    j = 0
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
        for i in range(j,len(x)):
            if y[i] == lion_class:
                j = i+1
                a.imshow(cv2.cvtColor(x[i], cv2.COLOR_BGR2RGB))
                break
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, Cropping2D
from keras.utils import np_utils
encoder = LabelBinarizer()
encoder.fit(y)
y = encoder.transform(y).astype(float)
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64,64,3)))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(x, y, epochs=10, verbose=0)
model.summary()
img = cv2.imread("../input/Train/" + filename)

x_test = []

for i in range(0,np.shape(img)[0],64):
    for j in range(0,np.shape(img)[1],64):                
        thumb = img[i:i+64,j:j+64,:]        
        if np.shape(thumb) == (64,64,3):
            x_test.append(thumb)

x_test = np.array(x_test)
y_predicted = model.predict(x_test, verbose=0)
y_predicted = encoder.inverse_transform(y_predicted)
reference = pd.read_csv('../input/Train/train.csv')
reference.ix[0:0]
print(Counter(y_predicted).items())