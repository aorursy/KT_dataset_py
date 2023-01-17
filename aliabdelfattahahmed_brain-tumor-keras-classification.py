# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install imutils
import os
import keras.backend as K
import imutils
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import glob
no_files=[]
yes_files=[]
for file in glob.glob("/kaggle/input/brain-mri-images-for-brain-tumor-detection/no/*.jpg"):
    no_files.append(file)
for file in glob.glob("/kaggle/input/brain-mri-images-for-brain-tumor-detection/yes/*.jpg"):
    yes_files.append(file)
        
def crop_brain_contour(image, plot=False):
    
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            

    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)
        plt.tick_params(axis='both', which='both',top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Cropped Image')
        plt.show()
    
    return new_image
import os
try:
    os.rmdir('../output/kaggle/working/crop/yes')
except:
    pass
try:
    os.rmdir('../output/kaggle/working/crop/no')
except:
    pass
try:
    os.makedirs('../output/kaggle/working/crop/yes')
except:
    pass
try:
    os.makedirs('../output/kaggle/working/crop/no')
except:
    pass


ex_img = cv2.imread('/kaggle/input/brain-mri-images-for-brain-tumor-detection/yes/Y107.jpg')
ex_crop_img = crop_brain_contour(ex_img, True)



for file in no_files:
    
    ex_img = cv2.imread(file)
    ex_crop_img = crop_brain_contour(ex_img, False)
    filename='../output/kaggle/working/crop/no/'+os.path.basename(file)
    
    cv2.imwrite(filename,ex_crop_img)
    
for file in yes_files:
    ex_img = cv2.imread(file)
    ex_crop_img = crop_brain_contour(ex_img, False)
    cv2.imwrite('../output/kaggle/working/crop/yes/'+os.path.basename(file),ex_crop_img)
    
import glob
no_files_crop=[]
yes_files_crop=[]
for file in glob.glob("../output/kaggle/working/crop/no/*.jpg"):
    no_files_crop.append(file)
for file in glob.glob("../output/kaggle/working/crop/yes/*.jpg"):
    yes_files_crop.append(file)
        
df=pd.DataFrame(columns=['filename','class'])
for file in no_files_crop:
    df=df.append({'filename':file,'class':'no'},ignore_index=True)

for file in yes_files_crop:
    df=df.append({'filename':file,'class':'yes'},ignore_index=True)
from sklearn.utils import shuffle
df_shuffle=shuffle(df)
df_shuffle.groupby(['class']).count().hist()
from sklearn.model_selection import train_test_split

X_train,X_=train_test_split(df_shuffle,test_size=0.2,random_state=0)

X_test,X_val=train_test_split(X_,test_size=0.2,random_state=0)



X_train.groupby(['class']).count()
X_val.groupby(['class']).count()
X_train
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_generator=ImageDataGenerator(rescale=1/255)

train_it = img_generator.flow_from_dataframe(X_train, class_mode='binary',
                                             featurewise_std_normalization=True,
                                             image_size=(256, 256))
test_it = img_generator.flow_from_dataframe(X_test, class_mode='binary',image_size=(256, 256),featurewise_std_normalization=True)
val_it = img_generator.flow_from_dataframe(X_val, class_mode='binary',image_size=(256, 256),featurewise_std_normalization=True)
import matplotlib.pyplot as plt

fig,ax=plt.subplots(2,2,figsize=(6,6))

images,labels = train_it.next()
x=0
y=0
for i in range(0,16):
    image = images[i]
    if y<2 and x<2:
        ax[y][x].imshow(image)
    if x>2:
        y=y+1
        x=0
    x=x+1
plt.show()

# example of tending the vgg16 model
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout

# load model without classifier layers
vgg = VGG16(include_top=False, input_shape=(256, 256, 3))
# add new classifier layers
flat1 = Flatten()(vgg.output)
dropout = Dropout(0.5)(flat1)
dense1 = Dense(1024, activation='relu')(flat1)
batch =  BatchNormalization()(dense1)
dense2 = Dense(1024, activation='relu')(batch)
dropout = Dropout(0.5)(dense2)
output = Dense(1, activation='sigmoid')(dropout)
# define new model
model = Model(inputs=vgg.inputs, outputs=output)
# summarize
model.summary()

for layer in vgg.layers:
    layer.trainable=False

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2)
callbacks=[reduce_lr]
#model.fit_generator(train_it, epochs=30,steps_per_epoch=5, 
#                    validation_data=val_it, validation_steps=3,callbacks=callbacks)
if os.path.isfile('/kaggle/working/model.h5'):
    model.load_weights("/kaggle/working/model.h5")
else:
    model.fit_generator(train_it, epochs=180,steps_per_epoch=5, 
                    validation_data=val_it, validation_steps=3,callbacks=callbacks)
model.save_weights('/kaggle/working/model.h5')
scores=model.evaluate_generator(generator=test_it,steps=10)
print("Accuracy = ", scores[1])
df_shuffle
df_yes=df_shuffle[df_shuffle['class']=='yes']
name=list(df_yes.iloc[1:2,:]['filename'])[0]
print(name)
ex_img = cv2.imread(name)
ex_img = cv2.resize(ex_img,(256,256))
plt.imshow(ex_img)

from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

x_reshape = ex_img.reshape((1, ex_img.shape[0],ex_img.shape[1], ex_img.shape[2]))

image = preprocess_input(x_reshape)

model.predict(image)
df_no=df_shuffle[df_shuffle['class']=='no']
name=list(df_no.iloc[1:2,:]['filename'])[0]
print(name)
ex_img = cv2.imread(name)
ex_img = cv2.resize(ex_img,(256,256))
plt.imshow(ex_img)

from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

x_reshape = ex_img.reshape((1, ex_img.shape[0],ex_img.shape[1], ex_img.shape[2]))

image = preprocess_input(x_reshape)

model.predict(image)
y_pred=[]
y_true=[]


for i,r in df_shuffle.iterrows():
    ex_img = cv2.imread(r['filename'])
    ex_img = cv2.resize(ex_img,(256,256))
    plt.imshow(ex_img)

    from tensorflow.keras.preprocessing import image
    from keras.applications.vgg16 import preprocess_input

    x_reshape = ex_img.reshape((1, ex_img.shape[0],ex_img.shape[1], ex_img.shape[2]))

    image = preprocess_input(x_reshape)
    
    pred=model.predict(image)
    
    y_true.append(r['class'])
    

    
 
    y_pred.extend(pred[0])
   
y_pred_=[int(y>0.2) for y in y_pred]
y_test=[1 if x=='yes' else 0 for x in y_true]
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred_)
def get_xai(x):
    
    
    x=x/255
    x_ = np.expand_dims(x, axis=0)
        

    y_pred = model.predict(x_)
    
    print('pred',y_pred)
    last_conv_layer = model.get_layer('block5_conv3')
    argmax = np.argmax(y_pred[0])
    print(argmax)
    output = model.output[:, argmax]
    print(output)
    print(last_conv_layer.output)
    grads = K.gradients(output, last_conv_layer.output)[0]

    #tf.print(grads)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))




    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])



    from keras.applications.vgg16 import preprocess_input
    #x = preprocess_input(x)
    #print( pooled_grads_value[pooled_grads_value>0])

    pooled_grads_value, conv_layer_output_value = iterate([x_])


    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    #plt.matshow(heatmap)
    #plt.show()
    import cv2
    heatmap = cv2.resize(heatmap, (x.shape[1], x.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = .05
    superimposed_img = heatmap * hif + x
    from matplotlib.pyplot import figure
    plt.figure(figsize=(5, 5))
    #fig, ax =figure(figsize=(10, 2))
    #ax.imshow(random.rand(8, 90), interpolation='nearest')
    plt.imshow(superimposed_img)
    #plt.axis('off')
    plt.show()
    
def get_heatmap(x):
    
    
    x=x/255
    x_ = np.expand_dims(x, axis=0)
        

    y_pred = model.predict(x_)
    
   
    last_conv_layer = model.get_layer('block5_conv3')
    argmax = np.argmax(y_pred[0])
    
    output = model.output[:, argmax]
   
    grads = K.gradients(output, last_conv_layer.output)[0]

    #tf.print(grads)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))




    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])



    from keras.applications.vgg16 import preprocess_input
    #x = preprocess_input(x)
    #print( pooled_grads_value[pooled_grads_value>0])

    pooled_grads_value, conv_layer_output_value = iterate([x_])


    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    #plt.matshow(heatmap)
    #plt.show()
    import cv2
    heatmap = cv2.resize(heatmap, (x.shape[1], x.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap
df_yes=df_shuffle[df_shuffle['class']=='yes']
name=list(df_yes.iloc[1:2,:]['filename'])[0]
print(name)
ex_img = cv2.imread(name)
ex_img = cv2.resize(ex_img,(256,256))
plt.imshow(ex_img)
get_xai(ex_img)
###from tqdm import tqdm

###images=[]
###predictions=[]


###for i,r in tqdm(df_shuffle.iterrows()):
###    ex_img = cv2.imread(r['filename'])
###    ex_img = cv2.resize(ex_img,(256,256))
###    plt.imshow(ex_img)

###    from tensorflow.keras.preprocessing import image
###    from keras.applications.vgg16 import preprocess_input

###    x_reshape = ex_img.reshape((1, ex_img.shape[0],ex_img.shape[1], ex_img.shape[2]))

###    image = preprocess_input(x_reshape)
    
###    pred=model.predict(image)
###    images.append(get_heatmap(ex_img).flatten())
###    predictions.extend(pred[0])
### import csv

### df_shuffle.to_csv('/kaggle/working/df.csv')

### with open('/kaggle/working/maps.csv', 'wb') as images:
###    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
###    wr.writerow(mylist)
