!pip install imutils
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from keras.utils import to_categorical
from keras.layers import Dense, Input, Conv2D, Flatten, MaxPool2D, Activation,Dropout
from keras.models import Model, model_from_json, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D

from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator


from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
MODEL_WEIGHTS_SAVE_PATH = "./output/trained_model_weights.h5"
MODEL_JSON_SAVE_PATH = "./output/trained_model.json"

MODEL_CHECKPOINT_PATH = "./output/model_checkpoint.h5"

TRAIN_VALIDATION_PLOT = "./output/train_validation_plot.png"


if not (os.path.exists("./output/")):
    os.makedirs("./output")

#Declaring constants
FIG_WIDTH=20 # Width of figure
ROW_HEIGHT=3 # Height of each row when showing a figure which consists of multiple rows


RESIZE_DIM=32 # The images will be resized to 50x50 pixels

CHKPT = False

BATCH_SIZE = 32
EPOCHS = 33
LEARNING_RATE = 1e-4
data_dir=os.path.join('..','input')
paths_train_a=glob.glob(os.path.join(data_dir,'training-a','*.png'))
paths_train_b=glob.glob(os.path.join(data_dir,'training-b','*.png'))
paths_train_e=glob.glob(os.path.join(data_dir,'training-e','*.png'))
paths_train_c=glob.glob(os.path.join(data_dir,'training-c','*.png'))
paths_train_d=glob.glob(os.path.join(data_dir,'training-d','*.png'))
paths_train_all=paths_train_a+paths_train_b+paths_train_c+paths_train_d+paths_train_e

paths_test_a=glob.glob(os.path.join(data_dir,'testing-a','*.png'))
paths_test_b=glob.glob(os.path.join(data_dir,'testing-b','*.png'))
paths_test_e=glob.glob(os.path.join(data_dir,'testing-e','*.png'))
paths_test_c=glob.glob(os.path.join(data_dir,'testing-c','*.png'))
paths_test_d=glob.glob(os.path.join(data_dir,'testing-d','*.png'))
paths_test_f=glob.glob(os.path.join(data_dir,'testing-f','*.png'))+glob.glob(os.path.join(data_dir,'testing-f','*.JPG'))
paths_test_auga=glob.glob(os.path.join(data_dir,'testing-auga','*.png'))
paths_test_augc=glob.glob(os.path.join(data_dir,'testing-augc','*.png'))
paths_test_all=paths_test_a+paths_test_b+paths_test_c+paths_test_d+paths_test_e+paths_test_f+paths_test_auga+paths_test_augc

path_label_train_a=os.path.join(data_dir,'training-a.csv')
path_label_train_b=os.path.join(data_dir,'training-b.csv')
path_label_train_e=os.path.join(data_dir,'training-e.csv')
path_label_train_c=os.path.join(data_dir,'training-c.csv')
path_label_train_d=os.path.join(data_dir,'training-d.csv')
def connectedComp(img):

	img = cv2.threshold(img, 105, 255, cv2.THRESH_BINARY)[1]  # ensure binary
	ret, labels = cv2.connectedComponents(img)

	# Map component labels to hue val
	label_hue = np.uint8(179*labels/np.max(labels))
	blank_ch = 255*np.ones_like(label_hue)
	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

	# cvt to BGR for display
	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

	# set bg label to black
	labeled_img[label_hue==0] = 0

	return img

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray,33,33,99)
    thresh_gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 2)
    equ = connectedComp(thresh_gray)
    color = cv2.cvtColor(equ,cv2.COLOR_GRAY2RGB)
    return color


def get_key(path):
    # seperates the name of the image from the path
    key=path.split(sep=os.sep)[-1]
    return key

  
def get_data(paths_img,path_label=None,resize_dim=None):
    '''reads images from the filepaths, resizes them (if given), and returns them in a numpy array
    Args:
        paths_img: image filepaths
        path_label: pass image label filepaths while processing training data, defaults to None while processing testing data
        resize_dim: if given, the image is resized to resize_dim x resize_dim (optional)
    Returns:
        X: group of images
        y: categorical true labels
    '''
    X=[] # initialize empty list for resized images
    
    for i,path in enumerate(paths_img):
        img=cv2.imread(path,cv2.IMREAD_COLOR) # images loaded in color (BGR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = process_image(img)
        if resize_dim is not None:
            img=cv2.resize(img,(resize_dim,resize_dim),interpolation=cv2.INTER_AREA) # resize image to 28x28
#         X.append(np.expand_dims(img,axis=2)) # expand image to 28x28x1 and append to the list.
        X.append(img) # expand image to 28x28x1 and append to the list
        # display progress
        if i==len(paths_img)-1:
            end='\n'
        else: end='\r'
        print('processed {}/{}'.format(i+1,len(paths_img)),end=end)
        
    X=np.array(X) # tranform list to numpy array
    if  path_label is None:
        return X
    else:
        df = pd.read_csv(path_label) # read labels
        df=df.set_index('filename') 
        y_label=[df.loc[get_key(path)]['digit'] for path in  paths_img] # get the labels corresponding to the images
        y=to_categorical(y_label,10) # transfrom integer value to categorical variable
        return X, y

      
def imshow_group(X,y=None,y_pred=None,n_per_row=10):
    '''helper function to visualize a group of images along with their categorical true labels (y) and prediction probabilities.
    Args:
        X: images
        y: categorical true labels
        y_pred: predicted class probabilities
        n_per_row: number of images per row to be plotted
    '''
    n_sample=len(X)
    img_dim=X.shape[1]
    j=np.ceil(n_sample/n_per_row)
    fig=plt.figure(figsize=(FIG_WIDTH,ROW_HEIGHT*j))
    for i,img in enumerate(X):
        plt.subplot(j,n_per_row,i+1)
        plt.imshow(img)
        if y is not None:
                plt.title('true label: {}'.format(np.argmax(y[i])))
        if y_pred is not None:
            top_n=3 # top 3 predictions with highest probabilities
            ind_sorted=np.argsort(y_pred[i])[::-1]
            h=img_dim+4
            for k in range(top_n):
                string='pred: {} ({:.0f}%)\n'.format(ind_sorted[k],y_pred[i,ind_sorted[k]]*100)
                plt.text(img_dim/2, h, string, horizontalalignment='center',verticalalignment='center')
                h+=4
        plt.axis('off')
    plt.show()
X_train_a,y_train_a=get_data(paths_train_a,path_label_train_a,resize_dim=RESIZE_DIM)
X_train_b,y_train_b=get_data(paths_train_b,path_label_train_b,resize_dim=RESIZE_DIM)
X_train_c,y_train_c=get_data(paths_train_c,path_label_train_c,resize_dim=RESIZE_DIM)
X_train_d,y_train_d=get_data(paths_train_d,path_label_train_d,resize_dim=RESIZE_DIM)
X_train_e,y_train_e=get_data(paths_train_e,path_label_train_e,resize_dim=RESIZE_DIM)
X_train_all=np.concatenate((X_train_a,X_train_b,X_train_c,X_train_d,X_train_e),axis=0)
y_train_all=np.concatenate((y_train_a,y_train_b,y_train_c,y_train_d,y_train_e),axis=0)
X_train_all.shape, y_train_all.shape
X_test_a=get_data(paths_test_a,resize_dim=RESIZE_DIM)
X_test_b=get_data(paths_test_b,resize_dim=RESIZE_DIM)
X_test_c=get_data(paths_test_c,resize_dim=RESIZE_DIM)
X_test_d=get_data(paths_test_d,resize_dim=RESIZE_DIM)
X_test_e=get_data(paths_test_e,resize_dim=RESIZE_DIM)
X_test_f=get_data(paths_test_f,resize_dim=RESIZE_DIM)
X_test_auga=get_data(paths_test_auga,resize_dim=RESIZE_DIM)
X_test_augc=get_data(paths_test_augc,resize_dim=RESIZE_DIM)
X_test_all=np.concatenate((X_test_a,X_test_b,X_test_c,X_test_d,X_test_e,X_test_f,X_test_auga,X_test_augc))
(Xtrain,Xvalid,Ytrain,Yvalid) = train_test_split(X_train_all,y_train_all,test_size=0.2, shuffle=True, random_state=94743)
Xtrain = Xtrain.astype('float')/255.0
Xvalid = Xvalid.astype('float')/255.0
n_sample = 50

ind=np.random.randint(0,len(Xtrain), size=n_sample)
imshow_group(X=Xtrain[ind])
datagen = ImageDataGenerator(
    rotation_range=35,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1)
class SmallVGGNet:
    @staticmethod
    def build(width,height,depth,classes):
        model = Sequential()
        
        inputShape = (height,width,depth)
        chanDim = -1
        
        if(K.image_data_format()=='channels_first'):
            inputShape = (depth,height,width)
            chanDim = 1
            
        model.add(Conv2D(32,(3,3),padding="same",input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        
        model.add(Conv2D(128,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        
        model.add(Conv2D(256,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(256,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
#         model.add(Dense(1024))
#         model.add(Activation("relu"))
#         model.add(BatchNormalization())
#         model.add(Dropout(0.5))
        
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
#         if(CHKPT):
#           model.load_weights(MODEL_CHECKPOINT_PATH)
        
        model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
        return model
K.tensorflow_backend.clear_session() # destroys the current graph and builds a new one
model=SmallVGGNet.build(RESIZE_DIM,RESIZE_DIM,3,10)# create the model
K.set_value(model.optimizer.lr, LEARNING_RATE) # set the learning rate
model.summary()
h=model.fit_generator(
    datagen.flow(Xtrain, Ytrain, batch_size=BATCH_SIZE),
    
    steps_per_epoch=len(Xtrain)/EPOCHS,
    
    epochs=EPOCHS, 
            verbose=1, 
            validation_data=(Xvalid,Yvalid),
            shuffle=True,
            callbacks=[
                ModelCheckpoint(filepath=MODEL_CHECKPOINT_PATH),
                EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=0, mode='auto',restore_best_weights=True)
            ]
            )
# plot the training loss and accuracy
H=h
N = np.arange(0,len(H.history["loss"]))
plt.style.use("ggplot")
plt.figure()

plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (LeNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(TRAIN_VALIDATION_PLOT)
predictions_prob=model.predict(X_test_all) # get predictions for all the testing data

n_sample=200
np.random.seed(42)
ind=np.random.randint(0,len(X_test_all), size=n_sample)
imshow_group(X=X_test_all[ind],y=None,y_pred=predictions_prob[ind])
labels=[np.argmax(pred) for pred in predictions_prob]
labels
keys=[get_key(path) for path in paths_test_all ]
keys
def create_submission(predictions,keys,path):
    result = pd.DataFrame(
        predictions,
        columns=['label'],
        index=keys
        )
    result.index.name='key'
    result.to_csv(path, index=True)
create_submission(predictions=labels,keys=keys,path='submission_simple_keras_starter.csv')
!ls