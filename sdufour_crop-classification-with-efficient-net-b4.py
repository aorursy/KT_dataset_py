!pip install git+https://github.com/qubvel/efficientnet
# LOAD LIBRARIES

import time

startNB = time.time()



import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, AvgPool2D, MaxPool2D , Flatten , Dropout, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import efficientnet.tfkeras as efn



from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics import recall_score, accuracy_score, precision_recall_fscore_support, classification_report,confusion_matrix, f1_score

from sklearn.utils import class_weight



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns





import cv2

import os



print(tf.__version__)



print('TensorFlow version =',tf.__version__)
# VERSION MAJOR and MINOR for logging

mm = 1; rr = 1



# Default batch size can be changed later

SEED = 36

BATCH_SIZE = 64

DIM = 128

img_size = DIM

LR = 1e-3

DECAY = 0.75



# BEGIN LOG FILE

f = open(f'log-{mm}-{rr}.txt','a')

print('Logging to "log-%i-%i.txt"'%(mm,rr))

f.write('TensorFlow version ={tf.__version__}')

f.write('#############################\n')

f.write(f'Trial mm={mm}, rr={rr}\n')

f.write('efNetB4, batch_size='+str(BATCH_SIZE)+', seed='+str(SEED)+', '+str(DIM)+'x'+str(DIM)+', fold=0, LR '+str(1e-3)+' with '+str(0.75)+' decay\n')

f.write('#############################\n')

f.close()
# Let's create a function that will import and label the image set

labels = ["jute", "maize", "sugarcane", "wheat", "rice"]



def get_data(data_dir):

    data = [] 

    path = os.path.join('/kaggle/input/', data_dir)

    for label in labels:

        path_label = os.path.join(path, label)

        for img in os.listdir(path_label):

            try:

                img_arr = cv2.imread(os.path.join(path_label, img), cv2.IMREAD_COLOR)

                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size

                # Images are charged as BGR we switch channels to RGB

                RGB_arr = resized_arr[:,:,[2,1,0]]

                data.append([RGB_arr, labels.index(label)])

            except Exception as e:

                print(img, e)                    

    return np.array(data)



def get_extra_data(data_dir):

    """

    Serves for the extra data set of 8 images

    """

    data = [] 

    path_label = os.path.join('/kaggle/input/', data_dir)

    for img in os.listdir(path_label):

        try:

            img_arr = cv2.imread(os.path.join(path_label, img), cv2.IMREAD_COLOR)

            resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size

            # Images are charged as BGR we switch channels to RGB

            RGB_arr = resized_arr[:,:,[2,1,0]]

            for label in labels:

                if label in img:

                    # print(img, label)

                    data.append([RGB_arr, labels.index(label)])

        except Exception as e:

            print(img, e)                    

    return np.array(data)
train_data = get_data('agriculture-crop-images/kag2')

test_extra_data =  get_extra_data('testssss/test_crop_image')
x_data = []

y_data = []



x_test_extra = []

y_test_extra = []



for feature, label in train_data:

    x_data.append(feature)

    y_data.append(label)

    

for feature, label in test_extra_data: 

    x_test_extra.append(feature)

    y_test_extra.append(label)
sns.set_style('darkgrid')

sns.countplot(y_data).set_title('Train data')
sns.set_style('darkgrid')

sns.countplot(y_test_extra).set_title('Test data')
X_train = np.array(x_data).reshape(-1, img_size, img_size, 3)

X_test_extra = np.array(x_test_extra).reshape(-1, img_size, img_size, 3)



# We convert numerical to one hot encoding

y_train = np.array(tf.keras.utils.to_categorical(y_data, num_classes=5))

y_test_extra = np.array(tf.keras.utils.to_categorical(y_test_extra, num_classes=5))



print(X_train.shape, y_train.shape, X_test_extra.shape, y_test_extra.shape)
def build_model():

    

    # We input the images we have reshaped in 3 channels (RGB)

    inp = tf.keras.Input(shape=(DIM,DIM,3))

    # We use the pretrained weights but not the top

    base_model = efn.EfficientNetB4(weights='imagenet',include_top=False, input_shape=(DIM,DIM,3))



    x = base_model(inp)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # use a strong droptout because we have few input images

    x = tf.keras.layers.Dropout(0.5)(x)

    # predict for 5 classes

    x = tf.keras.layers.Dense(5, activation='softmax',name='x1',dtype='float32')(x)

    

    model = tf.keras.Model(inputs=inp, outputs=x)

    opt = tf.keras.optimizers.Adam(lr=0.00001)

    model.compile(loss='categorical_crossentropy', optimizer = opt,\

              metrics=['categorical_accuracy'])

        

    return model
# CUSTOM LEARNING SCHEUDLE

LR_START = 1e-5

LR_MAX = 1e-3

LR_RAMPUP_EPOCHS = 5

LR_SUSTAIN_EPOCHS = 0

LR_STEP_DECAY = 0.75



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = LR_MAX * LR_STEP_DECAY**((epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)//10)

    return lr

    

lr2 = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)



rng = [i for i in range(100)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y); 

plt.xlabel('epoch',size=14); plt.ylabel('learning rate',size=14)

plt.title('Training Schedule',size=16); plt.show()
train_datagen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode="nearest"

)



# Test gen is useless in this case

# test_datagen = ImageDataGenerator()
class CustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, valid_data, target, fold, mm=0, rr=0, patience=10):

        self.valid_inputs = valid_data

        self.valid_outputs = target

        self.fold = fold

        self.patience = patience

        self.mm = mm

        self.rr = rr

        

    def on_train_begin(self, logs={}):

        self.valid_f1 = [0]

        

    def on_epoch_end(self, epoch, logs={}):

        # At the end of the epoch we predict the classes on validation

        preds = self.model.predict(self.valid_inputs)

        # we transform vector prediction into numerical class

        preds = np.argmax(preds,axis=1)



        # Calculate metrics

        p, r, f1score, support = precision_recall_fscore_support(self.valid_outputs,preds,average='macro')

        a = accuracy_score(self.valid_outputs,preds)



        # LOG TO FILE

        f = open('log-%i-%i.txt'%(self.mm,self.rr),'a')

        f.write('#'*25); f.write('\n')

        f.write('#### FOLD %i EPOCH %i\n'%(self.fold+1,epoch+1))

        f.write('#### PRECISION: p=%.5f' % (p) )

        f.write('#### RECALL: r=%.5f' % (r) )

        f.write('#### F1SCORE: f1=%.5f' % (f1score) )

        f.write('#### ACCURACY: a1=%.5f\n' % (a) )





        print('\n'); print('#'*25)

        print('#### FOLD %i EPOCH %i'%(self.fold+1,epoch+1))

        print('#### PRECISION: p=%.5f' % (p) )

        print('#### RECALL: r=%.5f' % (r) )

        print('#### F1SCORE: f1=%.5f' % (f1score) )

        print('#### ACCURACY: a1=%.5f' % (a) )

        print('#'*25)

        

        # Stop training after multiple epochs if validation f1 score is not improving

        self.valid_f1.append(f1score)

        x = np.asarray(self.valid_f1)

        if np.argsort(-x)[0]==(len(x)-self.patience-1):

            print('#### F1 no increase for %i epochs: EARLY STOPPING' % self.patience)

            f.write('#### F1 no increase for %i epochs: EARLY STOPPING\n' % self.patience)

            self.model.stop_training = True

            

        if (f1score>0.000)&(f1score>np.max(self.valid_f1[:-1])):

            print('#### Saving new best...')

            f.write('#### Saving new best...\n')

            self.model.save_weights('fold%i-m%i-%i.h5' % (self.fold,self.mm,self.rr))

            

        f.close()
# TRAIN MODEL

# Predictions will be saved for future uses

oo = np.zeros((X_train.shape[0],5))



# We will get 5 split of our data which could give 5 attempts at training

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)



# We retransform y with argmax because split method can't handle onehotencoding

for fold,(idxT,idxV) in enumerate(skf.split(X_train,np.argmax(y_train, axis=1))):

         

    print('#'*25)

    print('### FOLD %i' % (fold+1))

    print('### train on %i images. validate on %i images'%(len(idxT),len(idxV)))

    print('#'*25)

    

    K.clear_session()

    model = build_model()

    

    # We identify the training flow wwith the IDs given by split method

    # Shuffle is very important 

    train_flow = train_datagen.flow(

        x=X_train[idxT],

        y=y_train[idxT],

        batch_size=BATCH_SIZE,

        shuffle=True,

        sample_weight=None,

        seed=SEED

    )

    

    # CustomCallback will allow us to save best model weights

    cc = CustomCallback(valid_data=X_train[idxV], target=np.argmax(y_train[idxV], axis=1), fold=fold, mm=mm, rr=rr, patience=15)

    

    h = model.fit(train_flow, epochs = 30, verbose=1, callbacks=[cc, lr2])



    print('#### Loading best weights...')

    model.load_weights('fold%i-m%i-%i.h5' % (fold,mm,rr))

    

    oo = model.predict(X_train)



    # SAVE OOF and IDXV

    np.save('oo-%i-%i'%(mm,rr),oo)

    np.save('idxV-%i-%i'%(mm,rr),idxV)

    np.save('Y_train-%i-%i'%(mm,rr),y_train)

    # we will limit ourself to one fold

    break
def display_stats_and_confusion_matrix(y_pred, y, names=labels):

    """

    Print stats and display confustion matrix

    y must be provided as numerical values not one hot

    """

    cm = confusion_matrix(y,y_pred)

    cm = pd.DataFrame(cm , index = names , columns = names)

    cm.index.name = 'Label'

    cm.columns.name = 'Predicted'



    precision, recall, fscore, support = precision_recall_fscore_support(y_pred, y, average=None)

    print("#########################")

    for p,l in zip(precision, names):

        print("#### Precision for %s %.2f" % (l, p))

    print("#########################")        

    for r,l in zip(recall, names):

        print("#### Recall for %s %.2f" % (l, r))

    print("#########################")        

    for f,l in zip(fscore, names):

        print("#### F1Score for %s %.2f" % (l, f))

    print("#########################")



    group_counts = ["{0:0.0f}".format(value) for value in cm.to_numpy().flatten()]

    # Percentage are normalized so as to interpret read values

    group_percentages = ["{0:.2%}".format(value) for value in cm.to_numpy().flatten()/np.sum(cm.to_numpy())]

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(5,5)



    plt.figure(figsize = (10,10))

    sns.heatmap(cm,

                annot=labels,

                cmap= "coolwarm",

                linecolor = 'black',

                linewidth = 1,

                fmt='')    
y_pred_num = np.argmax(model.predict(X_train[idxV]),axis=1)

y_num = np.argmax(y_train[idxV],axis=1)

display_stats_and_confusion_matrix(y_pred_num,y_num)
y_pred_extra_num = np.argmax(model.predict(X_test_extra), axis=1)

y_extra_num = np.argmax(y_test_extra,axis=1)

display_stats_and_confusion_matrix(y_pred_extra_num,y_extra_num)
df = pd.DataFrame({'label':y_extra_num, 'prediction':y_pred_extra_num, 'img': X_test_extra.reshape(len(y_extra_num),-1).tolist()})
print("### Correctly classifed images")

plt.figure(figsize=(20,20))

i_ = 0



correctImages = df[df['label'] == df['prediction']]

for index, row in correctImages.iterrows():

    im = np.array(row['img']).reshape(DIM,DIM,3)

    actual_label = labels[row['label']]

    predicted = labels[row['prediction']]

    plt.subplot(5, 5, i_+1).title.set_text("Label: %s " % (predicted))

    plt.imshow(im)

    plt.axis('off')

    i_ += 1

    if index >= 25:

        break
print("### Misclassified images")

plt.figure(figsize=(20,20))

i_ = 0



multipleImages = df[df['label'] != df['prediction']]

for index, row in multipleImages.iterrows():

    im = np.array(row['img']).reshape(DIM,DIM,3)

    actual_label = labels[row['label']]

    predicted = labels[row['prediction']]

    plt.subplot(5, 5, i_+1).title.set_text("Label: %s Predicted: %s"%(actual_label, predicted))

    plt.imshow(im)

    plt.axis('off')

    i_ += 1

    if index >= 25:

        break