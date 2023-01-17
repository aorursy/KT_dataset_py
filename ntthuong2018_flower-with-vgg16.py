from keras.preprocessing.image import load_img, save_img, img_to_array

import numpy as np

from scipy.optimize import fmin_l_bfgs_b

import time

from keras.applications.imagenet_utils import preprocess_input

from keras import backend as K

import tensorflow as tf

import keras

import matplotlib.pyplot as plt

import cv2

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image

import matplotlib.pyplot as plt 

import seaborn as sns

import pandas as pd 

import numpy as np 

import os 
def resizeImage(imgPath):

    import cv2

    #img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)

    img1 = plt.imread(imgPath)

 

    scale_percent = 50 # percent of original size

    width = int(img1.shape[1] * scale_percent / 100)

    height = int(img1.shape[0] * scale_percent / 100)

    dim = (width, height)

    # resize image

    resized = cv2.resize(img1, dim)

    return resized

def resizeImageSame(imgPath):

    import cv2

    img1 = plt.imread(imgPath)

    dim = (224, 224)

    # resize image

    resized = cv2.resize(img1, dim)

    return resized



def centering_image(imgpath):

    img = plt.imread(imgpath)

    

    if(img.shape[0] > img.shape[1]):

        tile_size = (int(img.shape[1]*224/img.shape[0]),224)

    else:

        tile_size = (224, int(img.shape[0]*224/img.shape[1]))

    

    img = cv2.resize(img, dsize=tile_size)

    size = [224,224]

    img_size = img.shape[:2]

    

    # centering

    row = (size[1] - img_size[0]) // 2

    col = (size[0] - img_size[1]) // 2

    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)

    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

     

    #out put 224*224px 

    img = img[16:240, 16:240]

    



    return resized


#os.listdir(""../input/")

l1 = len(os.listdir("../input/flower_photos/flower_photos/daisy"))

l2 = len(os.listdir("../input/flower_photos/flower_photos/roses"))

l3 = len(os.listdir("../input/flower_photos/flower_photos/dandelion"))

l4 = len(os.listdir("../input/flower_photos/flower_photos/sunflowers"))

l5 = len(os.listdir("../input/flower_photos/flower_photos/tulips"))

#import matplotlib.pyplot as plt; plt.rcdefaults()



objects = ('Daisy', 'Rose', 'Dandelion', 'Sunflower', 'Tulip')

y_pos = np.arange(len(objects))

performance = [l1,l2,l3,l4,l5]



plt.barh(y_pos,performance, align='center', alpha=0.5)

plt.yticks(y_pos, objects)

plt.ylabel('Type of Flower')

plt.title('Data set')



plt.show()




img1 = "../input/flower_photos/flower_photos/roses/4860145119_b1c3cbaa4e_n.jpg"

img2 = "../input/flower_photos/flower_photos/sunflowers/5979668702_fdaec9e164_n.jpg"

img3 = "../input/flower_photos/flower_photos/daisy/14167534527_781ceb1b7a_n.jpg"

img4 = "../input/flower_photos/flower_photos/dandelion/461632542_0387557eff.jpg"

img5 = "../input/flower_photos/flower_photos/tulips/14087792403_f34f37ba3b_m.jpg"

imgs = [img1, img2, img3, img4, img5]



def showImage():

    f, ax = plt.subplots(1, 5)

    f.set_size_inches(80, 30)

    for i in range(0,5):

        ax[i].imshow(resizeImage(imgs[i]))

    plt.axis("on")

    plt.show()

    

showImage()



def showImageCenter():

    f, ax = plt.subplots(1, 5)

    f.set_size_inches(80, 30)

    for i in range(0,5):

        ax[i].imshow(centering_image(imgs[i]))

    plt.axis("on")

    plt.show()

    

showImageCenter()
import os

x_ = list()

y = list()



for i in os.listdir("../input/flower_photos/flower_photos/daisy"):

    try:

        path = "../input/flower_photos/flower_photos/daisy/"+i

        img = centering_image(path)

        x_.append(img)

        #y.append("daisy")

        y.append(0)

    except:

        None

for i in os.listdir("../input/flower_photos/flower_photos/dandelion"):

    try:

        path = "../input/flower_photos/flower_photos/dandelion/"+i

        img = centering_image(path)

        x_.append(img)

        #y.append("dandelion")

        y.append(1)

    except:

        None

for i in os.listdir("../input/flower_photos/flower_photos/roses"):

    try:

        path = "../input/flower_photos/flower_photos/roses/"+i

        img = centering_image(path)

        x_.append(img)

        #y.append("rose")

        y.append(2)

    except:

        None

for i in os.listdir("../input/flower_photos/flower_photos/sunflowers"):

    try:

        path = "../input/flower_photos/flower_photos/sunflowers/"+i

        img = centering_image(path)

        x_.append(img)

        #y.append("sunflower")

        y.append(3)

    except:

        None

for i in os.listdir("../input/flower_photos/flower_photos/tulips"):

    try:

        path = "../input/flower_photos/flower_photos/tulips/"+i

        img = centering_image(path)

        x_.append(img)

        #y.append("tulip")

        y.append(4)

    except:

        None

x_ = np.array(x_)


from keras.utils.np_utils import to_categorical

y = to_categorical(y,num_classes = 5)



classNames = ["daisy","dandelion", "rose","sunflower","tulip"]

# test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_,y,test_size = 0.2,random_state = 20)

# validation and trains split

x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = 0.15,random_state = 40)



unique, counts = np.unique(y_train.argmax(axis=1), return_counts=True)

uniqueVal, countsVal = np.unique(y_val.argmax(axis=1), return_counts=True)

dict(zip(unique, counts))
dict(zip(uniqueVal, countsVal))
plt.bar(classNames, counts)

plt.bar(classNames, countsVal)

plt.show()
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout

from keras.layers import Dense, Flatten

from keras.models import Model

from keras.optimizers import Adam

from keras.optimizers import SGD

def createModel():

    _input = Input(shape=(224,224,3))



    conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(_input)

    conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)

    pool1  = MaxPooling2D((2, 2))(conv2)



    conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)

    conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)

    pool2  = MaxPooling2D((2, 2))(conv4)



    conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)

    conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)

    conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)

    pool3  = MaxPooling2D((2, 2))(conv7)



    conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)

    conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)

    conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)

    pool4  = MaxPooling2D((2, 2))(conv10)



    conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)

    conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)

    conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)

    pool5  = MaxPooling2D((2, 2))(conv13)



    flat   = Flatten()(pool5)

    dense1 = Dense(4096, activation="relu")(flat)

    #dropout = Dropout(0.5)(dense1)

    dense2 = Dense(1024, activation="relu")(dense1)

    dropout1 = Dropout(0.5)(dense2)

    output = Dense(5, activation="softmax")(dropout1)



    model  = Model(inputs=_input, outputs=output)

    #vgg16_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



    #model.compile(optimizer='adam',

    #              loss='categorical_crossentropy',

    #              metrics=['accuracy'])

    model.compile(optimizer=Adam(lr=0.0002),loss='categorical_crossentropy',metrics=['accuracy'])

    

    return model
modelFitLRate = createModel()

epochs = 10

history = modelFitLRate.fit(x_train, y_train, validation_data=(x_val, y_val),batch_size=64, epochs=epochs)
import matplotlib.pyplot as plt



def showResultPrediction(predictions):

    num_rows = 5

    num_cols = 3

    num_images = num_rows*num_cols

    plt.figure(figsize=(2*2*num_cols, 2*num_rows))

    for i in range(num_images):

      plt.subplot(num_rows, 2*num_cols, 2*i+1)

      plot_image(i, predictions, y_test, x_test)

      plt.subplot(num_rows, 2*num_cols, 2*i+2)

      plot_value_array(i, predictions, y_test)

    plt.show()

    

    

def showChartEpochAccuracy(history):

    # show a nicely formatted classification report

    print("[INFO] evaluating network...")

    # plot the training loss and accuracy

    N = epochs

    plt.style.use("ggplot")

    plt.figure()

    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")

    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")

    plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")

    plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")

    plt.title("Training Loss and Accuracy Dataset")

    plt.xlabel("Epochs #")

    plt.ylabel("Loss/Accuracy")

    #plt.xticks(sd)

    #plt.xticklabels(learningValue)

    plt.legend(loc="lower left")

    plt.show()

  



def showChartLearningRate(history):

    # show a nicely formatted classification report

    print("[INFO] evaluating network...")

    # plot the training loss and accuracy

    N = epochs

    plt.style.use("ggplot")

    plt.figure()

    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")

    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")

    plt.plot(np.arange(0, N), history.history["mean_absolute_error"], label="train_mean_absolute_error")

    plt.plot(np.arange(0, N), history.history["val_mean_absolute_error"], label="val_mean_absolute_error")

    plt.title("Training Loss and mean absolute error on Dataset")

    plt.xlabel("Epochs #")

    plt.ylabel("Loss/Mean_absolute_error")

    #plt.xticks(sd)

    #plt.xticklabels(learningValue)

    plt.legend(loc="lower left")

    plt.show()

  



def plot_image(i, predictions_array, true_label, img):

    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    img = cv2.resize(img,(128,128))

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    true_label= np.argmax(true_label)

    if predicted_label == true_label:

        color = 'blue'

    else:

        color = 'red'

  

    plt.xlabel("{} {:2.0f}% ({})".format(classNames[predicted_label],

                                100*np.max(predictions_array),

                                classNames[true_label]),

                                color=color)



def plot_value_array(i, predictions_array, true_label):

    predictions_array, true_label = predictions_array[i], true_label[i]

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    thisplot = plt.bar(range(5), predictions_array, color="#777777")

    plt.ylim([0, 1])

    predicted_label = np.argmax(predictions_array)

    true_label = np.argmax(true_label)

    thisplot[predicted_label].set_color('red')

    thisplot[true_label].set_color('blue')
showChartEpochAccuracy(history)


from keras.optimizers import SGD

from keras.callbacks import LearningRateScheduler

sd=[]

class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.losses = [1,1]



    def on_epoch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))

        sd.append(step_decay(len(self.losses)))

        print('lr:', step_decay(len(self.losses)))



epochs = 20

learning_rate = 0.0001

decay_rate = 5e-6



model = createModel()



adam = Adam(lr=learning_rate,decay=decay_rate)

sgd = SGD(lr=0.0001, momentum=0.9, decay=decay_rate)

#model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.compile(loss='mean_squared_error',optimizer=adam,metrics=['mean_absolute_error','accuracy'])

def step_decay(losses):

    i = float(2*np.sqrt(np.array(history.losses[-1])))

    #print('i:',i)

    if i <0.6:

        lrate=0.001*1/(1+0.7*len(history.losses))

        decay_rate=2e-6

    else:

        lrate =0.0001

   

    return lrate

history=LossHistory()

lrate=LearningRateScheduler(step_decay)



myhistory = model.fit(x_train,y_train,validation_data=(x_val, y_val),batch_size=64, epochs=epochs,callbacks=[history,lrate], verbose=1)

mypredict = model.predict(x_test)
evalute = model.evaluate(x_test, y_test)
print("Accuracy: {:.2f}%".format(evalute[2] * 100))  

print("Loss: {}".format(evalute[0])) 
# Example of a confusion matrix in Python

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 



results = confusion_matrix(y_test.argmax(axis=1), mypredict.argmax(axis=1))

print(results)

print ('Accuracy Score :',accuracy_score(y_test.argmax(axis=1), mypredict.argmax(axis=1)) )

print ('Report : ')

print (classification_report(y_test.argmax(axis=1), mypredict.argmax(axis=1)) )

print('Learning Rate', sd)
showChartLearningRate(myhistory)
showResultPrediction(mypredict)
#model = createModel() 

#history = model.fit(x_train, y_train, validation_data=(x_val, y_val),batch_size=128, epochs=20)

#predictions = model.predict(x_test)
def get_output_layer(model, layer_name):

    # get the symbolic outputs of each "key" layer (we gave them unique names).

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer = layer_dict[layer_name]

    return layer
model.summary()
def _load_image(img_path):

    img = image.load_img(img_path, target_size=(224,224))

    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    return img 

def cam(img_path):

  

    import numpy as np

    from keras.applications.vgg16 import decode_predictions

    import matplotlib.image as mpimg

    from keras import backend as K

    import pandas as pd

    import matplotlib.pyplot as plt

    %matplotlib inline

    #K.clear_session()

    

    img=mpimg.imread(img_path)

    plt.imshow(img)

    x = _load_image(img_path)

    preds = model.predict(x)

    

    #predictions = pd.DataFrame(decode_predictions(preds, top=3)[0],columns=['col1','category','probability']).iloc[:,1:]

    argmax = np.argmax(preds[0])

    output = model.output[:, argmax]

    last_conv_layer = model.get_layer('conv2d_26')

    grads = K.gradients(output, last_conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([x])

    for i in range(512):

        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)

    heatmap = np.maximum(heatmap, 0)

    heatmap /= np.max(heatmap)

    import cv2

    img = cv2.imread(img_path)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    hif = .8

    superimposed_img = heatmap * hif + img

    output = 'output.jpeg'

    cv2.imwrite(output, superimposed_img)

    img=mpimg.imread(output)

    plt.imshow(img)

    plt.axis('off')

   # plt.title(predictions.loc[0,'category'].upper())

    return None
cam(img1)
cam(img2)
cam(img3)
cam(img4)
cam(img5)
# Fine-tuning

from keras.models import Model

from keras import optimizers

from keras import applications

from keras.preprocessing.image import ImageDataGenerator



from keras.layers import GlobalAveragePooling2D,Dense,Flatten,Dropout

from keras.models import Sequential

from keras.callbacks import ModelCheckpoint

K.clear_session()



#load the VGG16 model without the final layers(include_top=False)

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

print('Loaded model!')



#Let's freeze the first 15 layers - if you see the VGG model layers below, 

# we are freezing till the last Conv layer.

for layer in base_model.layers[:15]:

    layer.trainable = False

    

base_model.summary()
# In the summary above of our base model, trainable params is 7,079,424



# Now, let's create a top_model to put on top of the base model(we are not freezing any layers of this model) 

top_model = Sequential()  

#top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))

top_model.add(Flatten(input_shape=base_model.output_shape[1:]))

top_model.add(Dense(1024, activation='relu'))

top_model.add(Dropout(0.5))

top_model.add(Dense(5, activation='softmax')) 

top_model.summary()
# In the summary above of our base model, trainable params is 2,565



# Let's build the final model where we add the top_model on top of base_model.

model = Sequential()

model.add(base_model)

model.add(top_model)

model.summary()



model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              metrics=['accuracy'])
# Time to train our model !

epochs = 10

batch_size=32

best_model_finetuned_path = 'best_finetuned_model_weight.hdf5'



train_datagen = ImageDataGenerator(

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



test_datagen = ImageDataGenerator()



train_generator = train_datagen.flow(

    x_train,y_train,

    batch_size=batch_size)



validation_generator = test_datagen.flow(

    x_val,y_val,

    batch_size=batch_size)



checkpointer = ModelCheckpoint(best_model_finetuned_path,save_best_only = True,verbose = 1)



history = model.fit_generator(

    train_generator,

    steps_per_epoch=len(x_train) // batch_size,

    epochs= epochs ,

    validation_data=validation_generator,

    validation_steps=len(x_val) // batch_size,

    callbacks=[checkpointer])
model.load_weights(best_model_finetuned_path)  

   

(eval_loss, eval_accuracy) = model.evaluate(  

     x_test, y_test, batch_size=batch_size, verbose=1)



print("Accuracy: {:.2f}%".format(eval_accuracy * 100))  

print("Loss: {}".format(eval_loss)) 
# Let's visualize some random test prediction.

def visualize_pred_vgg16(y_pred):

# plot a random sample of test images, their predicted labels, and ground truth

    fig = plt.figure(figsize=(16, 9))

    for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):

        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])

        ax.imshow(np.squeeze(x_test[idx]))

        pred_idx = np.argmax(y_pred[idx])

        true_idx = np.argmax(y_test[idx])

        ax.set_title("{} ({})".format(classNames[pred_idx], classNames[true_idx]),

                     color=("green" if pred_idx == true_idx else "red"))



visualize_pred_vgg16(model.predict(x_test))
showChartEpochAccuracy(history)