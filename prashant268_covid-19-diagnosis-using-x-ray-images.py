from keras.layers import Dense, Flatten, AveragePooling2D, Dropout

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



from skimage.segmentation import mark_boundaries

import lime

from lime import lime_image

import cv2
data_path = '/kaggle/input/chest-xray-covid19-pneumonia/Data'
baseModel = VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)



for layer in baseModel.layers:

  layer.trainable = False



headModel = baseModel.output

headModel = AveragePooling2D()(headModel)

headModel = Flatten()(headModel)

headModel = Dense(128, activation="relu")(headModel)

headModel = Dropout(0.2)(headModel)

headModel = Dense(3, activation='softmax')(headModel)



model = Model(inputs=baseModel.input, outputs=headModel)
model.summary()
from keras.optimizers import Adam

opt=Adam(learning_rate=0.0001)

model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale = 1./255,

                                   zoom_range = 0.2,

                                   rotation_range=15,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory(data_path + '/train',

                                                 target_size = (224, 224),

                                                 batch_size = 16,

                                                 class_mode = 'categorical',

                                                 shuffle=True)



test_set = test_datagen.flow_from_directory(data_path + '/test',

                                            target_size = (224, 224),

                                            batch_size = 16,

                                            class_mode = 'categorical',

                                            shuffle = False)
history = model.fit_generator(training_set,

                              validation_data=test_set,

                              epochs=130)    
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs=range(len(acc))
plt.plot(epochs,acc,label='Trainin_acc',color='blue')

plt.plot(epochs,val_acc,label='Validation_acc',color='red')

plt.legend()

plt.title("Training and Validation Accuracy")

plt.plot(epochs,loss,label='Training_loss',color='blue')

plt.plot(epochs,val_loss,label='Validation_loss',color='red')

plt.legend()

plt.title("Training and Validation loss")
class_dict = {0:'COVID19',

              1:'NORMAL',

              2:'PNEUMONIA'}
file_path =  '/test/COVID19/COVID19(129).jpg'

test_image = cv2.imread(data_path + file_path)

test_image = cv2.resize(test_image, (224,224),interpolation=cv2.INTER_NEAREST)

plt.imshow(test_image)

test_image = np.expand_dims(test_image,axis=0)

probs = model.predict(test_image)

pred_class = np.argmax(probs)



pred_class = class_dict[pred_class]



print('prediction: ',pred_class)

def find_true_class(file_path):

    true_class = None

    if 'COVID19' in file_path:

        true_class = 'COVID19'

    elif 'PNEUMONIA' in file_path:

        true_class = 'PNEUMONIA'

    elif 'NORMAL' in file_path:

        true_class = 'NORMAL'

    return true_class
def visualize(file_path,ax,text_loc):

    test_image = cv2.imread(data_path + file_path)

    test_image = cv2.resize(test_image, (224,224),interpolation=cv2.INTER_NEAREST)

    test_image = np.expand_dims(test_image,axis=0)

    probs = model.predict(test_image)

    pred_class = np.argmax(probs)

    pred_class = class_dict[pred_class]



    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(test_image[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)





    ax.imshow(mark_boundaries(temp, mask))

    fig.text(text_loc, 0.9, "Predicted Class: " + pred_class , fontsize=13)

    true_class = find_true_class(file_path)

    if true_class is not None:

        fig.text(text_loc, 0.86, "Actual Class: " + true_class , fontsize=13)

fig,ax = plt.subplots(1,3,figsize=(18,6))

visualize('/test/COVID19/COVID19(152).jpg',ax[0],0.15)

visualize('/test/NORMAL/NORMAL(1001).jpg',ax[1],0.4)

visualize('/test/PNEUMONIA/PNEUMONIA(1037).jpg',ax[2],0.7)
model.save('covid_model.h5')