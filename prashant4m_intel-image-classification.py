import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing import image
inception = InceptionV3(input_shape=(150,150,3),
                       include_top=False,
                       weights='imagenet')
#inception.summary()
for layer in inception.layers:
    layer.trainable = False
last_layer = inception.get_layer('mixed7')
last_output = last_layer.output
last_layer.output_shape
from tensorflow.keras.optimizers import Adam

def create_model(model,feature_extractor):
# Flatten the output layer to 1 dimension
    x = Flatten()(feature_extractor)

    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = Dense(512,activation='relu')(x)
    # Add a dropout rate of 0.2
    x = Dropout(0.2)(x) 
    
    # Add a fully connected layer with 256 hidden units and ReLU activation
    x = Dense(256,activation='relu')(x)
    # Add a dropout rate of 0.2
    x = Dropout(0.2)(x) 
    
    # Add a fully connected layer with 128 hidden units and ReLU activation
    x = Dense(128,activation='relu')(x)
    # Add a dropout rate of 0.2
    x = Dropout(0.2)(x) 
    
    # Add a final softmax layer for classification
    x = Dense(6,activation="softmax")(x)           
    
    model = Model(model.input, x) 
    
    model.compile(optimizer = Adam(lr=0.0001), 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])
    return model
#model.summary()
train_dir = r'../input/intel-image-classification/seg_train/seg_train'
validation_dir = r'../input/intel-image-classification/seg_test/seg_test'
# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255,rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True)
    
# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1.0/255)
def data_generator(target_size):
    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                       target_size=target_size,
                                                       batch_size=32,
                                                       class_mode = 'categorical'
                                                       )     
    
    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                       target_size=target_size,
                                                       batch_size=32,
                                                       class_mode = 'categorical'
                                                       )
    return train_generator,validation_generator
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
callbacks = [EarlyStopping(monitor='val_acc',patience=4),
    tf.keras.callbacks.ModelCheckpoint(filepath='/kaggle/working/model.{epoch:02d}-{val_loss:.2f}.h5',save_best_only=True)
]
def fit_model(model,train_generator,validation_generator,epochs,callbacks,dis):
    print(dis)
    history = model.fit_generator(train_generator,validation_data=validation_generator,epochs=epochs,callbacks=[callbacks])
    return history
model.save(r"/kaggle/working/{}.h5".format("Using_InceptionV3"))
train_generator.class_indices
pred_dir = r'../input/intel-image-classification/seg_pred/seg_pred'
filename = os.listdir(pred_dir)
classes = ['Buildings','Forest','Glacier','Mountain','Sea','Street']
def predict(model,img,i,target_size):
    plt.subplot(2,3,i)
    img = image.load_img(img,target_size=target_size)
    img = image.img_to_array(img)
    img  = img / 255.0
    probabilities = model.predict(img.reshape(1,target_size[0],target_size[1],target_size[2]))
    plt.imshow(img)
    print(classes[np.argmax(probabilities)]+"("+str(np.max(probabilities))+")")
    
# Import VGG19 pre-trained Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
vgg19 = VGG19(input_shape=(224,224,3),
           include_top=False,
           weights='imagenet')
mbnet = MobileNetV2(input_shape=(224,224,3),
                   include_top = False,
                   weights='imagenet')
vgg19.summary()
#mbnet.summary()
epochs = 40
def vgg_model():    
    last_layer = vgg19.get_layer('block5_pool')
    last_output = last_layer.output
    vgg19_model = create_model(vgg19,last_output)
    
    print("Generating Image Data...")
    train_gen,valid_gen = data_generator((224,224))
    
    discription = "Fitting and Fine Tuning VGG19 Model..."
    history = fit_model(vgg19_model,train_gen,valid_gen,epochs,callbacks,discription)
    return vgg19_model,history

vgg19_model,history = vgg_model() #Accuracy:
#Saving the model
vgg19_model.save(r"/kaggle/working/{}.h5".format("Using_VGG19"))
#Predict Using VGG Model
n = 6 # Number of image used for prediction
j = 1
indices = np.random.randint(0,7300,n)
for i in indices:
    predict(vgg19_model,os.path.join(pred_dir,filename[i]),j,(224,224,3))
    j = j + 1