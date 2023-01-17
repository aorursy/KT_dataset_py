#for modeling
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#for preprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
#for visualizing
import matplotlib.pyplot as plt

def create_model():
    """
    Creates CNN model's artitechture.
    """
    #creating the artitechure
    model=Sequential([
                     Convolution2D(32, (3, 3), input_shape=( 128, 128, 3), activation="relu"),#first Conv layer
                     MaxPooling2D(pool_size=(2,2)),#pooling layer
                     Convolution2D(64, (3, 3), activation="relu"),#second conv layer
                     MaxPooling2D(pool_size=(2,2)),#pooling for second layer
                     Flatten(),#flattening
                     Dense(units=128,activation="relu"),#hidden layer
                     Dense(units=1,activation="sigmoid")#output layer
    ])
    #compiling the model
    model.compile(
                    loss="binary_crossentropy", 
                     optimizer="adam",
                     metrics=["accuracy"]
                 )
    return model
def train_model(train_data,test_data):
    """
    Create model and fit it on train data and return the model back.
    """
    model=create_model()
    earlyStop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                              patience=5) #callback to stop early
    model.fit_generator(train_data,
                       validation_data=test_data,
                       epochs=10,
                       callbacks=[earlyStop]
                       )
    return model
def preprocessing(test_data=False,pred_data=False,path=None):
    """
    Taking the type of data into consideration, returns its preprocessed form.
    """
    if test_data:
        test_gen=ImageDataGenerator(rescale=1./255,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
        test_set=test_gen.flow_from_directory("/kaggle/input/dogs-cats-images/dataset/test_set/",
                                       target_size=(128, 128),
                                       batch_size=32,
                                       class_mode='binary')
        return test_set
    elif pred_data:
        from keras.preprocessing import image
        test_img=image.load_img(path,target_size=(128, 128))
        test_img=image.img_to_array(test_img)
        test_img=np.expand_dims(test_img,axis=0)
        return test_img
        
    else:
        
        train_gen=ImageDataGenerator(rescale=1./255,
                                      zoom_range=0.2,
                                      horizontal_flip=True
                                    )
        train_set=train_gen.flow_from_directory("/kaggle/input/dogs-cats-images/dataset/training_set/",
                                                target_size=(128, 128),
                                                batch_size=32,
                                                class_mode='binary')
        return train_set
       
train_data=preprocessing()
test_data=preprocessing(test_data=True)
model=train_model(train_data,test_data)
def show_preds(path,model=model):
    image=preprocessing(path=path,pred_data=True)
    pred_view=tf.keras.preprocessing.image.load_img(path,target_size=(224,224))
    predictions=model.predict(image)
    if predictions[0][0]==1:
        predictions="This Image Contains a dog."
    else:
        predictions="This Image Contains a cat."
    plt.imshow(pred_view)
    plt.title(predictions,color="Green")
    plt.xticks([]);
    plt.yticks([]);
show_preds("/kaggle/input/for-testing/d1.jpg") #only give paths and make predictions
show_preds("/kaggle/input/for-testing/c2.jpeg")
show_preds("/kaggle/input/for-testing/d2.jpg")
