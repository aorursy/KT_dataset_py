from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout

from keras.models import Sequential, Model

from keras.optimizers import SGD, Adam

from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
TRAIN_DIR = "/kaggle/input/datasetV2/dataset"

HEIGHT = 300

WIDTH = 300

BATCH_SIZE = 8

NUM_EPOCHS = 10

num_train_images = 10000
# ignore the fully connected layers with include_top=False

base_model = ResNet50(weights='imagenet', 

                      include_top=False,

                      input_shape=(HEIGHT, WIDTH, 3))





train_datagen =  ImageDataGenerator(

      preprocessing_function=preprocess_input,

      rotation_range=90,

      horizontal_flip=True,

      vertical_flip=True

    )



train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 

                                                    target_size=(HEIGHT, WIDTH), 

                                                    batch_size=BATCH_SIZE)
def build_finetune_model(base_model, dropout, fc_layers, num_classes):

    for layer in base_model.layers:

        layer.trainable = False



    x = base_model.output

    x = Flatten()(x)

    for fc in fc_layers:

        # New FC layer, random init

        x = Dense(fc, activation='relu')(x) 

        x = Dropout(dropout)(x)



    # New softmax layer

    predictions = Dense(num_classes, activation='softmax')(x) 

    

    finetune_model = Model(inputs=base_model.input, outputs=predictions)



    return finetune_model
class_list = ["code", "notcode"]

FC_LAYERS = [1024, 1024]

dropout = 0.5



finetune_model = build_finetune_model(base_model, 

                                      dropout=dropout, 

                                      fc_layers=FC_LAYERS, 

                                      num_classes=len(class_list))



adam = Adam(lr=0.00001)

finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])



# filepath="ResNet50_model_weights.h5"

# checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')

# callbacks_list = [checkpoint]

callbacks_list = []



history = finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8, 

                                       steps_per_epoch=num_train_images // BATCH_SIZE, 

                                       shuffle=True, callbacks=callbacks_list)
# model_json = finetune_model.to_json()

# with open("code_identifier_model.json", "w") as json_file:

#     json_file.write(model_json)



# # serialize weights to HDF5

# finetune_model.save_weights("code_identifier_model_weights.h5")



finetune_model.save('code_identifier_model.h5')  # creates a HDF5 file 'my_model.h5'

del finetune_model  # deletes the existing model



#### returns a compiled model identical to the previous one 

# from keras.models import load_model

# model = load_model('my_model.h5')
# from keras.models import load_model

# from keras.preprocessing import image

# from keras.applications.resnet50 import preprocess_input, decode_predictions

# import numpy as np



# model = load_model('model.h5')



# ## FROM SO ANSWER -----------------------------------------------------------------



# # model.compile(loss='binary_crossentropy',

# #               optimizer='rmsprop',

# #               metrics=['accuracy'])



# # img = cv2.imread('test.jpg')

# # img = cv2.resize(img,(320,240))

# # img = np.reshape(img,[1,320,240,3])



# # classes = model.predict_classes(img)



# # print classes



# ## ---------------------------------------------------------------------------------



# img_path = 'Data/Jellyfish.jpg'

# img = image.load_img(img_path, target_size=(224, 224))

# x = image.img_to_array(img)

# x = np.expand_dims(x, axis=0)

# x = preprocess_input(x)



# preds = model.predict(x)



# # decode the results into a list of tuples (class, description, probability)

# print('Predicted:', decode_predictions(preds, top=3)[0])
