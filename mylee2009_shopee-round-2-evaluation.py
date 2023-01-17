import pandas as pd
import numpy as np
import tensorflow as tf
import time
import cv2
from tensorflow.keras.optimizers import SGD, RMSprop
import os
def load_models(json, h5):

    json_file = open(json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = tf.keras.models.model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(h5)
    return loaded_model

#loaded_model = model_from_json(loaded_model_json)
loaded_model = load_models('../input/xception-retraining-2/model_xception_v2_retrained.json', '../input/xception-retraining-2/model_xception_v2_retrained.h5')

opt = SGD(lr=.01, momentum=.9)
loaded_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
TEST_DIR = '../input/shopee-round-2-product-detection-challenge/test/test/'
test = pd.read_csv('../input/shopee-round-2-product-detection-challenge/test.csv')
test_images = [i for i in list(test.filename)]
from keras.preprocessing import image

results = []

for count,i in enumerate(test_images):    
# https://www.analyticsvidhya.com/blog/2019/01/build-image-classification-model-10-minutes/
    img = image.load_img(TEST_DIR+i, target_size=(299,299,3))
    img = image.img_to_array(img)
    img = np.reshape(img,[1,299,299,3])/255
    #img = img/255

    classes = np.argmax(loaded_model.predict(img), axis=1)[0]
    
    if count % 1000 == 0:
        print(count)
        # total around 12,000 rows
    
    results.append([i, classes])
results_df = pd.DataFrame(results, columns=['filename', 'category'])
results_df['category']=results_df['category'].astype(str).str.zfill(2)
results_df.head(20)
results_df.to_csv('results1.csv', index=False)
# #https://stackoverflow.com/questions/52270177/how-to-use-predict-generator-on-new-images-keras/52273258
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(
#         directory='../input/shopee-round-2-product-detection-challenge/test/',
#         target_size=(299, 299),
#         batch_size=20,
#         shuffle=False,
#         class_mode='categorical')

# pred=loaded_model.predict_generator(test_generator, steps=len(test_generator), verbose=1)


# # Get classes by np.round
# cl = np.round(pred)
# cl_argmax = np.argmax(cl, axis = 1)
# cl_argmax_fill = np.char.zfill(cl_argmax.astype(str), 2)
# # Get filenames (set shuffle=false in generator is important)
# filenames=[e[5:] for e in test_generator.filenames]


# intermediate=pd.DataFrame({"filename":filenames,"category":cl_argmax_fill})
# intermediate.head()

# results = []

# for i in test_images:
#     category = str(intermediate.loc[intermediate['filename']==i, 'category'].values[0])
    
#     results.append([i, category])
    
# results_df = pd.DataFrame(results, columns=['filename', 'category'])
# #results_df['category']=np.char.zfill(results_df['category'].astype(str), 2)
# results_df.head(20)

# results_df.to_csv('results.csv', index=False)

