import os

import json

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from scipy.misc import imread

from PIL import Image

from keras.models import load_model 

from keras.preprocessing.image import ImageDataGenerator
with open('../input/config/model.json') as json_file:

        model_config = json.load(json_file)['get_resnet50']

WEIGHT_PATH = '../input/resnet50/ResNet50-12-0.43.hdf5'

TEST_DATA_PATH = '../input/aptos2019-blindness-detection/test_images/'
model = load_model(WEIGHT_PATH)
test_data_gen = ImageDataGenerator(rescale=1./255)
submission_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

submission_df.head()
def read_and_preprocess_image(id_code):    

    image = imread(os.path.join(TEST_DATA_PATH, id_code + '.png'))

    image = np.array(Image.fromarray(image).resize(model_config['input_shape'][:-1][::-1])).astype(np.uint8)

    return image
id_codes = list(submission_df['id_code'])

x_test = np.array([read_and_preprocess_image(id_code) for id_code in id_codes])
test_generator = test_data_gen.flow(x_test)
class_labels = ['0', '1', '2', '3', '4']
steps_need = test_generator.n//test_generator.batch_size + 1

test_generator.reset() # you need to restart whenever you call the predict_generator.

pred = model.predict_generator(test_generator, steps = steps_need, verbose=1)
predicted_class_indices=np.argmax(pred,axis=-1)

predictions = [str(i) for i in predicted_class_indices]

predictions
print(np.unique(predictions))

print(len(predictions))
submission_df['diagnosis'] = predictions

submission_df.head()
submission_df.to_csv('submission.csv', index=False)