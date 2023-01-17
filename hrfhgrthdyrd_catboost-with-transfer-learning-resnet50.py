from catboost import CatBoostClassifier

import pandas as pd

import numpy as np

import h5py

from tqdm.notebook import tqdm

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.applications import ResNet50, imagenet_utils

from tensorflow.keras.preprocessing.image import img_to_array, load_img
db = h5py.File('/kaggle/input/resnet50-features-for-siimisic/featuresResNet50.hdf5', 'r')
model = CatBoostClassifier(iterations=1000, learning_rate=0.03,

                           loss_function='CrossEntropy', verbose=False,

                           custom_loss=['AUC'],

                           leaf_estimation_method='Newton', l2_leaf_reg=3,

                           task_type="GPU", devices='0:1')

model.fit(db['features'][:], db['labels'][:],

          plot=True)
db.close()
submission = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')

test_data = []

for i in range(test.shape[0]):

    test_data.append('/kaggle/input/siim-isic-melanoma-classification//jpeg/test/' + test['image_name'].iloc[i]+'.jpg')

df_test = pd.DataFrame(test_data)

df_test.columns = ['images']

ResNet50_model = Sequential()

ResNet50_model.add(ResNet50(weights='imagenet', include_top=False,

                            input_shape=(224, 224, 3)))

ResNet50_model.add(GlobalAveragePooling2D())

target = []

for imagePath in tqdm(df_test['images']):

    image = load_img(imagePath, target_size=(224, 224))

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = imagenet_utils.preprocess_input(image)

    features = ResNet50_model.predict(image, batch_size=1)

    features = features.reshape((features.shape[0], 2048))

    predict_prob = model.predict_proba(features)[0][1]

    target.append(predict_prob)

submission['target'] = target

submission.to_csv('catboost_submission.csv', index=False)

submission.head()