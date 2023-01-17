import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from tensorflow.keras.layers import Input, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
import pickle as pkl
import os
x = pkl.load(open('/kaggle/input/melanoma-pickled-files/numpy_image(3).pkl', 'rb'))
target = pkl.load(open('/kaggle/input/melanoma-pickled-files/numpy_target(2).pkl', 'rb'))
x_test = pkl.load(open('/kaggle/input/melanoma-pickled-files/numpy_image_test(2).pkl', 'rb'))
!pip install -q efficientnet
import efficientnet.tfkeras as efn

target = keras.utils.to_categorical(target)
train_details = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
train_details
train_details['sex'] = train_details['sex'].fillna('male')
train_details['age_approx'] = train_details['age_approx'].fillna(train_details['age_approx'].mean())
train_details['anatom_site_general_challenge'] = train_details['anatom_site_general_challenge'].fillna('head/neck')
from category_encoders import TargetEncoder
enc1 = TargetEncoder()
enc2 = TargetEncoder()

train_details['sex'] = enc1.fit_transform(train_details['sex'], train_details['target'])
train_details['anatom_site_general_challenge'] = enc2.fit_transform(train_details['anatom_site_general_challenge'], train_details['target'])

x_vec = train_details[['sex','age_approx','anatom_site_general_challenge']]
image_input = Input((144, 144, 3))
vector_input = Input((3,))

enet = efn.EfficientNetB7(input_shape=(144, 144, 3), weights='imagenet', include_top=False, pooling = 'avg')
enet_result = enet(image_input)
flat_layer = Flatten()(enet_result)

concat_layer= Concatenate()([vector_input, flat_layer])
output = Dense(2, activation = 'sigmoid')(concat_layer)

model = Model(inputs=[image_input, vector_input], outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss = 'binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

class_weights = {0:98, 1:2}

model.fit([x,x_vec], target, epochs=5, validation_split = 0.15, class_weight = class_weights)
test_details = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
test_details.head()
test_details['sex'] = test_details['sex'].fillna('male')
test_details['age_approx'] = test_details['age_approx'].fillna(test_details['age_approx'].mean())
test_details['anatom_site_general_challenge'] = test_details['anatom_site_general_challenge'].fillna('head/neck')

test_details['sex'] = enc1.transform(test_details['sex'])
test_details['anatom_site_general_challenge'] = enc2.transform(test_details['anatom_site_general_challenge'])

x_vec_test = test_details[['sex','age_approx','anatom_site_general_challenge']]
preds = model.predict([x_test, x_vec_test])
preds = [float(x[1]) for x in preds]

submission = {'image_name': test_details['image_name'], 'target': preds}
submission = pd.DataFrame(submission)

os.chdir('/kaggle/working')
submission.to_csv(r'efficientnetb7_pred.csv', index = False)
