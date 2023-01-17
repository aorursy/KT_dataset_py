import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import time 



import matplotlib.pyplot as plt



from keras import Sequential

from keras.layers import Dense, Dropout, InputLayer, Lambda, Input

from keras.preprocessing.image import load_img, img_to_array



from sklearn.model_selection import train_test_split
labels_df = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')

sample = pd.read_csv('/kaggle/input/dog-breed-identification/sample_submission.csv')
print('no of images in train dataset: {}'.format(len(labels_df)))

print('no of images in test dataset: {}'.format(len(sample)))
def images_to_array(directory, label_dataframe, target_size = (331, 331, 3)):

    images = np.zeros([len(label_dataframe), target_size[0], target_size[1], target_size[2]], dtype=np.uint8)

    img = ''

    for ix, image_name in enumerate(label_dataframe['id'].values):

        img_dir = os.path.join(directory, image_name + '.jpg')

        img = load_img(img_dir, target_size = target_size)

        images[ix] = img_to_array(img)

    del img

    label_dict = dict(enumerate(label_dataframe['breed'].unique()))

    return images, label_dict
t = time.time()

train_images, labels = images_to_array('/kaggle/input/dog-breed-identification/train', labels_df[:])

print('runtime in seconds: {}'.format(time.time() - t))
plt.figure(figsize = (20, 10))

for ix, image in enumerate(train_images[:16]):

    plt.subplot(4, 8, ix + 1)

    plt.imshow(image / 255.0)

    plt.xticks([])

    plt.yticks([])    
def get_feature(model_name, preprocess_input, images, target_size = (331,331,3)):

    base_model = model_name(input_shape = target_size, include_top=False, pooling = 'avg')



    model = Sequential()

    model.add(InputLayer(input_shape = target_size))

    model.add(Lambda(preprocess_input))

    model.add(base_model)



    feature = model.predict(images)

    

    print('feature-map shape: {}'.format(feature.shape))

    return feature
from keras.applications.inception_v3 import InceptionV3, preprocess_input



inception_preprocess = preprocess_input

inception_feature = get_feature(InceptionV3, preprocess_input, train_images)
from keras.applications.nasnet import NASNetLarge, preprocess_input

nasnet_preprocessor = preprocess_input

nasnet_features = get_feature(NASNetLarge, nasnet_preprocessor, train_images)
from keras.applications.xception import Xception, preprocess_input



xception_preprocess = preprocess_input

xception_feature = get_feature(Xception, xception_preprocess, train_images)
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input



resnet_preprocess = preprocess_input

resnet_feature = get_feature(InceptionResNetV2, resnet_preprocess, train_images)
final_features = np.concatenate([inception_feature, nasnet_features, xception_feature, resnet_feature], axis = 1)

print('final features shape: {}'.format(final_features.shape))

del train_images, inception_feature, nasnet_features, xception_feature, resnet_feature
class_to_index = dict({labels[ix]:ix for ix in labels.keys()})

index_to_class = labels
labels = labels_df['breed'].map(class_to_index)
def create_model(features_shape = 1024):

    model = Sequential()

    model.add(InputLayer(input_shape = (features_shape, )))

    #model.add(Dense(4096, activation = 'relu'))

    model.add(Dropout(0.7))

    model.add(Dense(len(class_to_index), activation = 'softmax'))

    

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer ='Adam', metrics = ['accuracy'])

    return model
#drop_neuron = [0.1, 0.2, 0.3]



#param_grid = dict(drop_neuron = drop_neuron)

#model = KerasClassifier(build_fn=create_model, epochs = 10, batch_size = 32)

#grid_search = GridSearchCV(estimator=model, param_grid=param_grid)

#result = grid_search.fit(x_train, y_train)
#means = result.cv_results_['mean_test_score']

#stds = result.cv_results_['std_test_score']

#params = result.cv_results_['params']

#print('best param: {}'.format(result.best_params_))

#for mean, stdev, param in zip(means, stds, params):

#    print("%f (%f) with: %r" % (mean, stdev, param))
model = create_model(final_features.shape[1])

model.summary()
model.fit(final_features, labels, epochs = 6, validation_split = 0.2)
del final_features, labels
def images_to_array(directory, label_dataframe, target_size = (331, 331,3)):

    images = np.zeros([len(label_dataframe), target_size[0], target_size[1], target_size[2]], dtype=np.uint8)

    img = ''

    for ix, image_name in enumerate(label_dataframe['id'].values):

        img_dir = os.path.join(directory, image_name + '.jpg')

        img = load_img(img_dir, target_size = target_size)

        images[ix] = img_to_array(img)

    del img

    return images
t = time.time()

test_images = images_to_array('/kaggle/input/dog-breed-identification/test', sample)

print('runtime in seconds: {}'.format(time.time() - t))
resnet_feature = get_feature(InceptionResNetV2, resnet_preprocess, test_images)

xception_feature = get_feature(Xception, xception_preprocess, test_images)

nasnet_features = get_feature(NASNetLarge, nasnet_preprocessor, test_images)

inception_feature = get_feature(InceptionV3, preprocess_input, test_images)
final_features = np.concatenate([inception_feature, nasnet_features, xception_feature, resnet_feature], axis = 1)

print('final features shape: {}'.format(final_features.shape))

del test_images, inception_feature, nasnet_features, xception_feature, resnet_feature
prediction = model.predict(final_features)

submission = pd.DataFrame({'id':sample.id})

prediction = pd.DataFrame(prediction)

prediction.columns = class_to_index.keys()
submission = pd.concat([submission, prediction], axis = 1)

submission.to_csv('submission.csv', index = False)