# import standard libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# load train data

train_data = pd.read_csv('/kaggle/input/syde522/train.csv')
# different classes

train_data.Category.unique()
# reading images

from skimage.io import imread

import matplotlib.pyplot as plt



%matplotlib inline



train_data['train_file'] = train_data.Id.apply(lambda x: '/kaggle/input/syde522/train/train/{0}'.format(x))





plt.figure(figsize=(15, 4))

for idx, (_, entry) in enumerate(train_data.sample(n=5).iterrows()):

    

    plt.subplot(1, 5, idx+1)

    plt.imshow(imread(entry.train_file))

    plt.axis('off')

    plt.title(entry.Category)

    #plt.title(imread(entry.train_file).shape)

    

# Feature extraction by pre trained CNN



#..



from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input





CNNmodel = VGG16(weights=None,input_shape=(150, 150, 3), include_top=False, pooling='max',)

CNNmodel.load_weights('/kaggle/input/vgg16-weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')



img_path = '/kaggle/input/syde522/train/train/0011.png'

img = image.load_img(img_path, target_size=(150,150))



def getFeatures(img):

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    featur = CNNmodel.predict(x)

    return featur.reshape(1, 512)



#print(features.flatten().shape)
# implement k-NN

from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing



features=np.zeros((1,512))

labels=np.array('init')



for idx, (_, entry) in enumerate(train_data.iterrows()):

    img=imread(entry.train_file)

    feature = getFeatures(img)

    features=np.append(features,feature,axis=0)

    labels=np.append(labels,entry.Category)



    

labels=labels.reshape(features[:,1].size )



features=np.delete(features, 0, 0)

labels=np.delete(labels, 0, 0)

print(features.shape)

print(labels.shape)

#print(np.mean(features))





##Normalize feature space!!

#normalize(features, norm='l2', axis=1, copy=True, return_norm=False)

features = (preprocessing.normalize(features))

#print(np.mean(features))





neigh = KNeighborsClassifier(n_neighbors=7)

neigh.fit(features, labels)

# preparing the submission



import glob

import os



def predictor(imgID):

    # this will be your glorious classifier and not a random predictor

    img=imread('/kaggle/input/syde522/test/test/'+imgID)

    feature = getFeatures(img)

    #normalize

    feature = preprocessing.normalize(feature)

    prediction=neigh.predict(feature)

    return prediction[0]



test_files = glob.glob('/kaggle/input/syde522/test/test/*.png')

test_file_id = [os.path.basename(test_file) for test_file in test_files]



test_submission = pd.DataFrame({'Id': test_file_id, 'Category': [predictor(test_img_id) for test_img_id in test_file_id]})



test_submission.to_csv('submission.csv', index=False)