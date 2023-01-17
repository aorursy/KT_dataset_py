# import libraries

from IPython.display import Image, display

import numpy as np

import os

from os.path import join

from PIL import ImageFile

import pandas as pd

from matplotlib import cm

import seaborn as sns

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.ensemble import IsolationForest

from sklearn import svm

from sklearn.mixture import GaussianMixture

from sklearn.isotonic import IsotonicRegression

import re



ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.style.use('fivethirtyeight')

%matplotlib inline
# import car images from natural images

train_img_dir_n = "../input/natural-images/data/natural_images/car"

train_img_paths_n = [join(train_img_dir_n,filename) for filename in os.listdir(train_img_dir_n)]
# import car images from stanford cars

train_img_dir_s = "../input/stanford-cars-dataset/cars_train/cars_train"

all_train_img_paths_s = [join(train_img_dir_s,filename) for filename in os.listdir(train_img_dir_s)]



# split cars data into train, test, and val

train_img_paths, test_img_paths_car = train_test_split(all_train_img_paths_s+train_img_paths_n, test_size=0.25, random_state=42)

train_img_paths, val_img_paths_car = train_test_split(train_img_paths, test_size=0.25, random_state=42)
#  import ~car images

natural_images_path = "../input/natural-images/data/natural_images/"

test_img_paths_no_car = []

for d in [d for d in os.listdir("../input/natural-images/data/natural_images") if d!= "car"]:

    test_img_dir_na = natural_images_path+d

    test_img_paths_no_car.append([join(test_img_dir_na,filename) for filename in os.listdir(test_img_dir_na)])

    

test_img_paths_no_car_flat = [item for sublist in test_img_paths_no_car for item in sublist]

test_img_paths_no_car, val_img_paths_no_car = train_test_split(test_img_paths_no_car_flat, test_size = 0.25, random_state = 42)
def natural_img_dir(image_path):

    path_regex = r"natural_images\/(\w*)"

    if 'natural_images' in image_path:

        return re.findall(path_regex,image_path,re.MULTILINE)[0].strip()

    else:

        return 'car'
# create test dataframe

all_test_paths = test_img_paths_car+test_img_paths_no_car

test_path_df = pd.DataFrame({

    'path': all_test_paths,

    'is_car': [1 if path in test_img_paths_car else 0 for path in all_test_paths]

})

test_path_df = shuffle(test_path_df,random_state = 0).reset_index(drop = True)

test_path_df['image_type'] = test_path_df['path'].apply(lambda x: natural_img_dir(x))

all_test_paths = test_path_df['path'].tolist()
print('Distribution of Image Types in Test Set')

print(test_path_df['image_type'].value_counts())
# create val dataframe

all_val_paths = val_img_paths_car+val_img_paths_no_car

val_path_df = pd.DataFrame({

    'path': all_val_paths,

    'is_car': [1 if path in val_img_paths_car else 0 for path in all_val_paths]

})

val_path_df = shuffle(val_path_df,random_state = 0).reset_index(drop = True)

val_path_df['image_type'] = val_path_df['path'].apply(lambda x: natural_img_dir(x))

all_val_paths = val_path_df['path'].tolist()
print('Distribution of Image Types in Validation Set')

print(val_path_df['image_type'].value_counts())
# prepare images for resnet50

image_size = 224



def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):

    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    #output = img_array

    output = preprocess_input(img_array)

    return(output)



X_train = read_and_prep_images(train_img_paths)

X_test = read_and_prep_images(all_test_paths)

X_val = read_and_prep_images(all_val_paths)
# get features from resnet50 



resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



# X : images numpy array

resnet_model = ResNet50(input_shape=(image_size, image_size, 3), weights=resnet_weights_path, include_top=False, pooling='avg')  # Since top layer is the fc layer used for predictions



X_train = resnet_model.predict(X_train)

X_test = resnet_model.predict(X_test)

X_val = resnet_model.predict(X_val)
# Apply standard scaler to output from resnet50

ss = StandardScaler()

ss.fit(X_train)

X_train = ss.transform(X_train)

X_test = ss.transform(X_test)

X_val = ss.transform(X_val)



# Take PCA to reduce feature space dimensionality

pca = PCA(n_components=512, whiten=True)

pca = pca.fit(X_train)

print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))

X_train = pca.transform(X_train)

X_test = pca.transform(X_test)

X_val = pca.transform(X_val)
# Train classifier and obtain predictions for OC-SVM

oc_svm_clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)  # Obtained using grid search

if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)  # Obtained using grid search



oc_svm_clf.fit(X_train)

if_clf.fit(X_train)



oc_svm_preds = oc_svm_clf.predict(X_test)

if_preds = if_clf.predict(X_test)



# Further compute accuracy, precision and recall for the two predictions sets obtained
svm_if_results=pd.DataFrame({

  'path': all_test_paths,

  'oc_svm_preds': [0 if x == -1 else 1 for x in oc_svm_preds],

  'if_preds': [0 if x == -1 else 1 for x in if_preds]

})





svm_if_results=svm_if_results.merge(test_path_df)

svm_if_results.head()
print('roc auc score: if_preds')

if_preds=svm_if_results['if_preds']

actual=svm_if_results['is_car']

print(roc_auc_score(actual, if_preds))

print(classification_report(actual, if_preds))

sns.heatmap(confusion_matrix(actual, if_preds),annot=True,fmt='2.0f')

plt.show()
print('roc auc score: oc_svm_preds')

oc_svm_preds=svm_if_results['oc_svm_preds']

actual=svm_if_results['is_car']

print(roc_auc_score(actual, oc_svm_preds))

print(classification_report(actual, oc_svm_preds))

sns.heatmap(confusion_matrix(actual, oc_svm_preds),annot=True,fmt='2.0f')

plt.show()
y_val = val_path_df['is_car'].tolist()



gmm_clf = GaussianMixture(covariance_type='spherical', n_components=18, max_iter=int(1e7))  # From Article (These params should be optimized for this problem)

gmm_clf.fit(X_train)

log_probs_val = gmm_clf.score_samples(X_val)

isotonic_regressor = IsotonicRegression(out_of_bounds='clip')

isotonic_regressor.fit(log_probs_val, y_val)  # y_val is for labels 0 - not car 1 - car (validation set)



# Obtaining results on the test set

log_probs_test = gmm_clf.score_samples(X_test)

test_probabilities = isotonic_regressor.predict(log_probs_test)

test_predictions = [1 if prob >= 0.5 else 0 for prob in test_probabilities]

gmm_results = pd.DataFrame({

  'path': all_test_paths,

  'gmm_preds': test_predictions

})



gmm_results = gmm_results.merge(test_path_df)

gmm_results.head()
print('roc auc score: gmm_preds')

gmm_preds = gmm_results['gmm_preds']

actual = gmm_results['is_car']

print(roc_auc_score(actual, gmm_preds))

print(classification_report(actual, gmm_preds))

sns.heatmap(confusion_matrix(actual, gmm_preds),annot = True,fmt = '2.0f')

plt.show()
print('False Positive Actual Image Types for OC SVM: ')

print(svm_if_results[svm_if_results['oc_svm_preds']>svm_if_results['is_car']]['image_type'].value_counts())
for index, row in svm_if_results[svm_if_results['oc_svm_preds']!=svm_if_results['is_car']].head(25).iterrows():

    if row['oc_svm_preds']==1:

        print('FALSE POSITIVE')

        print('oc_svm_preds: ' + str(row['oc_svm_preds']) + ' | actual: '+ str(row['is_car']))

        display(Image(row['path']))

    else:

        print('FALSE NEGATIVE')

        print('oc_svm_preds: ' + str(row['oc_svm_preds']) + ' | actual: '+ str(row['is_car']))

        display(Image(row['path']))
print('False Positive Actual Image Types for IF: ')

print(svm_if_results[svm_if_results['if_preds']>svm_if_results['is_car']]['image_type'].value_counts())
for index, row in svm_if_results[svm_if_results['if_preds']!=svm_if_results['is_car']].head(25).iterrows():

    if row['if_preds']==1:

        print('FALSE POSITIVE')

        print('if_preds: ' + str(row['if_preds']) + ' | actual: '+ str(row['is_car']))

        display(Image(row['path']))

    else:

        print('FALSE NEGATIVE')

        print('if_preds: ' + str(row['if_preds']) + ' | actual: '+ str(row['is_car']))

        display(Image(row['path']))
print('False Positive Actual Image Types for GMM: ')

print(gmm_results[gmm_results['gmm_preds']>gmm_results['is_car']]['image_type'].value_counts())
for index, row in gmm_results[gmm_results['gmm_preds']!=gmm_results['is_car']].head(25).iterrows():

    if row['gmm_preds']==1:

        print('FALSE POSITIVE')

        print('gmm_preds: ' + str(row['gmm_preds']) + ' | actual: '+ str(row['is_car']))

        display(Image(row['path']))

    else:

        print('FALSE NEGATIVE')

        print('gmm_preds: ' + str(row['gmm_preds']) + ' | actual: '+ str(row['is_car']))

        display(Image(row['path']))