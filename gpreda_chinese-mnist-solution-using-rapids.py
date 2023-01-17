%%time

import sys

!cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import cudf, cuml

import pandas as pd, numpy as np

from sklearn.model_selection import train_test_split, KFold

from cuml.neighbors import KNeighborsClassifier, NearestNeighbors

#import matplotlib.pyplot as plt

print('cuML version',cuml.__version__)
IMAGE_PATH = '..//input//chinese-mnist//data//data//'

IMAGE_WIDTH = 64

IMAGE_HEIGHT = 64

IMAGE_CHANNELS = 1

TEST_SIZE = 0.2

VAL_SIZE = 0.2
import os

os.listdir("..//input//chinese-mnist")
data_df=pd.read_csv('..//input//chinese-mnist//chinese_mnist.csv')
image_files = list(os.listdir(IMAGE_PATH))

print("Number of image files: {}".format(len(image_files)))
def create_file_name(x):

    

    file_name = f"input_{x[0]}_{x[1]}_{x[2]}.jpg"

    return file_name
data_df["file"] = data_df.apply(create_file_name, axis=1)
file_names = list(data_df['file'])

print("Matching image names: {}".format(len(set(file_names).intersection(image_files))))
print(f"Number of suites: {data_df.suite_id.nunique()}")

print(f"Samples: {data_df.sample_id.unique()}")
train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE, random_state=42, stratify=data_df["code"].values)
train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=42, stratify=train_df["code"].values)
print("Train set rows: {}".format(train_df.shape[0]))

print("Test  set rows: {}".format(test_df.shape[0]))

print("Val   set rows: {}".format(val_df.shape[0]))
import cv2

def read_image(file_name):

    image_data = cv2.imread(IMAGE_PATH + file_name, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image_data, (IMAGE_WIDTH * IMAGE_HEIGHT, 1))



    return image[0,:]
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(train_df['character'])

print(le.classes_)
def prepare_data(dataset,label_encoding=le):

    X = np.stack(dataset['file'].apply(read_image))

    y = label_encoding.transform(dataset['character'])

    return X, y
X_train, y_train = prepare_data(train_df)

X_val, y_val = prepare_data(val_df)

X_test, y_test = prepare_data(test_df)
for k in range(1,16):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_hat_p = knn.predict_proba(X_val)

    y_tr_hat_p = knn.predict_proba(X_train)

    y_pred = pd.DataFrame(y_hat_p).values.argmax(axis=1)

    y_tr_pred = pd.DataFrame(y_tr_hat_p).values.argmax(axis=1)

    acc = (y_pred==y_val).sum()/y_val.shape[0]

    acc_tr = (y_tr_pred==y_train).sum()/y_train.shape[0]

    print(f"k: {k} accuracy: {round(acc,3)} accuracy(train): {round(acc_tr,3)}")
from sklearn.metrics import classification_report

print(classification_report(y_val, y_pred, target_names=le.classes_))