%reset -f

import os

import zipfile

import hashlib

from six.moves.urllib.error import URLError

from six.moves.urllib.request import urlretrieve

import argparse

from tqdm import tqdm

from keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet, decode_predictions

from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inception

from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg

from keras.preprocessing import image

from keras import backend as K

from keras.models import Model

from keras.layers import GlobalAveragePooling2D, Concatenate

import numpy as np



import numpy as np

import pandas as pd

import time, os, glob

import cv2



from keras.applications.vgg16 import VGG16

from keras.optimizers import SGD, Adam

from keras.layers import GlobalAveragePooling2D

from keras.models import Model

from keras.applications.imagenet_utils import preprocess_input



from os.path import isfile, join, abspath, exists, isdir, expanduser

from os import listdir, makedirs, getcwd, remove



# ! ls -la ../input/train_images
import pandas as pd 

import os

from glob import glob

from matplotlib.pyplot import imshow

import numpy as np

from PIL import Image



labels = pd.read_csv("../input/train.csv")

print(len(labels))

labels.head()





data_dir = '../input/'

train_dir = data_dir + '/train_images/'

test_dir = data_dir + '/test_images/'



imageList = glob(train_dir + '/**/*.png', recursive=True)

print ( "Number of images: {}". format (len (imageList)))

for img in imageList[0:5]:

    print(img)

    

%matplotlib inline

pil_im = Image.open(imageList[0], 'r')

imshow(np.asarray(pil_im))
from tqdm import tqdm # Enable progress bar

from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions # Load pre-trained model

from keras.models import Model

from keras.preprocessing import image

from keras.layers import Flatten, Input



    

class VGGFV:

    __name__ = "VGGFV"



    def __init__(self, feature_layer="block5_conv3", input_size=224,num_class=1,batch_size=32):

        base_model = VGG16(weights='imagenet', include_top=False,

                           input_shape=[input_size,input_size,3], classes=num_class)

                                        

        x2 = GlobalAveragePooling2D()(base_model.get_layer("block2_conv2").output)  # 128

        x3 = GlobalAveragePooling2D()(base_model.get_layer("block3_conv3").output)  # 256

        x4 = GlobalAveragePooling2D()(base_model.get_layer("block4_conv3").output)  # 512

        x5 = GlobalAveragePooling2D()(base_model.get_layer("block5_conv3").output)  # 512        

        x = Concatenate()([x2,x3,x4,x5])        



        model = Model(inputs=base_model.input, outputs=x)        

        optimizer = Adam(lr=0.0001)

        model.compile(loss='categorical_crossentropy',

                      optimizer=optimizer,

                      metrics=['accuracy'])

        

        self.model=model    

        self.batch_size = batch_size

        self.data_format = K.image_data_format()

    

    def predict(self, x):

        if self.data_format == "channels_first":

            x = x.transpose(0, 3, 1, 2)

        x = preprocess_vgg(x.astype(K.floatx()))

        return self.model.predict(x, batch_size=self.batch_size).squeeze()

    

def find_classes_melanoma(fullDir):

    def invert_dict(d):

        return dict([(v, k) for k, v in d.iter()])



    df_labels = pd.read_csv("../input/train.csv", sep=',')                

    class_to_idx={'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}        

    

    num_to_class= dict(zip(class_to_idx.values(), class_to_idx.keys()))            

    classes= ['0', '1', '2', '3', '4']



    print('Sorted Classes: {}'.format(classes))

    print('class_to_idx: {}'.format(class_to_idx))

    print('num_to_class: {}'.format(num_to_class))



    train = []

    for index, row in (df_labels.iterrows()):

        id = (row['id_code'])

        # currImage = os.path.join(fullDir, num_to_class[(int(row['melanoma']))] + '/' + id + '.jpg')

        currImage_on_disk = os.path.join(fullDir, id + '.png')

        if os.path.isfile(currImage_on_disk):

            if (int(row['diagnosis'])) == 0:

                trait = '0'

            elif (int(row['diagnosis'])) == 1:

                trait = '1'

            elif (int(row['diagnosis'])) == 2:

                trait = '2'

            elif (int(row['diagnosis'])) == 3:

                trait = '3'

            elif (int(row['diagnosis'])) == 4:                

                trait = '4'



            train.append(['{}'.format(currImage_on_disk), trait, class_to_idx[trait]])

    df = pd.DataFrame(train, columns=['file', 'category', 'category_id', ])

#         if os.path.isfile('full_melanoma_labels.csv'):

#             os.remove('full_melanoma_labels.csv')    

#         df.to_csv('full_melanoma_labels.csv', index=None)

    return classes, class_to_idx, num_to_class, df



# def find_classes(fullDir):    

#     print ("Full dir:{}".format(fullDir))

#     classes = [d for d in os.listdir(fullDir) if os.path.isdir(os.path.join(fullDir, d))]

#     classes.sort()

#     class_to_idx = {classes[i]: i for i in range(len(classes))}

#     num_to_class = dict(zip(range(len(classes)), classes))



#     print('Sorted Classes: {}'.format(classes))

#     print('class_to_idx: {}'.format(class_to_idx))

#     print('num_to_class: {}'.format(num_to_class))



#     train = []

#     for index, label in enumerate(classes):

#         path = fullDir + label + '/'

#         for file in listdir(path):

#             if file.endswith('tif') or file.endswith('jpg') or file.endswith('png'):

#                 train.append(['{}/{}/{}'.format(fullDir, label, file), label, index])



#     df = pd.DataFrame(train, columns=['file', 'category', 'category_id', ])



#     return classes, class_to_idx, num_to_class, df

    
feature_layer = "block5_conv3" # 512

input_size=224



model = VGGFV (feature_layer, input_size=input_size,num_class=1)



columns=['file', 'category_id', 'fv']

X_df_train=pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)

classes, class_to_idx, num_to_class, df= find_classes_melanoma(train_dir)

print (class_to_idx)



# Empty arrays for storing extracted features

X_train = []; X_test = []



# Extract bottleneck features of photos for traninig    

for index, row in tqdm(df.iterrows()):

    img = image.load_img(row['file'], target_size=(input_size, input_size))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)    

    vgg_feature_vector = model.predict(x)

    X_train.append(vgg_feature_vector)

#     print (vgg_feature_vector)

    

    X_df_train = X_df_train.append({ 'file':row['file'], 

                              'category_id':row['category'],                               

                              'fv':vgg_feature_vector },

                                 ignore_index=True)

    



# X_df_train.category_id.astype(int)    

print(X_df_train.info())    



X_df_train.to_csv('predictions_df.csv', index = False)

X_df_train.head(5)
import pandas 

X_df_train_SINGLE=X_df_train.copy(deep=True)

X_df_train_SINGLE.drop('file', axis=1, inplace=True)

answers_1_SINGLE = list (X_df_train_SINGLE['category_id'].values)

# answers_1_SINGLE= map(int, answers_1_SINGLE)

X_df_train_SINGLE = X_df_train_SINGLE.drop('category_id', axis=1)

X_df_train_SINGLE=X_df_train_SINGLE.apply(lambda x: pandas.to_numeric(x, errors='ignore'))



X_df_train_SINGLE.head(3)

# Function to make labels in the data frame into a list (i.e. 0 8 => [0, 8])

def labels_to_list(labels): return list(map(int, labels.split()))



# Process train & test set into an array format

X_train = np.array([x for x in X_df_train['fv']])

Y_train = np.array([list (X_df_train['category_id'].values)]).reshape(X_train.shape[0],1)

# Y_train=Y_train.reshape(2002,1)

# Y_train = np.array([labels_to_list(y) for y in X_df_train['category_id']])





# Check shape of array-format train & test set

print("X_train: ", X_train.shape)

print("Y_train: ", Y_train.shape)
# Load packages for splitting train & validation set, XGBoost classifier, 1-of-K encoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.multiclass import OneVsRestClassifier

from xgboost import XGBClassifier



# Package for estimating a time taken

import time; t=time.time()
from sklearn.utils import class_weight



# Convert list of labels to follow 1-of-K coding scheme

one_of_K_encoder = MultiLabelBinarizer()

Y_train_ = one_of_K_encoder.fit_transform(Y_train)



# Split train set into 8:2 (train : validation)

random_state = np.random.RandomState(999)

X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X_train, Y_train_, test_size=0.2, 

                                                        #stratify=['category_id'],

                                                        random_state=random_state)



mx_depth=9

# Train the XGBoost classifier

classifier = OneVsRestClassifier(XGBClassifier(num_class=len(classes), 

                                               gamma=0.024, 

                                               learning_rate=0.3, 

                                               max_depth=20, 

                                               nthread=4, 

                                               n_estimators=1000,

                                               objective="multi:softmax")) # 'multi:softprob'?



classifier = OneVsRestClassifier(XGBClassifier(num_class=len(classes),

                                base_score=0.5, colsample_bytree=0.5, 

                               gamma=0.017,learning_rate=0.15, max_delta_step=0, 

                               max_depth=mx_depth, min_child_weight=3, n_estimators=3000, 

                               nthread=-1, objective='multi:softmax', seed=0, 

                               silent=1, subsample=0.8, eta=0.3)) # class_weight='auto'



# multi:softprob' target to predict probabilities (2d array)

    

classifier.fit(X_train_, Y_train_)

# Show spent time

print("Time passed: ", "{0:.3f}".format(time.time() - t), "sec")
print (one_of_K_encoder.classes_)
# Predict labels using the trained model

# Y_predict = classifier.predict_proba(X_test_) # probs

# Predict labels using the trained model

Y_predict = classifier.predict(X_test_) # classes 





# Show some predicted values

print("Samples of predicted labels (in 1-of-K coding scheme):\n", Y_predict[1:8])

print("\nSamples of corresponding predicted labels:\n", one_of_K_encoder.inverse_transform(Y_predict[1:8]))
from sklearn.metrics import f1_score # For measuring F1 score metrics

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix





# Show global F1 score & on-label F1 score

print("Overall F1 score: ", f1_score(Y_test_, Y_predict, average='micro')) 

print("F1 score of each label : ", f1_score(Y_test_, Y_predict, average=None))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix



# labels = np.array([0, 1,2])

labels= one_of_K_encoder.classes_



def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10)):

    """

    Generate matrix plot of confusion matrix with pretty annotations.

    The plot image is saved to disk.

    args: 

      y_true:    true label of the data, with shape (nsamples,)

      y_pred:    prediction of the data, with shape (nsamples,)

      filename:  filename of figure file to save

      labels:    string array, name the order of class labels in the confusion matrix.

                 use `clf.classes_` if using scikit-learn models.

                 with shape (nclass,).

      ymap:      dict: any -> string, length == nclass.

                 if not None, map the labels & ys to more understandable strings.

                 Caution: original y_true, y_pred and labels must align.

      figsize:   the size of the figure plotted.

    """

    if ymap is not None:

        y_pred = [ymap[yi] for yi in y_pred]

        y_true = [ymap[yi] for yi in y_true]

        labels = [ymap[yi] for yi in labels]

    cm = confusion_matrix(y_true, y_pred, labels=None)

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=labels, columns=labels)

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=annot, fmt='', ax=ax)

#     plt.savefig(filename)
cm_analysis(Y_test_.argmax(axis=1), Y_predict.argmax(axis=1),labels)