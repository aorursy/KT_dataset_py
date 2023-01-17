# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#      for filename in filenames:
#          print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.applications import VGG16 ,ResNet50
from sklearn.preprocessing import LabelEncoder
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.layers import Input
import seaborn as sns
import pandas as pd
import itertools
import shutil
import pickle
import random
import time
import csv
import os
!pip install imutils
from imutils import paths
base_path="/kaggle/input"
work_path="/kaggle/working"
print("[ALERT]...loading the VGG16 model")
modelVgg=VGG16(weights="imagenet",include_top=False,input_tensor=Input(shape=(128, 128, 3)))
le=None
modelVgg.summary()
datasets=["training"]
batch_size=32
# os.remove("/kaggle/working/Vgg16.csv")
for train_images in (datasets):
    #we have to grab the images from the training path to extract the features
    p=os.path.sep.join([base_path,train_images])
    imagePaths=list(paths.list_images(p))
    #randomly shuffule the images to make sure all varianta are present randomly 
    random.shuffle(imagePaths)
    labels=[p.split(os.path.sep)[-2] for p in imagePaths]
    if le is None:
        le=LabelEncoder()
        le.fit(labels)
    print(set(labels))
    csvPathVgg=os.path.sep.join([work_path,"{}.csv".format("Vgg16")])
    csvVgg=open(csvPathVgg, "w")

    for (b, i) in enumerate(range(0, len(imagePaths), batch_size)):
        # extract the batch of images and labels, then initialize the
		# list of actual images that will be passed through the network
		# for feature extraction
        print("[INFO] processing batch {}/{}".format(b + 1,int(np.ceil(len(imagePaths) / float(batch_size)))))
        batchPaths = imagePaths[i:i + batch_size]
        batchLabels = le.transform(labels[i:i + batch_size])
        batchImages = []
        for imagePath in batchPaths:
            # the image is resized to 128*128 pixels, since the VGG16 model is loaded with same image resolution
            image = load_img(imagePath, target_size=(128,128))
            image = img_to_array(image)

            # preprocess the image by  expanding the dimensions and preprocessing using the imagenet utils
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)

            # add the image to the batch
            batchImages.append(image)

        batchImages = np.vstack(batchImages)
        featuresVgg = modelVgg.predict(batchImages, batch_size=batch_size)
        #reshaping the feature array to the output layer shape (4*4*512)
        featuresVgg = featuresVgg.reshape((featuresVgg.shape[0], 4 * 4 * 512))
        # loop over the class labels and extracted features
        for (label, vec) in zip(batchLabels, featuresVgg):
            # construct a row with "," to make sure while writing csv, each value(feature) goes to a sepearte column
            vec = ",".join([str(v) for v in vec])
            csvVgg.write("{},{}\n".format(label, vec))
print("extraction completed")
csvVgg=os.path.sep.join([work_path,"Vgg16.csv"])
def getcsvcount(csvfile):
    with open(csvVgg, 'r') as csv:
        first_line = csv.readline()
    ncol = first_line.count(',') + 1
    return ncol
csvVggcount=getcsvcount(csvVgg)
colsVgg=['feature_'+str(i) for i in range(csvVggcount-1)]
def load_data_split(splitPath):
    #data and label varialbles
    data=[]
    labels=[]
    for row in open(splitPath):
        row=row.strip().split(",")
        label=row[0]
        features=np.array(row[1:],dtype="float")

        data.append(features)
        labels.append(label)
    data=np.array(data)
    labels=np.array(labels)
    return(data,labels)
dataVgg,labelsVgg=load_data_split(csvVgg)
df_Vgg=pd.DataFrame(dataVgg,columns=colsVgg)
df_Vgg['labels'] = labelsVgg
print('Size of the dataframe: {}'.format(df_Vgg.shape))
# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df_Vgg.shape[0])
N=8000
df_subset = df_Vgg.loc[rndperm[:N],:].copy()
data_subset = df_subset[colsVgg].values
y=df_subset['labels'].values
#tSNE dimensionality reduction
time_start=time.time()
tsne=TSNE(n_components= 2,verbose=1,perplexity=40,
          n_iter=600)
tsne_results=tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
df_subset['tsne-one'] = tsne_results[:,0]
df_subset['tsne-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-one", y="tsne-two",
    hue=y,
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.9
)