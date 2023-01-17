!pip install opencv-python==3.4.2.17
!pip install opencv-contrib-python==3.4.2.17
!ls ../usr/lib/functions_py
!ls
%matplotlib inline



#%load_ext autoreload  # Autoreload has a bug : when you modify function in source code and run again, python kernel hangs :(

#%autoreload 2



import datetime as dt



import sys, importlib



from functions_py import * # MODIFIED for kaggle (replaced by functions_py instead of functions)

importlib.reload(sys.modules['functions_py']) # MODIFIED for kaggle



import pandas as pd



pd.set_option('display.max_rows', 200)

pd.set_option('display.max_columns', 200)



import datetime as dt



import os

import zipfile

import urllib



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np   

  

import plotly as py

import plotly.graph_objects as go

import ipywidgets as widgets



import qgrid



import glob



from pandas.plotting import scatter_matrix



from sklearn.model_selection import StratifiedShuffleSplit



from sklearn.model_selection import GridSearchCV



from sklearn.manifold import TSNE

from sklearn.manifold import LocallyLinearEmbedding

from sklearn.manifold import Isomap



from sklearn.cluster import KMeans

from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import silhouette_score

from sklearn.metrics import pairwise_distances

from sklearn.cluster import AgglomerativeClustering

from scipy.stats import entropy



from sklearn.feature_selection import RFE



from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.multioutput import MultiOutputClassifier

from sklearn.linear_model import Perceptron

from sklearn import tree



from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



from sklearn.metrics import classification_report



#from yellowbrick.classifier import ROCAUC

from sklearn.metrics import roc_auc_score



import nltk

import codecs



from sklearn.decomposition import LatentDirichletAllocation



#from nltk.corpus.reader.api import CorpusReader

#from nltk.corpus.reader.api import CategorizedCorpusReader



from nltk import pos_tag, sent_tokenize, wordpunct_tokenize



#import pandas_profiling



from bs4 import BeautifulSoup



DATA_PATH = os.path.join("../input", "stanford-dogs-dataset", "images") # Modified for kaggle

DATA_PATH = os.path.join(DATA_PATH, "Images")



#DATA_PATH_FILE_INPUT = os.path.join(DATA_PATH, "QueryResults_20190101-20200620.csv")

#DATA_PATH_FILE_INPUT = os.path.join(DATA_PATH, "QueryResults 20200301-20200620_1.csv")



DATA_PATH_FILE = os.path.join(DATA_PATH, "*.csv")

ALL_FILES_LIST = glob.glob(DATA_PATH_FILE)



ALL_FEATURES = []



plt.rcParams["figure.figsize"] = [16,9] # Taille par défaut des figures de matplotlib



import seaborn as sns

from seaborn import boxplot

sns.set()



#import common_functions



####### Paramètres pour sauver et restaurer les modèles :

import pickle

####### Paramètres à changer par l'utilisateur selon son besoin :



'''

RECOMPUTE_GRIDSEARCH = True  # CAUTION : computation is several hours long

SAVE_GRID_RESULTS = False # If True : grid results object will be saved to pickle files that have GRIDSEARCH_FILE_PREFIX

LOAD_GRID_RESULTS = False # If True : grid results object will be loaded from pickle files that have GRIDSEARCH_FILE_PREFIX

                          # Grid search results are loaded with full samples (SAMPLED_DATA must be False)

'''





RECOMPUTE_GRIDSEARCH = False  # CAUTION : computation is several hours long

SAVE_GRID_RESULTS = False # If True : grid results object will be saved to pickle files that have GRIDSEARCH_FILE_PREFIX

LOAD_GRID_RESULTS = True # If True : grid results object will be loaded from pickle files that have GRIDSEARCH_FILE_PREFIX



#GRIDSEARCH_CSV_FILE = 'grid_search_results.csv'



GRIDSEARCH_FILE_PREFIX = 'grid_search_results_'



# Set this to load (or train again / save) Clustering model to disk

SAVE_CLUSTERING_MODEL = True

CLUSTERING_FILE_MODEL_PREFIX = 'clustering_model'



SAVE_DESCRIPTORS = True

DESCRIPTORS_FILE_PREFIX = 'descriptors_file'





SAVE_BESTGRIDSEARCH_MODEL = False

LOAD_BESTGRIDSEARCH_MODEL = True

BESTGRIDSEARCH_FILE_MODEL_PREFIX = 'bestgridsearch_model_'



EXECUTE_INTERMEDIATE_MODELS = True # If True: every intermediate model (which results are manually analyzed in the notebook) will be executed





# Necessary for predictors used in the notebook :

from sklearn.linear_model import LinearRegression

from sklearn.compose import TransformedTargetRegressor

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



from sklearn.preprocessing import PolynomialFeatures



### For progress bar :

#from tqdm import tqdm_notebook as tqdm  #Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`

from tqdm.notebook import tqdm



# Statsmodel : 

import statsmodels.formula.api as smf



import statsmodels.api as sm

from scipy import stats



from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



from nltk.cluster import KMeansClusterer # NLTK algorithm will be useful for cosine distance



SAVE_API_MODEL = True # If True : API model ill be saved

API_MODEL_PICKLE_FILE = 'API_model_PJ7.pickle'





LEARNING_CURVE_STEP_SIZE = 100

from PIL import Image

from io import BytesIO
import cv2



sift = cv2.xfeatures2d.SIFT_create()

surf = cv2.xfeatures2d.SURF_create()
!ls ../
img = Image.open("../input/stanford-dogs-dataset/images/Images/n02099601-golden_retriever/n02099601_1010.jpg") 
display(img)
np_img = np.array(img)
# Pour le normaliser : argument density=True dans plt.hist

# Pour avoir l'histogramme cumulé : argument cumulative=True



n, bins, patches = plt.hist(np_img.flatten(), bins=range(256))

plt.show()
# Pour le normaliser : argument density=True dans plt.hist

# Pour avoir l'histogramme cumulé : argument cumulative=True



n, bins, patches = plt.hist(np_img.flatten(), bins=range(256), edgecolor='blue')

plt.show()
# Pour le normaliser : argument density=True dans plt.hist

# Pour avoir l'histogramme cumulé : argument cumulative=True



n, bins, patches = plt.hist(np_img.flatten(), bins=range(256), histtype='stepfilled')

plt.show()
PATH_TESTIMAGE = DATA_PATH + "/n02111889-Samoyed/" + "n02111889_1363.jpg"

img = Image.open(PATH_TESTIMAGE) 
'n02108422-bull_mastiff'.split('-')[1]
display(img)
img.size
img.mode
img.getpixel((20, 100))
np_image = np.array(img)
np_image.shape
np.array([1,2,3])
np_img = np.array(img)
np_img[:, :, 0]
# Pour le normaliser : argument density=True dans plt.hist

# Pour avoir l'histogramme cumulé : argument cumulative=True



n, bins, patches = plt.hist(np_img.flatten(), bins=range(256))

plt.show()
# Pour le normaliser : argument density=True dans plt.hist

# Pour avoir l'histogramme cumulé : argument cumulative=True



n, bins, patches = plt.hist(np_img[:, :, 0].flatten(), bins=range(256))

plt.show()
# Pour le normaliser : argument density=True dans plt.hist

# Pour avoir l'histogramme cumulé : argument cumulative=True



n, bins, patches = plt.hist(np_img[:, :, 1].flatten(), bins=range(256))

plt.show()
# Pour le normaliser : argument density=True dans plt.hist

# Pour avoir l'histogramme cumulé : argument cumulative=True



n, bins, patches = plt.hist(np_img[:, :, 2].flatten(), bins=range(256))

plt.show()
imgcv = cv2.imread(PATH_TESTIMAGE)
imgcv_gray = cv2.cvtColor(imgcv,cv2.COLOR_BGR2GRAY) # Gray scaling
imgcv_gray.shape
Image.fromarray(imgcv_gray)
kp = sift.detect(imgcv_gray,None)



imgcv_keypoints = cv2.drawKeypoints(imgcv_gray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=None)



cv2.imwrite('sift_keypoints.jpg',imgcv_keypoints)

Image.fromarray(imgcv_keypoints)
kp, des = sift.detectAndCompute(imgcv_gray,None)
len(kp)
des.shape
kp_sorted = sorted(kp, key=lambda k : k.response, reverse=True)
imgcv_keypoints_best = cv2.drawKeypoints(imgcv_gray, kp_sorted[0:300], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=None)



cv2.imwrite('sift_keypoints_best.jpg',imgcv_keypoints_best)
Image.fromarray(imgcv_keypoints_best)
kp_sorted[0:20]
kp[548]
kp_sorted_indices = [i[0] for i in sorted(enumerate(kp), key=lambda x:x[1].response, reverse=True)]
kp_sorted_indices
des[kp_sorted_indices[0:20], :].shape
flann_params = dict(algorithm = 1, trees = 5)

matcher = cv2.FlannBasedMatcher(flann_params, {}) 

bow_extract = cv2.BOWImgDescriptorExtractor( sift , matcher )

#bow_extract.setVocabulary( vocab ) # the 64x20 dictionary, you made before # <= selon la doc:  Each row of the vocabulary is a visual word (cluster center). 
#bowsig = bow_extract.compute(imgcv_gray, kp)
#from functions_py import *

#importlib.reload(sys.modules['functions_py'])
filename_images = []

np_images = []

labels = []



deslist = []



NB_CLASSES = 20

NB_DOGS_PER_RACE = 100 # 100. Note: we have 120 dog races in the dataset

NB_KEYPOINTS_PER_DOG = 200 # 200

NB_CLUSTERS = 1000 # 100

i = 0



df_des = pd.DataFrame(np.empty((0, 1+128), np.uint8), columns=['picnum'] +  [colname for colname in range(0,128)])  # We add 1 column to store the image number
df_des
%%time



if (SAVE_CLUSTERING_MODEL == True):  # If False, we'll load data later via pickle

    cnt_files = sum([len(files) for r, d, files in os.walk(DATA_PATH)])

    progbar = tqdm(range(cnt_files))



    cnt_classes = 0

    pic_number = 0

    for root, dirs, files in os.walk(DATA_PATH):

        path = root.split(os.sep)



        cnt_classes += 1

                

        if (cnt_classes <= NB_CLASSES + 1): # NB_CLASSES + 1 because first path is Image directory root, does not count as a class

            i = 0    

            for file in files:

                if (i > NB_DOGS_PER_RACE - 1):

                    break



                #print(f'pic number: {pic_number}, dog number in current race: {i}')

                #print(f'path: {path}')



                #Uncomment those 2 lignes to filter specific races (will reduce the number of classes)

                #image_label = os.path.join(root, file).split('/')[5].split('-')[1]

                #if ((image_label == 'Pomeranian') or (image_label == 'Samoyed')):



                # Append filename to global list

                filename_images.append(os.path.join(root, file))



                img = cv2.imread(os.path.join(root, file))



                kp, des = sift.detectAndCompute(img, None)



                kp_sorted_indices = [i[0] for i in sorted(enumerate(kp), key=lambda x:x[1].response, reverse=True)] # Sort by pixel importance descending

                des = des[kp_sorted_indices[0:NB_KEYPOINTS_PER_DOG], :]



                df_picnum = pd.DataFrame(np.full((des.shape[0], 1), pic_number, dtype=np.uint16), columns=['picnum']) # Add 1 column to store picture number

                #print(des.shape)

                #print(df_picnum.shape)



                df_des_1pic = pd.concat([df_picnum, pd.DataFrame(des.astype(np.uint8))], axis=1)

                #print(df_des_1pic.shape)

                #print(df_des_1pic.columns)



                #print('df_des.shape avant concat ')

                #print(df_des.shape)



                df_des = pd.concat([df_des, df_des_1pic], axis=0)



                #print('df_des.shape apres concat ')

                #print(df_des.shape)

                #print('!')



                progbar.update(1)



                i += 1

                pic_number += 1

          
df_des
#df_des.memory_usage()
df_des.info()
clusterer = Clusterer(n_clusters=NB_CLUSTERS)  # Uncomment this for normal clustering instead of mini batch

df_des.iloc[:, 1:].shape
#BATCH_SIZE = 1000
#minibatch_indexes = minibatch_generate_indexes(df_des.iloc[:, 1:], BATCH_SIZE + 1)  # Choose second argument (batch size) = nb instances / 100 ?
%time

 # Uncomment this for normal clustering instead of mini batch





if (SAVE_CLUSTERING_MODEL == True):

    clusterer.fit(df_des.iloc[:, 1:]) # We do this iloc in order not to include picnum column





# Uncomment this for mini batch kmeans

'''

clusterer = MiniBatchKMeans(n_clusters=NB_CLUSTERS,

                          random_state=42,

                          batch_size=BATCH_SIZE,

                          max_iter=10)

if (SAVE_CLUSTERING_MODEL == True):

    for (left_index, right_index) in minibatch_indexes:

        clusterer.fit(df_des.iloc[left_index:right_index, 1:])

'''     

    
#import functions_py

#importlib.reload(sys.modules['functions_py'])
 # Uncomment this for normal clustering instead of mini batch



if (SAVE_CLUSTERING_MODEL == True):

    with open(CLUSTERING_FILE_MODEL_PREFIX + 'model1' + '.pickle', 'wb') as f:

        pickle.dump(clusterer.clusterer, f, pickle.HIGHEST_PROTOCOL)

        clusterer_clusterer = clusterer.clusterer # We have to do this for after "pickle load" access of this variable, because pickle dump of clusterer object directly does not work: we had to pickle inner clusterer.clusterer object inside

        

else:

    with open(CLUSTERING_FILE_MODEL_PREFIX + 'model1' + '.pickle', 'rb') as f:

        clusterer_clusterer = pickle.load(f)





 # Uncomment this for mini batch kmeans clustering

'''

if (SAVE_CLUSTERING_MODEL == True):

    with open(CLUSTERING_FILE_MODEL_PREFIX + 'model1' + '.pickle', 'wb') as f:

        pickle.dump(clusterer, f, pickle.HIGHEST_PROTOCOL)

        clusterer_clusterer = clusterer # We have to do this for after "pickle load" access of this variable, because pickle dump of clusterer object directly does not work: we had to pickle inner clusterer.clusterer object inside

        

else:

    with open(CLUSTERING_FILE_MODEL_PREFIX + 'model1' + '.pickle', 'rb') as f:

        clusterer_clusterer = pickle.load(f)

'''
!ls -l
if (SAVE_DESCRIPTORS == True):

    with open(DESCRIPTORS_FILE_PREFIX + 'model1' + '.pickle', 'wb') as f:

        pickle.dump(df_des, f, pickle.HIGHEST_PROTOCOL)    

        

else:

    with open(DESCRIPTORS_FILE_PREFIX + 'model1' + '.pickle', 'rb') as f:

        df_des = pickle.load(f)
if (SAVE_CLUSTERING_MODEL == True):

    with open('filenames' + 'model1' + '.pickle', 'wb') as f:

        pickle.dump(filename_images, f, pickle.HIGHEST_PROTOCOL)

        

else:

    with open('filenames' + 'model1' + '.pickle', 'rb') as f:

        filename_images = pickle.load(f)
clusterer_clusterer.predict(df_des.iloc[0:6, 1:])
df_des
len(filename_images)
clusterer_clusterer.n_clusters
all_cluster_numbers = list(range(NB_CLUSTERS))
zero_matrix = np.zeros((len(df_des), NB_CLUSTERS), np.uint8)
zero_matrix.nbytes
df_des['picnum']
#dummies = pd.concat([df_des['picnum'], pd.DataFrame(zero_matrix, [colname for colname in range(NB_CLUSTERS)])])
!ls