## Specific installs / import in online notebook
!pip install opencv-python==3.4.2.17
!pip install opencv-contrib-python==3.4.2.17
%matplotlib inline



#%load_ext autoreload  # Autoreload has a bug : when you modify function in source code and run again, python kernel hangs :(

#%autoreload 2



import datetime as dt



import sys, importlib



from functions_py import * # MODIFIED for kaggle (replaced by functions_py instead of functions)

importlib.reload(sys.modules['functions_py']) # MODIFIED for kaggle



from display_factorial import *

importlib.reload(sys.modules['display_factorial'])



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

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



from sklearn.metrics import classification_report



from sklearn.metrics import confusion_matrix



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



from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer
from PIL import Image

from io import BytesIO
# Those variables must be consisten with what first notebook has been ran with

NB_CLASSES = 20



NB_DOGS_PER_RACE = 100 # 100. Note: we have 120 dog races in the dataset

NB_KEYPOINTS_PER_DOG = 200 # 200

NB_CLUSTERS = 1000
!ls -l ../input/openclassrooms-pj7-modelisation/
!date
with open('../input/openclassrooms-pj7-modelisation/clustering_modelmodel1.pickle', 'rb') as f:

    clusterer_clusterer = pickle.load(f)
with open('../input/openclassrooms-pj7-modelisation/filenamesmodel1.pickle', 'rb') as f:

    filename_images = pickle.load(f)
with open('../input/openclassrooms-pj7-modelisation/descriptors_filemodel1.pickle', 'rb') as f:

    df_des = pickle.load(f)
df_des
df_des.info()
df_des[['picnum']].groupby('picnum').size().sort_values(ascending=True)[0:200]
df_des[['picnum']].groupby('picnum').size()[df_des[['picnum']].groupby('picnum').size() < 200]
#!ls ../input/stanford-dogs-dataset/images/Images/n02113023-Pembroke
display(Image.open('../input/stanford-dogs-dataset/images/Images/n02113023-Pembroke/n02113023_10636.jpg'))
filename_images[9]
display(Image.open(filename_images[9]))
display(Image.open(filename_images[125]))
df_des.groupby('picnum').count()
df_des.reset_index(drop=True, inplace=True)
clusterer_clusterer.predict(df_des.iloc[0:6, 1:])
cluster_numbers = clusterer_clusterer.predict(df_des.iloc[:, 1:])
len(cluster_numbers)
df_des_clusters = pd.DataFrame(cluster_numbers)
df_des_clusters
df_des_clusters_dummies = pd.get_dummies(df_des_clusters[0])
picnums = df_des.iloc[:, 0]
picnums
df_des_clusters_dummies
df_picnum_clusters = pd.concat([picnums, df_des_clusters_dummies], axis=1)
df_picnum_clusters
df_picnum_agg_clusters = df_picnum_clusters.groupby('picnum').sum()
df_picnum_agg_clusters
df_picnum_agg_clusters = (df_picnum_agg_clusters.T / df_picnum_agg_clusters.T.sum()).T # Normalizing frequencies
df_picnum_agg_clusters.shape
df_picnum_agg_clusters.sum(axis=1)
pca = PCA(n_components=6)

scaler = StandardScaler().fit(df_picnum_agg_clusters)

X_scaled = scaler.transform(df_picnum_agg_clusters)

X_reduced = pca.fit_transform(X_scaled)
image_labels = [filename_image.split('/')[5].split('-')[1] for filename_image in filename_images]
display_scree_plot(pca)
from sklearn.preprocessing import LabelEncoder





df_dogs_races = pd.DataFrame(image_labels)

encoder = LabelEncoder()

numerical_races = encoder.fit_transform(df_dogs_races)
set(numerical_races)
py.offline.init_notebook_mode(connected=True)





trace_1 = go.Scatter(x = X_reduced[:,0], y = X_reduced[:,1],

                    name = 'Dogs',

                    mode = 'markers',

                    marker=dict(color=numerical_races),

                    text = numerical_races,

                    )





layout = go.Layout(title = 'Image features in 2 dimensions',

                   hovermode = 'closest',

)



fig = go.Figure(data = [trace_1], layout = layout)





py.offline.plot(fig, filename='features_2dplot.html') 
display_circles(pca.components_, 6, pca, [(0,1),(2,3),(4,5)])
image_labels = [filename_image.split('/')[5].split('-')[1] for filename_image in filename_images]
len(image_labels)
df_train, df_test, df_train_labels, df_test_labels = train_test_split(df_picnum_agg_clusters, image_labels, test_size=0.1, random_state=42, shuffle = True, stratify = image_labels)
df_train
#Check distribution of labels on test set : good, they are stratified

pd.DataFrame(df_test_labels).groupby(0)[0].count()
model = DecisionTreeClassifier()



model.fit(df_train, df_train_labels)

df_predictions_train = model.predict(df_train)

df_predictions_test = model.predict(df_test)
df_predictions_train
df_train_labels
precision_score(df_train_labels, df_predictions_train, average='micro')
recall_score(df_train_labels, df_predictions_train, average='micro')
precision_score(df_test_labels, df_predictions_test, average='micro')
recall_score(df_test_labels, df_predictions_test, average='micro')
 #__init__(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)[source]
# Standard scale does not make a big difference with random forest

#scaler = StandardScaler()

#df_train = scaler.fit_transform(df_train)

#df_test = scaler.transform(df_test)
model = RandomForestClassifier(max_depth=100, max_features=20, max_leaf_nodes=50,

                       n_estimators=5000, random_state=42)



model.fit(df_train, df_train_labels)

df_predictions_train = model.predict(df_train)

df_predictions_test = model.predict(df_test)
precision_score(df_train_labels, df_predictions_train, average='micro')
recall_score(df_train_labels, df_predictions_train, average='micro')
precision_score(df_test_labels, df_predictions_test, average='micro')
# Before gridsearch : was 0.45
recall_score(df_test_labels, df_predictions_test, average='micro')
confusion_mx = confusion_matrix(df_test_labels, df_predictions_test, labels=list(set(df_test_labels)))
row_sums = confusion_mx.sum(axis=1, keepdims=True)

norm_confusion_mx = confusion_mx / row_sums
np.fill_diagonal(norm_confusion_mx, 0)



figure = plt.figure() 

axes = figure.add_subplot(111) 

caxes = axes.matshow(norm_confusion_mx)

figure.colorbar(caxes)



plt.xticks(rotation=90) 

plt.yticks(rotation=0) 



plt.xticks(np.arange(20), list(set(df_test_labels)))

plt.yticks(np.arange(20), list(set(df_test_labels)))

plt.legend()
len(df_predictions_test)
len(df_test_labels)
df_predictions_compare = pd.DataFrame({'actual': df_test_labels, 'pred': df_predictions_test})
df_predictions_compare['count'] = 1
df_predictions_compare
misclass_df = df_predictions_compare[df_predictions_compare['actual'] != df_predictions_compare['pred']].groupby(['actual', 'pred']).sum().sort_values(['count'], ascending=False).reset_index()

misclass_df['pair'] = misclass_df['actual'] + ' / ' + misclass_df['pred']

misclass_df = misclass_df[['pair', 'count']].take(range(50))

misclass_df.sort_values(['count']).plot.barh(figsize=(8, 10), x='pair')

plt.title('Top misclassed pairs')
# Print dogs from certain class



train_indexes = [i for i,x in enumerate(df_train_labels) if x == 'silky_terrier'] 

indexes = df_train.iloc[train_indexes, :].index  # Get original indexes before the train/test split (the ones that match with filenames list)





MAX_DISPLAY_IMAGES = 102

cnt_display_images = 0



for index in indexes:

    if (cnt_display_images < MAX_DISPLAY_IMAGES):

        display(Image.open(filename_images[index]))

    

        cnt_display_images += 1
print(f'We count 28 images on {cnt_display_images} total for Great Pyrenees where humans are present ,  and 20 for Chihuahua !        13 for bull_mastiff, 10 for miniature_pinscher ')
'''

param_grid = {

        'max_depth': [5, 10, 100, 1000],

        'max_features': [5, 10, 20, 50, 100], 

        'max_leaf_nodes': [5, 10, 20, 50, 100],

        'n_estimators': [10, 100, 1000]

        }

'''



'''

param_grid = {

        'max_depth': [100, 500],

        'max_features': [20], 

        'max_leaf_nodes': [50],

        'n_estimators': [1000, 2000, 5000]

        }

'''



param_grid = {

        'max_depth': [100],

        'max_features': [20], 

        'max_leaf_nodes': [50],

        'n_estimators': [5000, 10000, 20000]

        }
# Commented out because I've already run it and checked the results (see cells below)

'''

scorer = make_scorer(precision_score_micro, greater_is_better=True)



grid_search = GridSearchCV(model, param_grid, verbose=10, error_score=np.nan, scoring=scorer, cv=5, iid=False)

grid_search.fit(df_train, df_train_labels)

'''
#grid_search.best_estimator_
#grid_search.best_estimator_

'''

Best estimator was, with all dog classes :

RandomForestClassifier(max_depth=5, max_features=10, max_leaf_nodes=20,

                       random_state=42)

                       

And with 2 dog classes :

RandomForestClassifier(max_depth=10, max_features=20, max_leaf_nodes=50,

                       random_state=42)

                       

And with 20 dog classes + better cluster number (1000) :

RandomForestClassifier(max_depth=100, max_features=20, max_leaf_nodes=50,

                       n_estimators=1000, random_state=42)



=> We try more estimators, and we get best :

RandomForestClassifier(max_depth=100, max_features=20, max_leaf_nodes=50,

                       n_estimators=5000, random_state=42)

'''
'''

df_grid_search_results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["mean_test_score"])],axis=1)

df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["std_test_score"], columns=["std_test_score"])],axis=1)

'''
'''

df_predictions_train = grid_search.predict(df_train)

df_predictions_test = grid_search.predict(df_test)

'''
model = KNeighborsClassifier(n_neighbors=10)



model.fit(df_train, df_train_labels)

df_predictions_train = model.predict(df_train)

df_predictions_test = model.predict(df_test)
precision_score(df_train_labels, df_predictions_train, average='micro')
recall_score(df_train_labels, df_predictions_train, average='micro')
precision_score(df_test_labels, df_predictions_test, average='micro')
recall_score(df_test_labels, df_predictions_test, average='micro')
df_picnum_agg_clusters.mean()
df_picnum_agg_clusters.mean().mean()
df_picnum_agg_clusters.std().mean()
df_picnum_agg_clusters.max().max()
df_picnum_agg_clusters.max().min()
#randomNums = np.random.normal(loc=2, scale=2.18, size=(12000, NB_CLUSTERS))

randomNums = np.random.normal(loc=2, scale=2.18, size=(NB_DOGS_PER_RACE * NB_CLASSES, NB_CLUSTERS))

randomInts = np.round(randomNums)
randomInts.min()
#df_random = pd.DataFrame(np.random.randint(0,100,size=(12000, NB_CLUSTERS)))

df_random = pd.DataFrame(randomInts)
df_train, df_test, df_train_labels, df_test_labels = train_test_split(df_random, image_labels, test_size=0.1, random_state=42, shuffle = True, stratify = image_labels)
#model = RandomForestClassifier(random_state=42, max_depth=15, max_features=20, max_leaf_nodes=100, n_estimators=1000)

model = RandomForestClassifier(random_state=42, max_depth=5, max_features=10, max_leaf_nodes=20)



model.fit(df_train, df_train_labels)

df_predictions_train = model.predict(df_train)

df_predictions_test = model.predict(df_test)
precision_score(df_train_labels, df_predictions_train, average='micro')
recall_score(df_train_labels, df_predictions_train, average='micro')
precision_score(df_test_labels, df_predictions_test, average='micro')
recall_score(df_test_labels, df_predictions_test, average='micro')
import cv2

sift = cv2.xfeatures2d.SIFT_create()



def visualize_keypoints(PATH_CLASS, nb_pics):

    for root, dirs, files in os.walk(PATH_CLASS):

        path = root.split(os.sep)



        #cnt_classes += 1

                



        i = 0    

        for file in files:

            if (i > nb_pics - 1):

                break

                

            img = cv2.imread(os.path.join(root, file))



            kp, des = sift.detectAndCompute(img, None)



            kp_sorted_indices = [i[0] for i in sorted(enumerate(kp), key=lambda x:x[1].response, reverse=True)] # Sort by pixel importance descending

            des = des[kp_sorted_indices[0:NB_KEYPOINTS_PER_DOG], :]

            #kp = kp[kp_sorted_indices[0:NB_KEYPOINTS_PER_DOG], :]



            imgcv_keypoints = cv2.drawKeypoints(img, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=None)

            

            

            write_path = os.path.join("/kaggle", "working/") + file + '_kp.jpg'

            print(write_path)

            cv2.imwrite(write_path,imgcv_keypoints)



            

            i += 1

            #pic_number += 1    

    

    
PATH_CLASS_1 = '../input/stanford-dogs-dataset/images/Images/n02085620-Chihuahua'

PATH_CLASS_2 = '../input/stanford-dogs-dataset/images/Images/n02111500-Great_Pyrenees'
visualize_keypoints(PATH_CLASS_1, 10)

visualize_keypoints(PATH_CLASS_2, 10)