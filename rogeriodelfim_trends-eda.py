import pandas as pd 

import numpy as np  

import matplotlib

import matplotlib.pyplot as plt 

import seaborn as sns           

color = sns.color_palette()

import plotly.offline as py

py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()



import cufflinks as cf

cf.go_offline()



# Venn diagram

from matplotlib_venn import venn2

import re

import nltk

from nltk.probability import FreqDist

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import string

eng_stopwords = stopwords.words('english')

import gc



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD



import os

import nilearn as nl
path = '/kaggle/input/trends-assessment-prediction'

print(os.listdir(path))
df_loading           = pd.read_csv(path +'/loading.csv')

df_train             = pd.read_csv(path +'/train_scores.csv')

df_sample_submission = pd.read_csv(path +'/sample_submission.csv')
print('Tamanho de df_loading', df_loading.shape)

print('Tamanho de df_train', df_train.shape)

print('Tamanho de sample_submission', df_sample_submission.shape)

print('test tamanho:', len(df_sample_submission)/5)

df_train.head()
df_train.info()
df_sample_submission.head()
display(df_loading.head())

display(df_loading.describe())
display(df_train.head())

display(df_train.describe())
df_sample_submission.head()
targets = list(df_train.columns[1:])

targets
# checking missing data

total   = df_train.isnull().sum().sort_values(ascending = False)

percent = (df_train.isnull().sum() / train_data.isnull().count() * 100).sort_values(ascending = False)
df_missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

df_missing_train_data.head()
# checking missing data

total   = df_loading.isnull().sum().sort_values(ascending = False)

percent = (df_loading.isnull().sum() / df_loading.isnull().count()*100).sort_values(ascending = False)



df_missing_test  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

df_missing_test.head()
def distributionTarget(df, type):



    targets   = df.columns[1:]

    fig, axes = plt.subplots(6, 5, figsize=(18, 15))

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

   

    axes = axes.ravel()

    

    if type == 1:

        bins = np.linspace(-0.05, 0.05, 20) 

    else: 

        bins = np.linspace(0, 100, 20) 



    for i, col in enumerate(targets):

        ax = axes[i]

        sns.distplot(df[col], label = col, kde = False, bins = bins, ax = ax)



    plt.tight_layout()

    plt.show()

    plt.close()
distributionTarget(df_loading, 1)
distributionTarget(df_train, 2)
def load_subject(filename, mask_niimg):

    """

    Carrega os dados salvo no formato .mat com

         o sinalizador da versão 7.3. Retornar os dados 

         em niimg, usando uma máscara niimg como modelo

         para cabeçalhos nifti.

        

     Args:

         filename <str> o nome do arquivo .mat para os dados do 

         objeto mask_niimg niimg o objeto mask niimg usado para cabeçalhos nifti         

    """

    subject_data = None



    with h5py.File(subject_filename, 'r') as f:

        subject_data = f['SM_feature'][()]

        

    # É necessário reorientar os eixos, pois o h5py vira a ordem dos eixos

    

    subject_data  = np.moveaxis(subject_data, [0,1,2,3], [3,2,1,0])

    subject_niimg = nl.image.new_img_like(mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)

    

    return subject_niimg
path_fMRI_mat = '/kaggle/input/trends-assessment-prediction/fMRI_train/'
import h5py

import nilearn.plotting as nlplt
mask_filename    = '/kaggle/input/trends-assessment-prediction/fMRI_mask.nii'

subject_filename = '../input/trends-assessment-prediction/fMRI_train/10015.mat'

smri_filename    = 'ch2better.nii'

mask_niimg       = nl.image.load_img(mask_filename)
subject_niimg = load_subject(subject_filename, mask_niimg)
print("Image shape is %s" % (str(subject_niimg.shape)))

num_components = subject_niimg.shape[-1]

print("Detected {num_components} spatial maps".format(num_components=num_components))

!wget https://github.com/Chaogan-Yan/DPABI/raw/master/Templates/ch2better.nii
nlplt.plot_prob_atlas(subject_niimg, 

                      bg_img     = smri_filename, 

                      view_type  = 'filled_contours', 

                      draw_cross = False, 

                      title      = 'All %d spatial maps' % num_components, 

                      threshold  = 'auto')

grid_size = int(np.ceil(np.sqrt(num_components)))

fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*10, grid_size*10))

[axi.set_axis_off() for axi in axes.ravel()]







for i, cur_img in enumerate(nl.image.iter_img(subject_niimg)):

    col = i % grid_size

    if col == 0:

        row += 1

    nlplt.plot_stat_map(cur_img, bg_img=smri_filename, title="IC %d" % i, axes=axes[row, col], threshold=3, colorbar=False)