# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = np.array(pd.read_csv(r'../input/survey.csv',header=None))

labels = data[0,:]

data_no_header = data[1:,:]

for i in range(27):

    globals()[labels[i]] = data_no_header[:,i]

headings = np.array(['treatment ','Age ','Gender ','self_employed ','family_history ','work_interfere ','no_employees ','remote_work ','tech_company ','benefits ','care_options ','wellness_program ','seek_help ','anonymity ','mental_health_consequence ','leave ','phys_health_consequence ','coworkers ','supervisor ','mental_health_interview ','phys_health_interview ','mental_vs_physical ','obs_consequence ','Country ','state'])





lower = []

gender_fixed = []

for i in Gender:

    low = np.str.lower(i)

    lower.append(low)

    



gender_unique = list(set(lower))



for i in lower:

    if i in ['ostensibly male, unsure what that really means' , 'msle' , 'male' , 'm','maile' , 'male (cis)' , 'man' , 'male ' , 'guy (-ish) ^_^','mal' , 'cis male']:

        gender_fixed.append('m')

    elif i in ['femake' , 'femail' , 'female (cis)' ,  'cis female' , 'f' , 'female ' , 'cis-female/femme' , 'female' , 'woman']:

        gender_fixed.append('f')

    elif i in ['trans woman' , 'neuter' , 'non-binary' , 'all' ,  'androgyne' , 'male leaning androgynous' , 'fluid' , 'trans-female' ,'genderqueer' , 'queer' , 'queer/she/they']:

        gender_fixed.append('t')

    else:

        gender_fixed.append('o')

        

Gender = np.array(gender_fixed)



age_fixed = []

for i in Age:

    if int(i) <0:

        age_fixed.append(30)

    elif int(i)  >100:

        age_fixed.append(40)

    else:

        age_fixed.append(int(i) )



Age = np.array(age_fixed)





for i in range(1259):

    if str(self_employed[i]) == 'nan':

        np.put(self_employed,i,'No')



for i in range(1259):

    if str(work_interfere[i]) == 'nan':

        np.put(work_interfere,i, 'Sometimes')

        

        

def change_to_int(array):

    for i in range(1259):

        if array[i] in ['Yes','f',"Somewhat difficult","Rarely",'6-25']:

            np.put(array,i,'1')

        elif array[i] in ['No','m','Never','Very difficult','1-5']:

            np.put(array,i,'0')

        elif array[i] in ["Don't Know","Not sure","Maybe","t","Sometimes",'26-100','Some of them',"Don't know"]:

            np.put(array,i,'2')

        elif array[i] in ["Somewhat easy","o","Often",'100-500']:

            np.put(array,i,'3')

        elif array[i] in ["Very easy",'500-1000']:

            np.put(array,i,'4')

        elif array[i] in ['More than 1000']:

            np.put(array,i,'5')

#         else:

#             print('error in {}[{}]'.format('array',i))

    

for arr in headings[:-2]:

    change_to_int(globals()[arr[:-1]])

    

    

data_cleaned = np.transpose(np.array([treatment,Age,Gender\

                            ,self_employed,family_history,work_interfere,no_employees\

                            ,remote_work\

                            ,tech_company,benefits,care_options,wellness_program\

                            ,seek_help,anonymity,mental_health_consequence\

                            ,leave,phys_health_consequence,coworkers,supervisor\

                            ,mental_health_interview,phys_health_interview\

                            ,mental_vs_physical,obs_consequence,Country,state]))





data_cleaned
headings = np.array(['treatment ','Age ','Gender ','self_employed ','family_history ','work_interfere ','no_employees ','remote_work ','tech_company ','benefits ','care_options ','wellness_program ','seek_help ','anonymity ','mental_health_consequence ','leave ','phys_health_consequence ','coworkers ','supervisor ','mental_health_interview ','phys_health_interview ','mental_vs_physical ','obs_consequence ','Country ','state'])

# data1 = np.concatenate((headings,data_cleaned),axis=0)

h1 = headings.reshape(1,25)

data1 = np.concatenate((h1,data_cleaned),axis=0)



# data_csv = pd.DataFrame(data1)

# data_csv.iloc[:,:]
# gender_fixed1 = [],self_employed_fixed = [],family_history_fixed=[],work_interfere_fixed=[],no_employees_fixed = []

# remote_work = [] tech_company=[]

# data_csv





   



data = data_cleaned
X = data[:,1:-2]

y = data[:,0]

y = y.astype('float')

import matplotlib.pyplot as plt

from sklearn import linear_model, datasets,neighbors

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA, NMF

from sklearn.feature_selection import SelectKBest, chi2

import sklearn.feature_selection as sl
a,pval = chi2(X,y)



# MI = sl.mutual_info_regression(X,y)

# MI_dic = pd.DataFrame(np.array([headings,MI,pval]).T)
import seaborn as sns

sns.barplot(y =headings[1:-2],x = pval)

plt.show()

from time import time

from mpl_toolkits.mplot3d.axes3d import Axes3D

from sklearn import (manifold, datasets, decomposition, ensemble, lda,random_projection)

n_neighbors = 3

pval_0_1 = pval<0.01

yn = ['N','Y']

# X1 = data[:,1:-2]

y=data[:,0].astype('int')

# X = X1[:,pval_0_1]

X = data[:,[0,1,3,9,14,19,21,22]]

#%%

# 将降维后的数据可视化,2维

def plot_embedding_2d(X, title=None):

    #坐标缩放到[0,1]区间

    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)

    X = (X - x_min) / (x_max - x_min)

    yn = ['N','Y']

    #降维后的坐标为（X[i, 0], X[i, 1]），在该位置画出对应的digits

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)

    for i in range(X.shape[0]):

        ax.text(X[i, 0], X[i, 1],str(yn[y[i]]),

                 color=plt.cm.Set1((y[i]+1) / 10.),

                 fontdict={'weight': 'bold', 'size': 9})



    if title is not None:

        plt.title(title)

        

def plot_embedding_3d(X, title=None):

    #坐标缩放到[0,1]区间

    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)

    X = (X - x_min) / (x_max - x_min)



    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for i in range(X.shape[0]):

        ax.text(X[i, 0], X[i, 1], X[i,2],str(yn[y[i]]),

                 color=plt.cm.Set1((y[i]+1) / 10.),

                 fontdict={'weight': 'bold', 'size': 9})



    if title is not None:

        plt.title(title)

        

print("Computing random projection")

rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)

X_projected = rp.fit_transform(X)

plot_embedding_2d(X_projected, "Random Projection")



#%%

#PCA

print("Computing PCA projection")

t0 = time()

X_pca = decomposition.TruncatedSVD(n_components=3).fit_transform(X)

plot_embedding_2d(X_pca[:,0:2],"PCA 2D")

plot_embedding_3d(X_pca,"PCA 3D (time %.2fs)" %(time() - t0))

plt.show()
#Isomap

print("Computing Isomap embedding")

t0 = time()

X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)

print("Done.")

plot_embedding_2d(X_iso,"Isomap (time %.2fs)" %(time() - t0))



# MDS

print("Computing MDS embedding")

clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)

t0 = time()

X_mds = clf.fit_transform(X)

print("Done. Stress: %f" % clf.stress_)

plot_embedding_2d(X_mds,"MDS (time %.2fs)" %(time() - t0))



#%%

# Random Trees

print("Computing Totally Random Trees embedding")

hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,max_depth=5)

t0 = time()

X_transformed = hasher.fit_transform(X)

pca = decomposition.TruncatedSVD(n_components=2)

X_reduced = pca.fit_transform(X_transformed)



plot_embedding_2d(X_reduced,"Random Trees (time %.2fs)" %(time() - t0))



#%%

# Spectral

print("Computing Spectral embedding")

embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,eigen_solver="arpack")

t0 = time()

X_se = embedder.fit_transform(X)

plot_embedding_2d(X_se,"Spectral (time %.2fs)" %(time() - t0))



#%%

# t-SNE

print("Computing t-SNE embedding")

tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)

t0 = time()

X_tsne = tsne.fit_transform(X)

print(X_tsne.shape)

plot_embedding_2d(X_tsne[:,0:2],"t-SNE 2D")

plot_embedding_3d(X_tsne,"t-SNE 3D (time %.2fs)" %(time() - t0))



plt.show()
data1 = np.array(pd.read_csv(r'../input/survey.csv',header=None))
data