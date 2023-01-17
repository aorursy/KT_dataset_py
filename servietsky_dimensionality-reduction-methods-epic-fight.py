# !jupyter labextension install jupyterlab-plotly



import numpy as np 

import pandas as pd 



from tensorflow.keras import Sequential



from sklearn.model_selection import cross_val_score

from collections import defaultdict

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold

from tqdm.notebook import tqdm 

from keras.utils.np_utils import to_categorical

import seaborn as sns

from sklearn.linear_model import LogisticRegression



import plotly.express as px

import matplotlib.pyplot as plt

import plotly.offline as pyo



from sklearn.manifold import LocallyLinearEmbedding

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA

from sklearn.decomposition import KernelPCA

from sklearn.decomposition import SparsePCA

from sklearn.decomposition import NMF

from sklearn.manifold import Isomap

from sklearn.manifold import TSNE

from sklearn.decomposition import TruncatedSVD

from sklearn.random_projection import GaussianRandomProjection

from sklearn.decomposition import FastICA

from sklearn.decomposition import MiniBatchDictionaryLearning

from sklearn.random_projection import SparseRandomProjection



from sklearn.datasets import make_classification

from xgboost import XGBClassifier

import lightgbm as lgb



import gc



pyo.init_notebook_mode()



train = pd.read_csv('../input/digit-recognizer/train.csv')



train = train.sample(10000).reset_index(drop=True) #i pick sample of 10k digit to increase speed



train.loc[:,'pixel0':] = train.loc[:,'pixel0':]/255



X_ = train.loc[:,'pixel0':]

y = train['label']



components = [784 ,int(785/2) ,int(785/4) ,int(785/8), int(785/16), int(785/32), int(785/64), int(785/128), int(785/256), 2]

# components = components[::-1]

batch_size = 501

epochs = 17

neurons = 958

optimizer = 'Adam'

random_state=42
def digit_show (df_train, number, name) :

    plt.figure(figsize=(20, 15))

    j = 1

    for i in range(10) :

        if number == 784 :

            plt.subplot(1,10,j)

            plt.gca().set_title('Image Reconstruction from Compressed Representation', fontsize=16)

        else :

            plt.subplot(1,10,j)

        j +=1

        plt.imshow(df_train[df_train['label'] == i].head(1).drop(labels = ["label"],axis = 1).values.reshape(28, 28), cmap='gray', interpolation='none')

        if number == 784 :

            plt.title("Original : {}".format(i))

        else :

            plt.title("{} {} Digit: {}".format(name, number, i))

    plt.tight_layout()

# this funcion take Algorithme in entry, use it for dimention reduction, train NN with results, plus 2D and 3D result, and accuracy performance.



def dimensionality_reduction_octagone(alg, name) :

    dim2 = []

    dim3 = []

    result = []

    names = []

    results = defaultdict(list)

    

    for i in tqdm(components) :

        if i == 784 :

            X= X_.values

        else :

            if name == 'KernelPCA' :

                alg_ = alg(n_components=i, fit_inverse_transform = True).fit(X_)

                X = alg_.transform(X_)

            else :

                alg_ = alg(n_components=i).fit(X_)

                X = alg_.transform(X_)

            

        if i == 2 :

            dim2 = X

        elif i ==3 :

            dim3 = X

            

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

        cvscores = []

        for train, test in kfold.split(X, y):

            model = Sequential()

            model.add(Dense(neurons, input_dim=i, activation='relu'))

            model.add(Dropout(0.2))

            model.add(Dense(neurons, activation='relu'))

            model.add(Dropout(0.2))

            model.add(Dense(10, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            model.fit(X[train], to_categorical(y[train],num_classes = 10), epochs=epochs, batch_size=batch_size, verbose = 0)

            scores = model.evaluate(X[test], to_categorical(y[test],num_classes = 10), verbose=0)

            cvscores.append(scores[1] * 100)

        results[name + ' ' + str(i)].append(cvscores)

        

        if name not in ('Isomap', 'GaussianRandomProjection', 'MiniBatchDictionaryLearning', 'SparseRandomProjection') :

            if i == 784 :

                digit = pd.merge(pd.DataFrame(X), pd.DataFrame(y), left_index=True, right_index=True)

                digit_show(digit, i, name)

            else :

                digit = pd.merge(pd.DataFrame(alg_.inverse_transform(X)), pd.DataFrame(y), left_index=True, right_index=True)

                digit_show(digit, i, name)

            

    

    plt.figure(figsize=(30,10))

    

    for key, value in results.items():

        if key == name + ' ' + '784' :

            names.append('Original')

        else :

            names.append(key)

            

        result.append(value)



    plt.xticks(rotation=45)

    ax = sns.boxplot(x=names, y= result)

    ax.set(xlabel= name + ' Components (Dimmentions)', ylabel='Accuracy %')

    ax.set_title('Accuracy Progression from '+ name + ' Compressed Representation')

   



        

    final_2D = pd.merge(pd.DataFrame(dim2), pd.DataFrame(y), left_index=True, right_index=True)

    final_2D.columns = ['X','Y','Label']

    final_2D.Label = final_2D.Label.astype('str')



    fig1 = px.scatter(final_2D, x='X', y='Y', color="Label", title= name + " 2 Components")

#     fig1.show()

    

    final_3D = pd.merge(pd.DataFrame(dim3), pd.DataFrame(y), left_index=True, right_index=True)

    final_3D.columns = ['X','Y','Z','Label']

    final_3D.Label = final_3D.Label.astype('str')



    fig2 = px.scatter_3d(final_3D, x='X', y='Y', z= 'Z', color="Label", size_max=0.2, title= name + " 3 Components")

#     fig2.update_traces(marker=dict(size=2,

#                               line=dict(width=0,

#                                         color='DarkSlateGrey')),

#                   selector=dict(mode='markers'))

#     fig2.show()

    

    

    fig1.show()

    fig2.show()

    gc.collect()
dimensionality_reduction_octagone(PCA, 'PCA')
dimensionality_reduction_octagone(IncrementalPCA, 'IncrementalPCA')
dimensionality_reduction_octagone(NMF, 'NMF')
dimensionality_reduction_octagone(KernelPCA, 'KernelPCA')
dimensionality_reduction_octagone(Isomap, 'Isomap')
dimensionality_reduction_octagone(TruncatedSVD, 'TruncatedSVD')
dimensionality_reduction_octagone(GaussianRandomProjection, 'GaussianRandomProjection')
dimensionality_reduction_octagone(FastICA, 'FastICA')
dimensionality_reduction_octagone(MiniBatchDictionaryLearning, 'MiniBatchDictionaryLearning')
dimensionality_reduction_octagone(SparseRandomProjection, 'SparseRandomProjection')