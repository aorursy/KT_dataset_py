# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from  matplotlib import  pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

def get_pca(features):

    pca = PCA(n_components=3)

    transformed = pca.fit_transform(features)

    scaler = MinMaxScaler()

    scaler.fit(transformed)

    return scaler.transform(transformed)
def pca_visualization(dim_red_features):

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    us_accent = ax.scatter(dim_red_features[:, 0], dim_red_features[:, 1], zs=dim_red_features[:, 2], zdir='z',

                           c="r", label="YT")

    # ax.scatter(yt_dim_red_features[:,0],yt_dim_red_features[:,1],zs=yt_dim_red_features[:,2],zdir = 'z',c="g")



    ax.set_xlabel('X Label')

    ax.set_ylabel('Y Label')

    ax.set_zlabel('Z Label')

    ax.legend()

    ax.set_title('64_features youtube videos')

    # plt.scatter(us_dim_red_features[:,0],us_dim_red_features[:,1])

    plt.show()
raw_features_df = pd.read_csv('../input/Speaker_Recognition.csv')

features_df = raw_features_df.iloc[:,9:21].as_matrix()
dim_red_features =  get_pca(features_df)
pca_visualization(dim_red_features)