import numpy as np

import pandas as pd

import os

import sqlite3

import random

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVR

from numpy.random import choice

import warnings

warnings.filterwarnings('ignore')

sns.set(style="whitegrid")

class ImpNaNs:

    def __init__(self,path,tb):

        self.path = path

        self.tb = tb

    def Read_data(self):

    

        """REDING THE DATA FROM SQLITE"""

        

        self.conn = sqlite3.connect(self.path)

        self.q = f'SELECT * FROM {self.tb}'

        self.df = pd.read_sql(self.q,self.conn)

        return self.df

    

    """MAKING MISSING VALUES FOR THE GIVEN COLUMN AND % OF NaNs"""

    

    def Random_nan(self,mcol,r=0.5):

        self.col = mcol

        self.m = ['nan']*int(r*self.df.shape[0])

        self.new_x = self.df[self.col][:]

        for idx,value in zip(choice(range(len(self.df)),size=len( self.m),replace=False), self.m):

            self.new_x[idx] = value

            self.df[self.col] = self.new_x

        return self.df

        """PREPARING THE DATA AND PREDICTING NaNS"""

    def Predict_NaN(self,miss_df,clf):

        self.miss_df = miss_df

        self.dfnan = self.miss_df[self.miss_df.isnull().any(1)]

        self.dfnan = self.dfnan.drop(['Id','Species',self.col],axis=1)

        self.new_df = self.miss_df.dropna()

        self.y = self.new_df.pop(self.col)

        self.X = self.new_df.drop(['Id','Species'],axis=1)

         

        if clf == 'Kmeans':

            self.clf = KMeans(n_clusters=3, random_state=1)

            self.clf.fit(self.X)

            self.p = self.clf.predict(self.dfnan)

        elif clf == 'SVM':

            self.clf = LinearSVR(max_iter=300)

            self.clf.fit(self.X,self.y)

            self.p = self.clf.predict(self.dfnan)

        self.dfnan[self.col] = list(self.p)

        for i in self.dfnan.index:

            for j in self.miss_df.index:

                if i == j:

                    self.miss_df[self.col][j] = self.dfnan[self.col][i]

                        



                    

        return self.miss_df

            

            
path = '/kaggle/input/iris/database.sqlite'

data = ImpNaNs(path,'Iris')

d = data.Read_data()

orige = d.copy()
df = data.Random_nan(mcol='PetalWidthCm',r=0.1)

df.isna().sum()
dkmns = data.Predict_NaN(clf='Kmeans',miss_df=df)
fig ,axs = plt.subplots(1,2,figsize=(12,4),sharex=True)



sns.scatterplot(x="PetalLengthCm", y="PetalWidthCm", hue = 'Species',

                     data=dkmns,ax = axs[0]).set_title('K-MEAN')

sns.scatterplot(x="PetalLengthCm", y="PetalWidthCm", hue = 'Species',

                     data=orige,ax = axs[1]).set_title('ORIGINAL DATA')

plt.show() 
data = ImpNaNs(path,'Iris')

d = data.Read_data()

orige = d.copy()

df = data.Random_nan(mcol='PetalWidthCm',r=0.1)

dsvm = data.Predict_NaN(clf='SVM',miss_df=df)
fig ,axs = plt.subplots(1,2,figsize=(12,4),sharex=True)



sns.scatterplot(x="PetalLengthCm", y="PetalWidthCm", hue = 'Species',

                     data=dsvm,ax = axs[0]).set_title('SVM')

sns.scatterplot(x="PetalLengthCm", y="PetalWidthCm", hue = 'Species',

                     data=orige,ax = axs[1]).set_title('ORIGINAL DATA')

plt.show()  