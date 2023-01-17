# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

data= pd.read_csv("../input/pulsar_stars.csv")
data.head()
data.info()
data.describe()
# pd.plotting.scatter_matrix



color_list= ['red' if i==0 else 'green' for i in data.loc[:,'target_class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns!='target_class'],

                                        c=color_list,

                                        figsize=[15,15],

                                        diagonal='hist',

                                        alpha=0.5,

                                        s=200,

                                        marker='*',

                                        edgecolor='black')

plt.show()

            

                                    
# pair plot



sns.pairplot(data=data,

            palette="husl",

            hue="target_class",

            vars=[" Mean of the integrated profile",

                   " Excess kurtosis of the integrated profile",

                   " Skewness of the integrated profile",

                   " Mean of the DM-SNR curve",

                   " Excess kurtosis of the DM-SNR curve",

                   " Skewness of the DM-SNR curve"])

plt.suptitle("PairPlot of Data Without Std. Dev. Fields", fontsize=18)

plt.tight_layout()

plt.show()
# Correlation HeatMap

plt.figure(figsize=(16,12))

sns.heatmap(data=data.corr(),annot=True,cmap="bone",linewidths=1,fmt=".2f",linecolor="blue")

plt.title("Correlation Map",fontsize=22)

plt.tight_layout()

plt.show()
sns.countplot(x="target_class", data=data)

data.loc[:,'target_class'].value_counts()
# K-NEAREST NEIGHBORS (KNN)


