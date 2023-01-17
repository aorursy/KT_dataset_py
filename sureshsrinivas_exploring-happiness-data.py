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
#Read in the DataSet into Panda DataFrames

#How does the data look

pdf15=pd.read_csv('../input/2015.csv')

pdf16=pd.read_csv('../input/2016.csv')

pdf17=pd.read_csv('../input/2017.csv')

pdf17.head(15)

pdf17.tail(15)
#What are the columns

pdf17.dtypes
pdf16.dtypes
# What are the correlations

import seaborn as sns

corrmat = pdf17.corr()

sns.heatmap(corrmat, 

            xticklabels=corrmat.columns.values,

            yticklabels=corrmat.columns.values)





corrmat16 = pdf16.corr()

sns.heatmap(corrmat16, 

            xticklabels=corrmat16.columns.values,

            yticklabels=corrmat16.columns.values)
# Extract the US stats

def eUS(pdf, i):

    cpdf = pdf.set_index('Country')

    us = cpdf.loc['United States']

    print(us[i])



print('US Happiness Scores')

eUS(pdf15, 'Happiness Score')    

eUS(pdf16, 'Happiness Score')

eUS(pdf17, 'Happiness.Score')

print('US GDP')

eUS(pdf15, 'Economy (GDP per Capita)')

eUS(pdf16, 'Economy (GDP per Capita)')

eUS(pdf17, 'Economy..GDP.per.Capita.')

# needed imports

from pylab import plot,show

from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage

from scipy.cluster.vq import kmeans,vq



hdata = pdf17['Happiness.Score']

hcdata = pdf17['Country']

# computing K-Means with K = 3 (3 clusters)

centroids,_ = kmeans(hdata,3)

idx,_ = vq(hdata, centroids)





!which python

!python --version
!ps x

!pip list