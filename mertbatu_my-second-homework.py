# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')
data.info()
data.head()
#correlation map

f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
series = data ['radius_mean']

print(type(series))

data_frame = data[['radius_mean']]

print(type(data_frame))
x = data['radius_mean'] < 15

data[x]
y = data['diagnosis'] == 'B'

data[y]
z = data['diagnosis'] == 'M'

data[z]
data.radius_mean.plot(kind = 'line', color = 'B',label = 'Radius Mean', linewidth = 1,alpha = .5,grid = True, linestyle = ':')

data.texture_mean.plot(kind = 'line', color = 'R',label = 'Texture Mean', linewidth = 1,alpha = .5,grid = True, linestyle = '-.')



plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('Radius Mean')              # label = name of label

plt.ylabel('Texture Mean')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 



data[y].plot(kind='scatter', x='radius_mean', y='texture_mean',alpha = 0.5,color = 'g')

plt.xlabel('Radius Mean')              # label = name of label

plt.ylabel('Texture Mean')

plt.title('Benign-Radius & Texture Means Scatter Plot') 

data[z].plot(kind='scatter', x='radius_mean', y='texture_mean',alpha = 0.5,color = 'r')

plt.xlabel('Radius Mean')              # label = name of label

plt.ylabel('Texture Mean')

plt.title('Malignant-Radius & Texture Means Scatter Plot')            # title = title of plot

plt.show()
threshold = sum(data.radius_mean)/len(data.radius_mean)



data["risk_level"] = ["high" if i>threshold else "low" for i in data.radius_mean]

data.loc[:10,["risk_level","radius_mean","texture_mean"]]
thresholdB = sum(data[y].radius_mean)/len(data[y].radius_mean)

thresholdM = sum(data[z].radius_mean)/len(data[z].radius_mean)



data["risk_level"] = ["high" if i>thresholdM  else "low" if i<thresholdB else "moderate" for i in data.radius_mean]

print("benign: ",thresholdB," ","malignant: ",thresholdM)

data.loc[:10,["risk_level","radius_mean","texture_mean"]]

threshold_textureB = sum(data[y].texture_mean)/len(data[y].texture_mean)

threshold_textureM = sum(data[z].texture_mean)/len(data[z].texture_mean)



def texture_risk(*args):

    

    """This function determines the effects of the texture mean to the risk."""

    

    return ["high" if i>threshold_textureM  else "low" if i<threshold_textureB else "moderate" for i in data.texture_mean]

    

data["texture_risk"] =texture_risk(data["texture_mean"])

print("benign: ",threshold_textureB," ","malignant: ",threshold_textureM)

data.loc[:10,["texture_risk","texture_mean"]]
data[(data.radius_mean > 17) & (data.diagnosis == 'B')]