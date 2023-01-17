# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib as mpl



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv')
data.info()

data.head(10)
data.columns
# Log Distrubution of the Parameters



data['views_log'] = np.log(data['views'] + 1)

data['likes_log'] = np.log(data['likes'] + 1)

data['dislikes_log'] = np.log(data['dislikes'] + 1)





plt.figure(figsize = (18,6))  

plt.figure(1)

plt.title('Log Distrubution',fontsize=16)



plt.subplot(311)

g1 = sns.distplot(data['views_log'],color='blue')

g1.set_title('Views Distrubution',fontsize=8)



plt.subplot(312)

g1 = sns.distplot(data['likes_log'],color='black')

g1.set_title('Likes Distrubution',fontsize=8)



plt.subplot(313)

g1 = sns.distplot(data['dislikes_log'],color='red')

g1.set_title('Dislikes Distrubution',fontsize=8)

plt.show()
# Plot of the watch rates of the parameters



plt.figure(figsize = (18,6)) 



plt.subplot(311)

data.views.plot(kind ='line', color='b',alpha=0.9,grid=True)

plt.xlabel('Number')              # label = name of label

plt.ylabel('Views')

plt.subplot(312)

data.likes.plot(kind ='line', color='b',alpha=0.9,grid=True)

plt.xlabel('Number')              # label = name of label

plt.ylabel('Likes')

plt.subplot(313)

data.dislikes.plot(kind ='line', color='r',alpha=0.9,grid=True)

plt.xlabel('Number')              # label = name of label

plt.ylabel('Dislikes')

plt.show()
# Histogram Graphic for the parameters

# bins = number of bar in figure

plt.figure(figsize = (18,6))



plt.subplot(311)

data.views.plot(kind = 'hist',bins = 500)

plt.subplot(312)

data.likes.plot(kind = 'hist',bins = 500)

plt.subplot(313)

data.dislikes.plot(kind = 'hist',bins = 500)

plt.show()
# Like-Dislike Rates



data['like_rate'] =  data ['likes'] / data['views'] * 100

data['dislike_rate'] =  data ['dislikes'] / data['views'] * 100

data['comment_rate'] =  data ['comment_count'] / data['views'] * 100





plt.figure(figsize = (18,6))  

plt.figure(1)

plt.title('Log Distrubution',fontsize=16)



plt.subplot(311)

g1 = sns.distplot(data['like_rate'],color='blue')

g1.set_title('Like_Rate Distrubution',fontsize=8)



plt.subplot(312)

g1 = sns.distplot(data['dislike_rate'],color='black')

g1.set_title('Dislike_Rate Distrubution',fontsize=8)



plt.subplot(313)

g1 = sns.distplot(data['comment_rate'],color='red')

g1.set_title('Comment_Rate Distrubution',fontsize=8)

plt.show()
# Analysing the correlation between the parameters



plt.figure(figsize = (16,8))



#Let's verify the correlation of each value

sns.heatmap(data[['like_rate', 'dislike_rate', 'comment_rate',

         'views_log','likes_log','dislikes_log']].corr(), annot=True)

plt.show()
# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'Views':'High','Likes':'High','Dislikes':'Low'}

for key,value in dictionary.items():

    print(key," : ",value)

print('Desired Amounts for the Watching Rates')
data1=data['likes'].head()

data2=data['dislikes'].head()

print(pd.concat([data1,data2],axis=0)) #dikey

print(pd.concat([data1,data2],axis=1)) #yatay
data.loc[15:17]