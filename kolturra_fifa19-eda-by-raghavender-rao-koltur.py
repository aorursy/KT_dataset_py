# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import math

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

import PIL as Image

import requests

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import datetime

from mpl_toolkits import mplot3d

from mpl_toolkits.mplot3d import Axes3D

#import urllib, cStringIO
ScurrentDT = datetime.datetime.now()

print(str(ScurrentDT))
im = Image.Image.open(requests.get('https://d2gg9evh47fn9z.cloudfront.net/800px_COLOURBOX2627441.jpg', stream=True).raw)

im
fifa19 = pd.read_csv('../input/data.csv',index_col=0)

#fifa19.head()
del fifa19['Photo']

del fifa19['Flag']

del fifa19['Club Logo']
fifa19.shape
print("FIFA 19 null values count")

#fifa19.isnull().sum()
dfifa19= fifa19.dropna(subset=['Preferred Foot'])
print("Before dropping null values: ",len(fifa19))

print("After dropping null values: ",len(dfifa19))

print('Total ',len(fifa19)-len(dfifa19),' rows are dropped')

print("Before updating null values : ",fifa19.shape,"After updating null values : ",dfifa19.shape)
dfifa19['ValueChange']=dfifa19['Value'].str.replace('€','')

dfifa19['Valuemeter'] = dfifa19['ValueChange'].str.split('K|M').str[0].astype(float)

dfifa19['mul'] = dfifa19['ValueChange'].str[-1]

dfifa19['PlayerValue'] = np.where(dfifa19['mul']=='K',dfifa19['Valuemeter']*1000,dfifa19['Valuemeter']*1000000)

del dfifa19['Value']

del dfifa19['ValueChange']

del dfifa19['Valuemeter']

del dfifa19['mul']

dfifa19['ValueChange']=dfifa19['Wage'].str.replace('€','')

dfifa19['Valuemeter'] = dfifa19['ValueChange'].str.split('K|M').str[0].astype(float)

dfifa19['mul'] = dfifa19['ValueChange'].str[-1]

dfifa19['PlayerWage'] = np.where(dfifa19['mul']=='K',dfifa19['Valuemeter']*1000,dfifa19['Valuemeter']*1000000)

del dfifa19['Wage']

del dfifa19['ValueChange']

del dfifa19['Valuemeter']

del dfifa19['mul']
dfifa19['ValueChange']=dfifa19['Release Clause'].str.replace('€','')

dfifa19['Valuemeter'] = dfifa19['ValueChange'].str.split('K|M').str[0].astype(float)

dfifa19['mul'] = dfifa19['ValueChange'].str[-1]

dfifa19['PlayerReleaseClause'] = np.where(dfifa19['mul']=='K',dfifa19['Valuemeter']*1000,dfifa19['Valuemeter']*1000000)

del dfifa19['Release Clause']

del dfifa19['ValueChange']

del dfifa19['Valuemeter']

del dfifa19['mul']
#Changing Preferred Foot data categorical to numerical data (Left as 1 and Right 2)

dfifa19['Preferred Foot']=np.where(dfifa19['Preferred Foot']=='Left',1,2)

dfifa19['International Reputation']=dfifa19['International Reputation'].astype(np.int64)

dfifa19['Weak Foot']=dfifa19['Weak Foot'].astype(np.int64)
#print("FIFA 19 null values count after droping na from Preferred Foot")

dfifa19.isnull().sum()

#dfifa19.to_csv('MRaghu.csv')
dfifa19.describe().transpose()
dfifa19.corr()
# text = dfifa19['Name'].unique()

# text =""

# for i in range(len(dfifa19['Name'].unique())):

#     y =dfifa19.iloc[i,1]

#     z="".join(y.split())

#     text = text+" "+z

# #football_mask = np.array(Image.Image.open("football-157931_1280.png"))

# football_mask = np.array(Image.Image.open(requests.get('http://www.cndajin.com/data/wls/213/19500701.jpg',stream=True).raw))

# #football_mask

# football_mask[football_mask == 0] = 255

# #football_mask

# wc = WordCloud(background_color="black", max_words=1000, mask=football_mask, contour_width=3, contour_color='white')#'firebrick')

# # Generate a wordcloud

# wc.generate(text)

# # store to file

# wc.to_file("football-157931_1280.png")

# # show

# plt.figure(figsize=[10,10])

# plt.imshow(wc, interpolation='bilinear')

# plt.axis("off")

# plt.title('Football Players List')

# plt.show()
#from matplotlib.ticker import funcformatter

y = dfifa19.corr()

plt.figure(figsize=(100,40))

fmt = lambda x,pos: '{:.0%}'.format(x)

sns.heatmap(y,fmt="g",linewidths=.2,vmin=0, vmax=1,annot=True,square = True) #center=1,

plt.show()

#plt.savefig('Heatmap of dataset')
plt.figure(figsize=(25,5))

sns.countplot(x="Age", data=dfifa19).set_title('Age Wise Players Count')

plt.show()
plt.figure(figsize=(35,10))

sns.countplot(x="Nationality", data=dfifa19).set_title('Country Wise Number Of Players Count')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(35,10))

sns.barplot(x='Nationality',y='Age',data=dfifa19,hue=dfifa19['Age']).set_title('Country and Age Wise Status')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(100,10))

sns.countplot(x="Club", data=dfifa19).set_title('Club Wise Number Of Players Count')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(100,10))

sns.barplot(x='Club',y='Age',data=dfifa19).set_title('Club and Age Wise Status')#,hue=dfifa19['Age']

plt.xticks(rotation=90)

plt.show()
# plt.figure(figsize=(20,10))

# sns.pairplot(dfifa19[['Age','Overall','Potential','PlayerValue','PlayerWage','PlayerReleaseClause']])

# plt.show()
g = sns.PairGrid(dfifa19[['Age','Overall','Potential','PlayerValue','PlayerWage','PlayerReleaseClause']], diag_sharey=False)

g.map_lower(sns.kdeplot)

g.map_upper(sns.scatterplot)

g.map_diag(sns.kdeplot, lw=3)

plt.show()
plt.figure(figsize=(25,10))

#sns.boxenplot(x='Age',y='PlayerValue',data=dfifa19).set_title('Age and PlayerValue Wise Status')

#sns.scatterplot(x = school['reduced_lunch'],y = school['school_rating'])

sns.residplot(x='Age', y='PlayerValue',data = dfifa19,lowess=True, color="g").set_title('Age and PlayerValue Wise Status')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(25,10))

sns.boxenplot(x='Age',y='PlayerWage',data=dfifa19).set_title('Age and Player Wage Wise Status')

#sns.scatterplot(x = school['reduced_lunch'],y = school['school_rating'])

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,10))

#sns.boxenplot(x='Age',y='PlayerWage',data=dfifa19)

sns.scatterplot(x = 'PlayerValue',y = 'PlayerWage',data=dfifa19,hue=dfifa19['Age']).set_title('Player Value,Wage and Age Wise Status')

#plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(5,2))

sns.countplot(x="Preferred Foot", data=dfifa19).set_title('Perferred Foot Wise Number Of Players Count')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(35,10))

sns.barplot(x='Nationality',y='Age',data=dfifa19,hue=dfifa19['Preferred Foot']).set_title('Country,Age and Preferred Foot Wise Status')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(100,10))

sns.barplot(x='Club',y='Age',data=dfifa19,hue=dfifa19['Preferred Foot']).set_title('Club,Age and Preferred Foot Wise Status')#,hue=dfifa19['Age']

plt.xticks(rotation=90)

plt.show()
EcurrentDT1 = datetime.datetime.now()

print('Code started from : ',str(ScurrentDT))

print('Code Ended at : ',str(EcurrentDT1))