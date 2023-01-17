import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
content="""Brand

Brand2

y_2006

y_2006.1

y_2007

y_2008

y_2009

y_2010

y_2011

y_2012

y_2013

y_2014

y_2015

y_2016

y_2017

y_2018

Unnamed: 16

Unnamed: 17

Unnamed: 18

Unnamed: 19

Unnamed: 20

Unnamed: 21

Unnamed: 22

Unnamed: 23

Unnamed: 24 

Unnamed: 25

Unnamed: 26

Unnamed: 27

Unnamed: 28

Unnamed: 29

Unnamed: 30

Unnamed: 31

Unnamed: 32"""



columns_list = content.split("\n")

# for i in range(len(columns_list)):

#   columns_list[i] = columns_list[i].strip()
Data = pd.read_csv("../input/swisswatchbrands/swiss watch brands.csv",header=0,names = columns_list,index_col=False)

Data = Data.set_index('Brand')

Data.head()





print("Any missing sample in train set:",Data.isnull().values.any(), "\n")

Data = Data.replace([np.inf, -np.inf], np.nan)

Data =Data.fillna(0)

Data
#Swiss Watch Brand 2006

top_Brand = Data.sort_values(by='y_2006', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_Brand.Brand2, x=top_Brand.y_2006)

plt.xticks()

plt.xlabel('y_2007')

plt.ylabel('Brand2')

plt.title('Swiss Watch Brand 2006')

plt.show()
#Swiss Watch Brand 2007

top_Brand = Data.sort_values(by='y_2007', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_Brand.Brand2, x=top_Brand.y_2007)

plt.xticks()

plt.xlabel('y_2007')

plt.ylabel('Brand2')

plt.title('Swiss Watch Brand 2007')

plt.show()
#Swiss Watch Brand 2008

top_Brand = Data.sort_values(by='y_2008', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_Brand.Brand2, x=top_Brand.y_2008)

plt.xticks()

plt.xlabel('y_2008')

plt.ylabel('Brand2')

plt.title('Swiss Watch Brand 2008')

plt.show()
#Swiss Watch Brand 2009

top_Brand = Data.sort_values(by='y_2009', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_Brand.Brand2, x=top_Brand.y_2009)

plt.xticks()

plt.xlabel('y_2009')

plt.ylabel('Brand2')

plt.title('Swiss Watch Brand 2009')

plt.show()
#Swiss Watch Brand 2010

top_Brand = Data.sort_values(by='y_2010', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_Brand.Brand2, x=top_Brand.y_2010)

plt.xticks()

plt.xlabel('y_2010')

plt.ylabel('Brand2')

plt.title('Swiss Watch Brand 2010')

plt.show()
#Swiss Watch Brand 2011

top_Brand = Data.sort_values(by='y_2011', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_Brand.Brand2, x=top_Brand.y_2011)

plt.xticks()

plt.xlabel('y_2011')

plt.ylabel('Brand2')

plt.title('Swiss Watch Brand 2011')

plt.show()
#Swiss Watch Brand 2012

top_Brand = Data.sort_values(by='y_2012', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_Brand.Brand2, x=top_Brand.y_2012)

plt.xticks()

plt.xlabel('y_2012')

plt.ylabel('Brand2')

plt.title('Swiss Watch Brand 2012')

plt.show()
#Swiss Watch Brand 2013

top_Brand = Data.sort_values(by='y_2013', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_Brand.Brand2, x=top_Brand.y_2013)

plt.xticks()

plt.xlabel('y_2013')

plt.ylabel('Brand2')

plt.title('Swiss Watch Brand 2013')

plt.show()
#Swiss Watch Brand 2014

top_Brand = Data.sort_values(by='y_2014', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_Brand.Brand2, x=top_Brand.y_2014)

plt.xticks()

plt.xlabel('y_2014')

plt.ylabel('Brand2')

plt.title('Swiss Watch Brand 2014')

plt.show()
#Swiss Watch Brand 201

top_Brand = Data.sort_values(by='y_2015', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_Brand.Brand2, x=top_Brand.y_2015)

plt.xticks()

plt.xlabel('y_2015')

plt.ylabel('Brand2')

plt.title('Swiss Watch Brand 2015')

plt.show()
#Swiss Watch Brand 2016

top_Brand = Data.sort_values(by='y_2016', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_Brand.Brand2, x=top_Brand.y_2016)

plt.xticks()

plt.xlabel('y_2016')

plt.ylabel('Brand2')

plt.title('Swiss Watch Brand 2016')

plt.show()
#Swiss Watch Brand 2017

top_Brand = Data.sort_values(by='y_2017', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_Brand.Brand2, x=top_Brand.y_2017)

plt.xticks()

plt.xlabel('y_2017')

plt.ylabel('Brand2')

plt.title('Swiss Watch Brand 2017')

plt.show()
#Swiss Watch Brand 2018

top_Brand = Data.sort_values(by='y_2018', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_Brand.Brand2, x=top_Brand.y_2018)

plt.xticks()

plt.xlabel('y_2018')

plt.ylabel('Brand2')

plt.title('Swiss Watch Brand 2018')

plt.show()