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
content = """Row Labels

Flag

y_1991

y_1992

y_1993

y_1994

y_1995

y_1996

y_1997

y_1998

y_1999

y_2000

y_2001

y_2002

y_2003

y_2004

y_2005

y_2006

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

"""

columns_list = content.split("\n")

# for i in range(len(columns_list)):

#   columns_list[i] = columns_list[i].strip()

Data = pd.read_csv("../input/top15cherriesproducingcountries19612018/Top 15 Cherries Producing Countries 1961 - 2018.csv",header=0,names = columns_list,index_col=False)

Data = Data.set_index('Flag')

Data.head()



Data.info()

Data[0:10]
Data['Row_Labels'] = Data['Row Labels']

Data
print("Any missing sample in train set:",Data.isnull().values.any(), "\n")

Data = Data.replace([np.inf, -np.inf], np.nan)

Data =Data.fillna(0)

Data
#The Most Cherries Producing Countries 1991

Cherries = Data.sort_values(by='y_1991', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_1991)

plt.xticks()

plt.xlabel('y_1991')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 1991')

plt.show()
#The Most Cherries Producing Countries 1992

Cherries= Data.sort_values(by='y_1992', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_1992)

plt.xticks()

plt.xlabel('y_1992')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 1992')

plt.show()
#The Most Cherries Producing Countries 1993

Cherries = Data.sort_values(by='y_1993', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_1993)

plt.xticks()

plt.xlabel('y_1993')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 1993')

plt.show()
#The Most Cherries Producing Countries 1994

Cherries = Data.sort_values(by='y_1994', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_1994)

plt.xticks()

plt.xlabel('y_1994')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 1994')

plt.show()
#The Most Cherries Producing Countries 1995

Cherries = Data.sort_values(by='y_1995', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_1995)

plt.xticks()

plt.xlabel('y_1995')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 1995')

plt.show()
#The Most Cherries Producing Countries 1996

Cherries = Data.sort_values(by='y_1996', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_1996)

plt.xticks()

plt.xlabel('y_1996')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 1996')

plt.show()
#The Most Cherries Producing Countries 1997

Cherries = Data.sort_values(by='y_1997', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_1996)

plt.xticks()

plt.xlabel('y_1997')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 1997')

plt.show()
#The Most Cherries Producing Countries 1998

Cherries = Data.sort_values(by='y_1998', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_1998)

plt.xticks()

plt.xlabel('y_1998')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 1998')

plt.show()
#The Most Cherries Producing Countries 1999

Cherries = Data.sort_values(by='y_1999', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_1999)

plt.xticks()

plt.xlabel('y_1999')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 1999')

plt.show()
#The Most Cherries Producing Countries 2000

Cherries = Data.sort_values(by='y_2000', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2000)

plt.xticks()

plt.xlabel('y_2000')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2000')

plt.show()
#The Most Cherries Producing Countries 2001

Cherries = Data.sort_values(by='y_2001', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2001)

plt.xticks()

plt.xlabel('y_2001')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2001')

plt.show()
#The Most Cherries Producing Countries 2002

Cherries = Data.sort_values(by='y_2002', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2002)

plt.xticks()

plt.xlabel('y_2002')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2002')

plt.show()
#The Most Cherries Producing Countries 2003

Cherries = Data.sort_values(by='y_2003', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2003)

plt.xticks()

plt.xlabel('y_2003')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2003')

plt.show()
#The Most Cherries Producing Countries 2004

Cherries = Data.sort_values(by='y_2004', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2004)

plt.xticks()

plt.xlabel('y_2004')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2004')

plt.show()
#The Most Cherries Producing Countries 2005

Cherries = Data.sort_values(by='y_2005', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2005)

plt.xticks()

plt.xlabel('y_2005')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2005')

plt.show()
#The Most Cherries Producing Countries 2006

Cherries = Data.sort_values(by='y_2006', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2006)

plt.xticks()

plt.xlabel('y_2006')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2006')

plt.show()
#The Most Cherries Producing Countries 2007

Cherries = Data.sort_values(by='y_2007', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2007)

plt.xticks()

plt.xlabel('y_2007')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2007')

plt.show()
#The Most Cherries Producing Countries 2008

Cherries = Data.sort_values(by='y_2008', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2008)

plt.xticks()

plt.xlabel('y_2008')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2007')

plt.show()
#The Most Cherries Producing Countries 2009

Cherries= Data.sort_values(by='y_2009', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2009)

plt.xticks()

plt.xlabel('y_2009')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2009')

plt.show()
#The Most Cherries Producing Countries 2010

Cherries = Data.sort_values(by='y_2010', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2010)

plt.xticks()

plt.xlabel('y_2010')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2010')

plt.show()
#The Most Cherries Producing Countries 2011

Cherries = Data.sort_values(by='y_2011', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2011)

plt.xticks()

plt.xlabel('y_2011')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2011')

plt.show()
#The Most Cherries Producing Countries 2012

Cherries = Data.sort_values(by='y_2012', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2012)

plt.xticks()

plt.xlabel('y_2012')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2012')

plt.show()
#The Most Cherries Producing Countries 2013

Cherries = Data.sort_values(by='y_2013', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2013)

plt.xticks()

plt.xlabel('y_2013')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2013')

plt.show()
#The Most Cherries Producing Countries 2014

Cherries = Data.sort_values(by='y_2014', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2014)

plt.xticks()

plt.xlabel('y_2014')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2014')

plt.show()
#The Most Cherries Producing Countries 2015

Cherries = Data.sort_values(by='y_2015', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2015)

plt.xticks()

plt.xlabel('y_2015')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2015')

plt.show()
#The Most Cherries Producing Countries 2016

Cherries = Data.sort_values(by='y_2016', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2016)

plt.xticks()

plt.xlabel('y_2016')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2016')

plt.show()
#The Most Cherries Producing Countries 2017

Cherries = Data.sort_values(by='y_2017', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2017)

plt.xticks()

plt.xlabel('y_2017')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2017')

plt.show()
#The Most Cherries Producing Countries 2018

Cherries = Data.sort_values(by='y_2017', ascending=False)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=Cherries.Row_Labels, x=Cherries.y_2018)

plt.xticks()

plt.xlabel('y_2018')

plt.ylabel('Row Labels')

plt.title('The Most Cherries Producing Countries 2018')

plt.show()