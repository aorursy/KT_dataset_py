# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# initialise the variables 

train = pd.read_csv("../input/forest-cover-type-kernels-only/train.csv.zip")

test = pd.read_csv("../input/forest-cover-type-kernels-only/test.csv.zip")

sample_submission = pd.read_csv('../input/forest-cover-type-kernels-only/sample_submission.csv.zip')

sampleSubmission = pd.read_csv('../input/forest-cover-type-kernels-only/sampleSubmission.csv.zip')

## now let's see how many data points we have and how many coloums 

print("The number of data points is %i " % train.shape[0])

print("The number of coloums is %i " % train.shape[1])



train.isnull().sum().sum()
import seaborn as sns

import matplotlib.pyplot as plt



correlation = train.corr()



f,ax = plt.subplots(figsize=(15, 15))

cmap = sns.diverging_palette(200, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, vmax=0.75, center=0,

            square=True, linewidths=.5)
plt.plot(correlation)

correlation
train.drop(['Id'], inplace = True, axis = 1 )

train.drop(['Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )
corr = train.corr()



f,ax = plt.subplots(figsize=(15, 15))

cmap = sns.diverging_palette(200, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, vmax=0.75, center=0,

            square=True, linewidths=.5)


def plotRelation(first_feature, sec_feature):

    classes = np.array(list(train.Cover_Type.values))

    plt.scatter(first_feature, sec_feature, c = classes, s=10)

    plt.xlabel(first_feature.name)

    plt.ylabel(sec_feature.name)





f = plt.figure(figsize=(25,20))

for  i in range(train.shape[1]):

    for j in range(train.shape[1]):

        if ( i!=j):

            coll1 = train.iloc[:,i]

            coll2 = train.iloc[:,j]

            if ((coll1.corr(coll2)) > 0.2):

                 #f.add_subplot(444)

                 plotRelation(coll1,coll2)

                 #plt.plot(coll1,coll2)

                



            

    