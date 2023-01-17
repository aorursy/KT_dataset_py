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
# reading training data 

trainData = pd.read_csv("../input/learn-together/train.csv")
# visualising first 8 rows of data 

trainData.head(8).T
# different columns present in the dataset

trainData.columns.tolist()

# checking the size of the training data

print("The size of the training data: {}".format(trainData.shape))
#since 'Id' column is of no use we can remove it

trainData.drop("Id", axis=1, inplace=True)

trainData.columns.tolist()
#checking the size of the data

print("The size of train data after dropping the 'Id' column: {}".format(trainData.shape))
# checking for categorical values 

# by describing, if value lies in between 0 and 1 only, it will be categorical value

trainData.describe().T
# soil_type and wilderness_area min and max values are 0 and 1

# so marking them as categorical value4

trainData.iloc[:,10:-1] = trainData.iloc[:,10:-1].astype("category")
# generating heat map to visualise the correlation between different features

import seaborn as sns

import matplotlib.pyplot as plt

color = sns.color_palette()

sns.set_style('darkgrid')

corrmat = trainData.corr()

plt.figure(figsize=(10,10))

g = sns.heatmap(trainData.corr(), linewidths=0.8, annot=True, cmap="RdYlGn")
sns.distplot(trainData['Slope'])

plt.show()