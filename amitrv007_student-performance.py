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
import pandas as pd
# let's read the data into a dataframe first

df = pd.read_csv('../input/StudentsPerformance.csv')
# let's fetch the no. of rows and columns

print("The dataframe has %d rows & %d columns"%(df.shape[0],df.shape[1]))
# let's have a sneak peak at the data

df.head()
# let's visualize data distribution of girls vs boys for maths score

import seaborn as sns

sns.boxplot(x=df.gender,y=df['math score'],data=df)

# let's check if the dataset is balanced

# i.e to check if #girls appearing for maths exam is comparable to #boys

df.groupby(['gender'])['gender'].count()
# let's separate out the girl's scores from the dataframe

girls = df[df['gender'] == 'female']['math score']
# similar task for boys

boys = df[df['gender'] == 'male']['math score']
# no need to execute below piece of code

'''

# we will do the below steps 50 times as we wish to draw 50 samples

# Step 1 : Draw 100 data-points (sample)

# Step 2 : Calculate mean of the sample

# Step 3 : Save it into a Series

from sklearn.utils import shuffle

import numpy as np

girls_means = []

boys_means = []

for i in range(1,50):

    girls = shuffle(girls)

    boys = shuffle(boys)

    mean_g = np.mean(girls[:100])

    mean_b = np.mean(boys[:100])

    girls_means.append(mean_g)

    boys_means.append(mean_b)



girls_means = pd.Series(girls_means)

boys_means = pd.Series(boys_means)

'''
from scipy import stats

stats.ttest_ind(girls,boys)