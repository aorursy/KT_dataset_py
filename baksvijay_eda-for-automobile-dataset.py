# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set(color_codes=True)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
auto = pd.read_csv('/kaggle/input/automobile-dataset/Automobile_data.csv')
auto.head()
sns.distplot(auto['engine-size']);
sns.distplot(auto['city-mpg'], kde=False, rug=True);
sns.jointplot(auto['engine-size'],auto['wheel-base'])
sns.jointplot(auto['engine-size'], auto['wheel-base'], kind="hex")
sns.jointplot(auto['engine-size'], auto['wheel-base'], kind="kde")
sns.pairplot(auto[['city-mpg', 'engine-size', 'wheel-base']])
sns.stripplot(auto['fuel-type'], auto['city-mpg'])
sns.stripplot(auto['fuel-type'], auto['city-mpg'],jitter=True)
sns.swarmplot(auto['fuel-type'], auto['city-mpg'])
sns.boxplot(auto['num-of-doors'], auto['city-mpg'], hue=auto['fuel-type'])
sns.barplot(auto['body-style'], auto['city-mpg'], hue=auto['engine-location'])
sns.countplot(auto['body-style'])
sns.pointplot(auto['fuel-system'], auto['engine-size'], hue=auto['num-of-doors'])
sns.factorplot(x="fuel-type", 

               y="engine-size", 

               hue="num-of-doors", 

               col="engine-location", 

               data=auto, 

               kind="swarm")
sns.lmplot(x="city-mpg", y="engine-size", data=auto)
sns.lmplot(x="city-mpg", y="engine-size",hue="fuel-type", data=auto)