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
links = pd.read_csv('../input/the-movies-dataset/links.csv')

links_small = pd.read_csv('../input/the-movies-dataset/links_small.csv')



links.head(1), links_small.head(1)
print ('-links info-')



links.info()
print ('-links_small info-')



links_small.info()
print ('-links NaN sum-')



links.isnull().sum()
print ('-links_small NaN sum-')



links_small.isnull().sum()
links = links.fillna(0)

links_small = links.fillna(0)



links = links.set_index('movieId')

links_small = links.set_index('movieId')
links.head(1)