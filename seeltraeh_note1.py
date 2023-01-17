# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/ign.csv');

dataset.head()
games_from_2012 = dataset[dataset['release_year'] == 2012]

games_from_2012.head()
sns.distplot(games_from_2012['score'])
count = dataset['platform'].value_counts()

count
sns.barplot(x = count.head(), y = count.head().index)