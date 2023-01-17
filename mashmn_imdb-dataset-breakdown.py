# Author: Mahesh Nair

# Project: https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset/kernels

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/movie_metadata.csv')

data.head()
print(data.columns)
sns.pairplot(data, hue="imdb_score")