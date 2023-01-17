# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from sklearn.decomposition import PCA #principal component analysis module

from sklearn.cluster import KMeans #KMeans clustering

import matplotlib.pyplot as plt #Python defacto plotting library

import seaborn as sns # More snazzy plotting library

%matplotlib inline

# Any results you write to the current directory are saved as output.
movie =pd.read_csv('../input/movie_metadata.csv') #reads csv and creates dataframe called movie

movie.shape
movie.head(10)
movie.corr() 

"""

1. looks like gross, num_voted_users, num_user_for_reviews and movie_facebook_likes 

are all positively correlated with num_critic_for_reviews

2. Duration of the movie and director_facebook_likes are not significantly correlated 

with other variables (<0.4)

3. Actor_3facebook_likes is positively correlated with cast_total_facebooklikes and

actor_2_facebook_likes.

4. there isn't any significantly negative correlation among the variables (<-0.4)

"""
correlation = movie.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different fearures')