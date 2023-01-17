# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd
movie = pd.read_csv('../input/movie_metadata.csv')

movie.head()
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
class_movie = LabelEncoder()
movie['genres'] = class_movie.fit_transform(movie['genres'].values)
from sklearn.ensemble import RandomForestClassifier
feat_labels=movie['genres']
str_list = [] # empty list to contain columns with strings (words)

for colname, colvalue in movie.iteritems():

    if type(colvalue[1]) == str:

         str_list.append(colname)

# Get to the numeric columns by inversion            

num_list = movie.columns.difference(str_list) 
movie_num = movie[num_list]

#del movie # Get rid of movie df as we won't need it now

movie_num.head()
movie_num = movie_num.fillna(value=0, axis=1)
X,y = movie_num.iloc[:,1:].values,movie_num.genres.values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler

X_train_std = StandardScaler().fit_transform(X_train)
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline 

pca = PCA().fit(X_train_std,y_train) 

plt.plot(np.cumsum(pca.explained_variance_ratio_)) 

plt.show() 
pca = PCA(n_components=3)

x_12 = pca.fit_transform(X_train_std,y_train)
plt.figure(figsize = (7,7))

plt.scatter(x_12[:,0],x_12[:,2],c='goldenrod',alpha=0.5)

plt.ylim(-10,30)

plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)

X_clustered = kmeans.fit_predict(x_12)



LABEL_COLOR_MAP = {0 : 'r',1 : 'g',2 : 'b',3 : 'm'}

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]



plt.figure(figsize = (7,7))

plt.scatter(x_12[:,0],x_12[:,2], c= label_color, alpha=0.5) 

plt.show()