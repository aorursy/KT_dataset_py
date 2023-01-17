# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn # for visuals

sn.set(style="white", color_codes = True) #customizes graphs

import matplotlib.pyplot as mp #for visuals

%matplotlib inline

#how graphs are printed out

import warnings #suppress certain warnings from libraries

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



movie = pd.read_csv("../input/movie_metadata.csv", sep=",", header=0)

print(movie.shape)
movie.head(10)
movie.corr()
correlation = movie.corr()

mp.figure(figsize = (10,10))

sn.heatmap(correlation, vmax=1, square=True, annot=True,cmap='cubehelix')



mp.title("Correlation between Movie Info")
plot_keywords = movie.plot_keywords.map(lambda x:str(x).split(","))

empty_array = []

for i in plot_keywords:

    empty_array = np.append(empty_array, i)
from wordcloud import WordCloud

cloud = WordCloud(width=1440, height=1080, relative_scaling=0.5, stopwords=['title']).generate(" ".join(empty_array))

mp.figure(figsize=(20, 15))

mp.imshow(cloud)

mp.axis('off')

mp.show()
correlation["budget"].corr(correlation["gross"])
chart = movie.groupby(["director_name"])['gross'].count()

ax=mp.xticks(rotation=90)

chart.plot()