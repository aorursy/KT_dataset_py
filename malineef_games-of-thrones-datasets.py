# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn # for visuals

sn.set(style="white", color_codes = True) #customizes graphs

import matplotlib.pyplot as mp # for graphs

%matplotlib inline

#how the graphs are printed

import warnings #suppress warnings from libraries

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



battles = pd.read_csv("../input/battles.csv", sep=",", header = 0)

print(battles.shape)

deaths = pd.read_csv("../input/character-deaths.csv", sep=",", header=0)

print(deaths.shape)
predictions = pd.read_csv("../input/character-predictions.csv", sep=",", header=0)

print(predictions.shape)
battles.head(10)

#the first 10 lines from the battles dataset
deaths.head(10)

#the first 10 rows from the character deaths dataset
predictions.head(10)

#the first 10 rows from the predictions dataset
battles.corr()
deaths.corr()
predictions.corr()
battles['year'].corr(deaths['Gender'])
correlation = battles.corr()

mp.figure(figsize = (10,10))

sn.heatmap(correlation, vmax=1, square=True, annot=True,cmap='cubehelix')



mp.title("Attacking and Defending Correlation")
correlation = deaths.corr()

mp.figure(figsize = (10,10))

sn.heatmap(correlation, vmax=1, square=True, annot=True,cmap='cubehelix')



mp.title("Character Deaths")
correlation = predictions.corr()

mp.figure(figsize = (15,15))

sn.heatmap(correlation, vmax=1, square=True, annot=True,cmap='cubehelix')



mp.title("Predictions of Death")
battles.groupby(['attacker_king', 'defender_king']).count()['name'].plot(kind = 'barh')
battles.groupby(['attacker_king', 'attacker_outcome']).count()['name'].unstack().plot(kind = 'barh')
attackers = battles.attacker_king.map(lambda x:str(x).split(","))

empty_array = []

for i in attackers:

    empty_array = np.append(empty_array, i)
from wordcloud import WordCloud

cloud = WordCloud(width=1440, height=1080, relative_scaling=0.5, stopwords=['battle']).generate(" ".join(empty_array))

mp.figure(figsize=(20, 15))

mp.imshow(cloud)

mp.axis('off')

mp.show()