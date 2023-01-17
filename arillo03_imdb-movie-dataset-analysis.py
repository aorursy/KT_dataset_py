# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn#for visuals

sn.set(style="white", color_codes=True)#customizes the graphs

import matplotlib.pyplot as mp #for visuals

%matplotlib inline

import warnings #suppress certain warnings from libraries

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

movie_data = pd.read_csv("../input/movie_metadata.csv")

movie_data
movie_data.shape
movie_data.head(10)
movie_data.corr()
#Cleaning the data for all the NaN values

movie_data.fillna(value=0,axis=1,inplace=True)



#Getting average values from the dataset

movie_data.describe()
#Slicing the data in half for a clearer visualization

movie_sliced = movie_data[0:2501]



#Building the plot

mp.figure(figsize=(15,15))

sn.swarmplot(x='imdb_score', y='country', data = movie_sliced)

mp.title ('Which countries produce better movies?', fontsize=20, fontweight='bold')

mp.xlabel('Score')

mp.ylabel('Country')

mp.show()