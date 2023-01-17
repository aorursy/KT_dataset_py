# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

os.listdir("../input")

data=pd.read_csv("../input/movie_metadata.csv")

data.isnull().sum()

# dont worry about NA'S
data['specialparameter']=data.imdb_score*data.num_user_for_reviews
data.sort_values('specialparameter',ascending=False)
data1=data[['plot_keywords','specialparameter']]

data1.head()
dataquantile=data1.quantile(0.99)

dataquantile
data1[data1.specialparameter>dataquantile.specialparameter]