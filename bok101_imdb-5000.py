# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#import the dataset

df = pd.read_csv("../input/movie_metadata.csv")

# print first movie to inspect attributes

print(df.iloc[96])
df = df.replace(0,np.nan)

df[["imdb_score","movie_facebook_likes"]].dropna()
df.plot.scatter("imdb_score","movie_facebook_likes",grid=True);