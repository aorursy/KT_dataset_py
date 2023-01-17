# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/amazon-prime-tv-shows/Prime TV Shows Data set.csv",encoding='cp1252')
df.head(10)
df.rename(columns={'Name of the show': 'name', 'Year of release': 'year', "No of seasons available" : "seasons", "IMDb rating" : "rating", "Age of viewers" : "age", "Language" : 'language', "Genre": "genre"}, inplace=True)
df.info()
df.language.value_counts()
df[df["language"] == "Russian"]
df["rating"].fillna(0, inplace = True )
df.tail(10)
year_count = df.year.value_counts()
year_count
df.sort_values(by="rating", ascending=False).head(15)
most_rating = df[df["rating"] > 8.0]
most_rating["genre"].value_counts()
df.seasons.value_counts()
df[df["seasons"] == 20]
