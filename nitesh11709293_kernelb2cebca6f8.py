# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#importing datasets 



data = pd.read_csv("../input/IMDB-Movie-Data.csv")
data.columns
# Calculate mean rating of all movies 

data.groupby('Title')['Rating'].mean().sort_values(ascending=False).head() 
# Calculate count rating of all movies 

data.groupby('Title')['Rating'].count().sort_values(ascending=False).head() 
# creating dataframe with 'rating' count values 

ratings = pd.DataFrame(data.groupby('Title')['Rating'].mean())  

  

ratings['No of ratings'] = pd.DataFrame(data.groupby('Title')['Rating'].count()) 

  

ratings.head() 
import matplotlib.pyplot as plt 

import seaborn as sns 

  

sns.set_style('white') 

%matplotlib inline 
# plot graph of 'num of ratings column' 

plt.figure(figsize =(10, 4)) 

  

ratings['No of ratings'].hist(bins = 70)
# plot graph of 'ratings' column 

plt.figure(figsize =(10, 4)) 

  

ratings['Rating'].hist(bins = 70) 
# Sorting values according to  

# the 'num of rating column' 

moviemat = data.pivot_table(index ='Rank', 

              columns ='Title', values ='Rating') 

  

moviemat.head() 

  

ratings.sort_values('No of ratings', ascending = False).head(10) 
# analysing correlation with similar movies 

Thehost_user_ratings = moviemat['The Host'] 

StarTrek_user_ratings = moviemat['Star Trek'] 

  

Thehost_user_ratings.head() 

similar_to_Thehost = moviemat.corrwith(Thehost_user_ratings) 

similar_to_StarTrek = moviemat.corrwith(StarTrek_user_ratings) 

  

corr_Thehost = pd.DataFrame(similar_to_Thehost, columns =['Correlation']) 

corr_Thehost.dropna(inplace = True) 

  

corr_Thehost.head() 
corr_StarTrek = pd.DataFrame(similar_to_StarTrek, columns =['Correlation']) 

corr_StarTrek.dropna(inplace = True) 

  

corr_StarTrek.head() 