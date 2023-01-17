# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
sns.set_style('dark')
movie=pd.read_csv("/kaggle/input/u.data",sep="\t")
column=["user_id","item_id","rating","timestamp"]
movie.columns=column
movie.head()
df=pd.read_csv("/kaggle/input/u.item",sep="\|",header=None)
#c.columns=["item_id","title"]
c=df[[0,1]]
c.columns=["item_id","title"]
c.head()
df=pd.merge(movie,c,on="item_id")
df.head()
c=df.groupby('title').mean()['rating']
c.sort_values(ascending=False).head()
d=df.groupby('title').count()['rating']
d.sort_values(ascending=False)
Rate=pd.DataFrame(df.groupby('title').mean()['rating'])

Rate["no_of_ratings"]=pd.DataFrame(df.groupby('title').count()['rating'])
Rate.sort_values(by='rating',ascending=False)
plt.figure(figsize=(10,6))
plt.hist(Rate['no_of_ratings'],bins=70)
plt.show()
plt.figure(figsize=(10,6))
plt.hist(Rate['rating'],bins=70)
plt.show()
sns.jointplot(x=Rate['rating'],y=Rate['no_of_ratings'],data=Rate,alpha=0.85)
plt.show()
moviemat=df.pivot_table(index='user_id',columns='title',values='rating')
def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar_to_movie,columns=['correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(Rate['no_of_ratings'])
    predictions=corr_movie[corr_movie['no_of_ratings']>100].sort_values('correlation',ascending=False)
    return predictions
predictions=predict_movies('Titanic (1997)')
predictions.head()