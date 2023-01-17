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
import numpy as np # Basic linear algebra functions

import pandas as pd # Structured data operations and manipulations

import seaborn as sn # Statistical data visualization

import matplotlib.pyplot as plt # Python defacto plotting library

%matplotlib inline 

import warnings
data = pd.read_csv("../input/movie_metadata.csv")
data
data.shape
df = pd.DataFrame(data)

new_data = df.dropna() #remove NaN
new_data
new_data.shape
new_data.columns
new_data.head(10)
new_data['genres'].value_counts()
new_data.describe() # Get summary of numerical variables
new_data.corr()
# Set up the matplotlib figure

plt.rcParams['figure.figsize']=[12.0, 10.0]

plt.title('Pearson Correlation')

# Draw the heatmap using seaborn

sn.heatmap(new_data.corr(),linewidths=0.5,vmax=1.0, square=True, cmap="BuGn_r", linecolor='black', annot=True)
plt.rcParams['font.size']=11

plt.rcParams['figure.figsize']=[8.0, 5.5]

new_data.plot(kind='scatter', x='num_critic_for_reviews', y='movie_facebook_likes', color='Blue')

plt.title("Relationship Between Number Critic for Reviews and Movie Facebook Likes")

plt.tight_layout() 
plt.rcParams['font.size']=11

plt.rcParams['figure.figsize']=[8.0, 5.5]

new_data.plot(kind='scatter', x='num_critic_for_reviews', y='imdb_score', color='Red')

plt.title("Relationship Between Number Critic for Reviews and IMDB Score")

plt.tight_layout() 