# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/topic-survey/Topic_Survey_Assignment.csv")

df
print(df.shape)
df.info()


# use the inline backend to generate the plots within the browser

%matplotlib inline 



import matplotlib as mpl

import matplotlib.pyplot as plt



mpl.style.use('ggplot') #for ggplot-like style



# check for latest version of Matplotlib

print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0
# Sorting the values based on Very interested

df.sort_values(['Very interested'], ascending=False, axis=0, inplace=True)



# View top 5 rows of the data 

df.head()
#plotting

ax = df.plot(kind='bar', 

                figsize=(20, 8),

                rot=90,color = ['#5cb85c','#5bc0de','#d9534f'],

                width=.8,fontsize=14)





# Setting plot title

ax.set_title('Respondents Interest in Data Science Areas',fontsize=16)



# Setting figure background color

ax.set_facecolor('white')



# setting legend font size

ax.legend(fontsize=14,facecolor = 'white') 



# Removing the Border 

ax.get_yaxis().set_visible(False)



# Creating a function to display the percentage.



for p in ax.patches:

    ax.annotate(np.round(p.get_height(),decimals=2), 

                (p.get_x()+p.get_width()/2., p.get_height()), 

                ha='center', 

                va='center', 

                xytext=(0, 10), 

                textcoords='offset points',

                fontsize = 14

               )

plt.show()