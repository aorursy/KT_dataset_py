# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from ipywidgets import interact

from IPython.display import display

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

%matplotlib inline

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
categories=('A','B','C')



data = {

            'days':      np.random.randint(12, size=100), 

            'category':  np.random.choice(categories, 100),

            'value':     100.0 * np.random.random_sample(100)

       }



df = pd.DataFrame(data)



def select_days(number_of_days):

    df_filtered= df.loc[df['days'] == int(number_of_days)] 

    ax = df_filtered[["category", "value"]].boxplot( by="category", return_type='axes')

    ax["value"].set_title("Day " + number_of_days)

    #print (df_filtered)



days = [str(day) for day in np.arange(12)]

print(days)
f = interact(select_days, number_of_days=days)

display(f)