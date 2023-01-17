import os

for dirname, _, filenames in os.walk('/kaggle/input/gapminder'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 

import re

import mailbox



import scipy.stats

from IPython import display 

from ipywidgets import interact, widgets
gapminder = pd.read_csv('/kaggle/input/gapminder.csv')

gapminder.head()
with plt.rc_context():

      plt.rc("figure", figsize=(16,6))

      gapminder[gapminder['year'] == 1960].plot.scatter('babies_per_woman', 'age5_surviving')

      plt.show()
def plotyear(year):

    data = gapminder[gapminder['year'] == year]

    

    area = 0.000005 *data['population']

    colors = data['region'].map({'Africa': 'skyblue', 'Europe':'gold', 'America':'palegreen', 'Asia':'coral'})

    

    

    data.plot.scatter('babies_per_woman', 'age5_surviving', s=area, c=colors, 

                         linewidths = 1, edgecolors = 'k', figsize=(16,8))

    plt.show()

plotyear(1960) 
interact(plotyear, year = widgets.IntSlider(min=1948, max=2015, step = 1, value= 1960))