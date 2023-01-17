from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Read the files 

coronavirusinindia = pd.read_excel("/kaggle/input/coronavirusinindia.xlsx")

print("No. of tweets for #coronavirusinindia: ",coronavirusinindia.shape)

coronavirusinindia.head()
covid19 = pd.read_excel('/kaggle/input/covid19.xlsx')

print("Nocovid19. of tweets for #coronavirusinindia: ",covid19.shape)

covid19.head()