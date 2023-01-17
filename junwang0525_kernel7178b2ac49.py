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

import requests

import re

import numpy as np

#!conda install -c conda-forge beautifulsoup4 --yes

from bs4 import BeautifulSoup

import json # library to handle JSON files



from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe



#Matplotlib and associated plotting modules

import matplotlib.cm as cm

import matplotlib.colors as colors



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



#!conda install -c conda-forge geopy --yes

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values



# import k-means from clustering stage

from sklearn.cluster import KMeans



#!conda install -c conda-forge folium=0.5.0 --yes

import folium # map rendering library

from folium import plugins

print('Libraries imported.')
path1='https://www.doogal.co.uk/CountiesCSV.ashx?county=E11000009'

raw_data=pd.read_csv(path1,usecols=['Postcode','In Use?','Latitude','Longitude'])