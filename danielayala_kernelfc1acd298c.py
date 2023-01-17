# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt # visualizations

import scipy.stats  # statistics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

desempleo = pd.read_csv("../input/desempleo1/desempleo2.csv",  sep=',',encoding='latin-1')

migracion = pd.read_csv("../input/migracin/migracion venezuela1.csv", sep=',',encoding='latin-1')

Tasa_de_incidencia = pd.read_csv("../input/tasa-de-incidencia/Tasa de incidencia de la pobreza1.csv",sep=',', encoding='latin-1')
desempleo.head()

Tasa_de_incidencia.head()
y= 9, 8.44, 8.72, 8.59, 8.74, 8.63, 9.72

x=2012,2013, 2014,2015,2016,2017,2018

z = -68988,-43839,-2182,39587,109124,408471
plt.plot(x,y)