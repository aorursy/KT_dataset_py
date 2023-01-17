# Incluímos las librerías necesarias para el ejercicio

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



pd.set_option('display.max_columns', None)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Leemos el dataset

df = pd.read_csv("../input/movie_metadata.csv");



# Comprobamos que se ha cargado correctamente

df.head()
# Info sobre el campo movie_facebook_likes

df['movie_facebook_likes'].describe()
x = [] # [Completar código]

y = [] # [Completar código]



# Scatterplot del rating en imdb en función de los likes en fb

# [Completar código]



# Histograma de imdb_score

# [Completar código]