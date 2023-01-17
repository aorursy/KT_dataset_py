import pandas as pd

import matplotlib.pyplot as plt

import plotly

plotly.offline.init_notebook_mode()

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import numpy as np

import seaborn as sns

import calendar

%matplotlib inline
hepatitis = pd.read_csv('../input/hepatitis.csv')

measles = pd.read_csv('../input/measles.csv')

mumps = pd.read_csv('../input/mumps.csv')

pertussis = pd.read_csv('../input/pertussis.csv')

polio = pd.read_csv('../input/polio.csv')

rubella = pd.read_csv('../input/rubella.csv')

smallpox = pd.read_csv('../input/smallpox.csv')



disease = hepatitis.append(measles)

disease = disease.append(mumps)

disease = disease.append(pertussis)

disease = disease.append(polio)

disease = disease.append(rubella)

disease = disease.append(smallpox)
year  = [int(str(w)[0:4]) for w in disease['week']]

month = [int(str(w)[4:6]) for w in disease['week']]



disease.insert(0,'year',year)

disease.insert(1,'month',month)

disease = disease.drop('week',axis=1)



disease.head()
usa_pop = [[2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000, 1999, 1998, 

           1997, 1996, 1995, 1994, 1993, 1992, 1991, 1990, 1989, 1988, 1987, 1986, 1985, 1984, 

           1983, 1982, 1981, 1990, 1979, 1978, 1977, 1976, 1975, 1974, 1973, 1972, 1971, 1970, 

           1969, 1968, 1967, 1966, 1965, 1964, 1963, 1962, 1961, 1960, 1959, 1958, 1957, 1956, 

           1955, 1954, 1953, 1952, 1951, 1950, 1949, 1948, 1947, 1946, 1945, 1944, 1943, 1942, 

           1941, 1940, 1939, 1938, 1937, 1936, 1935, 1934, 1933, 1932, 1931, 1930, 1929, 1928],

           [310500000, 308110000, 306770000, 304090000, 301230000, 298380000, 295520000, 292810000, 

            290110000, 287630000, 284970000, 282160000, 279040000, 275850000, 272650000, 269390000, 

            266280000, 263130000, 259920000, 256510000, 252980000, 249620000, 246820000, 244500000, 

            242290000, 240130000, 237920000, 235820000, 233790000, 231660000, 229470000, 227220000, 

            225060000, 222580000, 220240000, 218040000, 215970000, 213850000, 211910000, 209900000, 

            207660000, 205050000, 202680000, 200710000, 198710000, 196560000, 194300000, 191890000, 

            189240000, 186540000, 183690000, 180670000, 177830000, 174880000, 171990000, 168900000, 

            165930000, 163030000, 160180000, 157550000, 154880000, 152270000, 149190000, 146630000, 

            144130000, 141390000, 139930000, 138400000, 136740000, 134860000, 133400000, 132120000, 

            130880000, 129820000, 128820000, 128050000, 127250000, 126370000, 125580000, 124840000, 

            124040000, 123080000, 121770000, 120510000]]


