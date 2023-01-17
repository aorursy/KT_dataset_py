# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pip

import Levenshtein as lev

import plotly.plotly as py

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from plotly.graph_objs import *

import matplotlib.pyplot as plt

import plotly.tools as tls

import plotly

plotly.offline.init_notebook_mode(connected=True) 
from plotly import __version__

print( __version__)
py.sign_in('kpschauhan92', 'dl8a5p0g70')
vgsales=pd.read_csv('../input/vgsales.csv')
def getting_series(name):

    #if type(name)!=str:

     #   return name

    #else:

    max_lev=0

    series_name=name

    for x in series:

        a=lev.ratio(x,name)

        if a>0.7 and a!=1 and a>max_lev :

            max_lev=a

            series_name=name

    return series_name

def refining_series(element):

    flag=0

    read_set.add(element)

    for x in series[series not in read_set]:

        if lev.ratio(x,element)> 80:

            flag=1

            return x

    if flag==0:

        return element

def splitter(x):

    #Function to remove the ':' and '/' parts from game names.

    if type(x)!=str:

        return x

    elif ':' in x:

        return remove_version(x.split(':')[0])

    elif '/' in x:

        return remove_version(x.split('/')[0])

    elif ' ' not in x:

        return x

    else:

        return remove_version(x)

def remove_version(x):

    #Function to remove the version number such as Black 2 or Grand theft auto V from game names.

    if x==None:

        return None

    elif type(x)!=str:

        return x

    else:

        a=x.split(' ')

        last_word=a[len(a)-1]

        if last_word.isdigit():

            return x[:-(len(last_word)+1)]

        elif last_word.endswith(('I','X','V')):

            return x[:-(len(last_word)+1)]

        else:

            return x
series=vgsales['Publisher'].apply(splitter).drop_duplicates().dropna()
def getting_publisher(name):

    max_lev=0

    pub_name=name

    for x in series[series!=name].sort_values():

        a=lev(x,name)

        if a>0.85 and a>max_lev:

            max_lev=a

            pub_name=x

        elif x in name and a>0.5 and a>max_lev:

            max_lev=a

            pub_name=x

    return pub_name

      
vgsales.iloc[:51]['Publisher'].apply(getting_publisher)