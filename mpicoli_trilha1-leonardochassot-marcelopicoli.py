# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





#This librarys is to work with matrices

import pandas as pd 

# This librarys is to work with vectors

import numpy as np

# This library is to create some graphics algorithmn

import seaborn as sns

# to render the graphs

import matplotlib.pyplot as plt

# import module to set some ploting parameters

from matplotlib import rcParams

# Library to work with Regular Expressions



# This function makes the plot directly on browser

%matplotlib inline

import seaborn as sns

# Seting a universal figure size 

rcParams['figure.figsize'] = 10,8

# Standard plotly imports

import plotly.graph_objs as go

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)
import re

dataset = pd.read_csv('../input/titanic.csv')

dataset.head()
dataset.info()
dataset.describe()
dataset['Sex'].iplot(kind='hist', xTitle='Gênero',yTitle='Número', title='Número de passageiros por genêro')
grupo_idade = [15,30,65,100]

categoria = {0:'Crianças (0 até 10 anos)',

             1: 'Jovens (10 até 30 anos)',

             2: 'Adultos (30 até 65 anos)',

             3: 'Idosos (Maiores que 65 anos)'}

dataset['Grupo_Idade'] = dataset['Age'].apply(lambda v: np.digitize(v, bins=grupo_idade))

dataset['Grupo_Idade'] = dataset['Grupo_Idade'].replace(categoria)

#dataset['Grupo_Idade'].value_counts()

dataset['Grupo_Idade'].iplot(kind='hist', xTitle='Gênero',yTitle='Número', title='Número de passageiros por classificação etária.')
c = dataset[['Age','Pclass']].groupby('Pclass').mean()

c.iplot(kind='bar')
dataset.boxplot('Age',by='Pclass',figsize=(50,10))
d = dataset[['Fare','Grupo_Idade']].groupby('Grupo_Idade').mean()

d
d.iplot(kind='bar')
e = dataset[['Fare','Pclass']].groupby('Pclass').mean()

e
e.iplot(kind='bar')
dataset[['Fare','Survived']].corr()
sns.set(rc={'figure.figsize':(10,5)})

p = sns.heatmap(dataset[['Fare','Survived']].corr())
dataset[['Survived','Sex']].groupby('Sex').mean()
corr = dataset[['Survived','Grupo_Idade']].groupby('Grupo_Idade').mean()

corr
corr.iplot(kind='bar')