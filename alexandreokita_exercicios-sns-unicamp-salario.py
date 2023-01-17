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
#Setup

import seaborn as sns

from matplotlib import pyplot as plt
#Sintaxe: sns.countplot(x = data_set['coluna'], data = data_set)



#Sintaxe sns.countplot(x = data_set['coluna'], data = data_set, order = data_set['coluna'].value_counts()...)
#Sintaxe: sns.swarmplot(x = data_set['coluna'], y = data_set['coluna2'])



#Sintaxe: sns.scatterplot(x = data_set['coluna'], y = data_set['coluna2'])



#sintaxe: sns.regplot(x = data_set['coluna'], y = data_set['coluna2'])



#sintaxe: sns.lmplot(data = data_set, x = 'coluna', y = 'coluna2', hue = 'coluna3')


