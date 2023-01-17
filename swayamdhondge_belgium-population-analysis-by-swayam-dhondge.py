# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        

df=pd.read_csv("../input/belgium-population-classification/BELGIUM_POPULATION_STRUCTURE_2018.csv",encoding='ISO-8859-1')





#df=pd.read_csv(r"D:\kaggle lernels\BELGIUM_POPULATION_STRUCTURE_2018.csv",encoding="ISO-8859-1")



df_hasselt=df[df['DISTRICT NAME'].str.contains('Hasselt')]

df_hasselt=df[df['DISTRICT NAME'].str.contains('Hasselt')]





print(df_hasselt.describe())









# Any results you write to the current directory are saved as output.


sns.distplot(df_hasselt['AGE'])

plt.title("BELGIUM DISTRICT:HASSELT POPULATION AGE DISTRIBUTION")