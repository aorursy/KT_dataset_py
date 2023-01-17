# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/data.csv')

#print(df)

print("Top 10 Clubs in FIFA19 on the basis of Overall rating")

print("Real Madrid:",df.loc[df['Club']=='Real Madrid','Overall'].sum())

print("Barcelona:",df.loc[df['Club']=='FC Barcelona','Overall'].sum())

print("Manchester United:",df.loc[df['Club']=='Manchester United','Overall'].sum())

print("Chelsea:",df.loc[df['Club']=='Chelsea','Overall'].sum())

print("Manchester City:",df.loc[df['Club']=='Manchester City','Overall'].sum())

print("Tottenham Hotspur:",df.loc[df['Club']=='Tottenham Hotspur','Overall'].sum())

print("Liverpool:",df.loc[df['Club']=='Liverpool','Overall'].sum())

print("Borussia Dortmund:",df.loc[df['Club']=='Borussia Dortmund','Overall'].sum())

print("Arsenal",df.loc[df['Club']=='Arsenal','Overall'].sum())

print("Atlético Madrid:",df.loc[df['Club']=='Atlético Madrid','Overall'].sum())

print("Paris Saint-Germain:",df.loc[df['Club']=='Paris Saint-Germain','Overall'].sum())

print("Lazio:",df.loc[df['Club']=='Lazio','Overall'].sum())












