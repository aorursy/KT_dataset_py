# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.plotly as py

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

dataset = pd.read_csv("../input/Video_Game_Sales_as_of_Jan_2017.csv")

df = pd.DataFrame(dataset)





salesRating = df[['Platform', 'Global_Sales', 'Critic_Score']]

platforms = salesRating.Platform.unique()

totalSales = salesRating[['Platform','Global_Sales']]

sumSales = totalSales.groupby(['Platform']).sum()

print(sumSales.sort_values(['Global_Sales'], ascending=False))
nPS2 = df[(df.Platform=='PS2')].size

nX360 = df[(df.Platform=='X360')].size

nPS3 = df[(df.Platform=='PS3')].size

nWii = df[(df.Platform=='Wii')].size

nDS = df[(df.Platform=='DS')].size



y = [nPS2, nX360, nPS3, nWii, nDS]

N = len(y) + 2000

x = ["PS2", "Xbox360", "PS3", "Wii", "DS"]

width = 1/1.5

plt.bar(x,y,width,color="green")
highest_selling = df[np.isfinite(df['Critic_Score'])].head(10)

highest_selling = highest_selling[['Name', 'Platform','Year_of_Release', 'Critic_Score', 'Global_Sales']]

highest_selling = highest_selling.sort_values(['Global_Sales'],ascending=False)

highest_selling


highest_selling = df[np.isfinite(df['Critic_Score'])].head(100)

highest_selling = highest_selling[['Name', 'Platform','Year_of_Release', 'Critic_Score', 'Global_Sales']]

highest_selling = highest_selling.sort_values(['Global_Sales'],ascending=False)

highest_selling[highest_selling.Year_of_Release>2010].head(10)

genres = df[['Genre', 'Global_Sales']]

genres.groupby(['Genre']).count().sort_values('Global_Sales', ascending=False)