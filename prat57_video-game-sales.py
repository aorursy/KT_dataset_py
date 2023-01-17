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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

import seaborn as sns

warnings.filterwarnings("ignore")



vg_sales = pd.read_csv('../input/videogamesales/vgsales.csv')

print(vg_sales.head())
years =  [2016,2017,2020]

total_sales_group =  vg_sales.groupby(['Year']).sum().drop(years)

average_sales_group = vg_sales.groupby(['Year']).mean().drop(years)

count_sales_group = vg_sales.replace(0,np.nan).groupby(['Year']).count().drop(years)
def lineplot(df, title = 'Sales by year' , ylabel = 'Sales', legendsize = 10

            , legendloc = 'upperleft'):

    year = df.index.values

    na = df.NA_Sales

    eu = df.EU_Sales

    jp = df.JP_Sales

    other = df.Other_Sales

    global_ = df.Global_Sales

    

    if df is count_sales_group:

        region_list = [na,eu,jp,other]

        columns = ['NA','EU','JP','OTHER']

    else:

        region_list = [na,eu,jp,other,global_]

        columns = ['NA','EU','JP','OTHER','WORLDWIDE']

        

        for i, region in enumerate(region_list):

            plt.plot(year, region, label = columns[i])

            

        plt.ylabel(ylabel)

        plt.xlabel('Year')

        plt.title(title)

        plt.legend(loc=legendloc, prop = {'size':legendsize})

        plt.show()

        plt.clf()

        

        for i, region in enumerate(region_list):

            plt.plot(year, region, label = columns[i])

            

        plt.yscale('log')

        plt.ylabel(ylabel)

        plt.xlabel('Year')

        plt.title(title + '(Log)')

        plt.legend(loc=legendloc, prop={'size':legendsize})

        plt.show()

        plt.clf()

lineplot(total_sales_group, title = 'Sales by Year', ylabel = 'Sales(In Millions)', legendsize=8)

japan1992_1996 = vg_sales[['Name', 'JP_Sales', 'NA_Sales']][(vg_sales.Year>=1992) & (vg_sales.Year<=1996) & (vg_sales.JP_Sales > vg_sales.NA_Sales)].sort_values(['JP_Sales'], ascending = False)

print(japan1992_1996.head(20))
lineplot(average_sales_group, title='Average Revenue Per Game Per Year', ylabel = 'Sales (In Millions)', legendsize=8, legendloc='upper right')

lineplot(count_sales_group, title = 'Of All Games Produced Did Every Region Sell Every Game?\n\nGames for Sale by Region by Year', ylabel ='Count', legendsize = 8, legendloc = 'upper left')
Top_games= vg_sales[['Name','Year','Global_Sales']].sort_values(['Global_Sales'], ascending=False)

print(Top_games.head(20))