# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Specify the paths of the CSV files to read

my_filepath1 = "../input/world-happiness/2015.csv"

my_filepath2 = "../input/world-happiness/2016.csv"

my_filepath3 = "../input/world-happiness/2017.csv"

my_filepath4 = "../input/world-happiness/2018.csv"

my_filepath5 = "../input/world-happiness/2019.csv"
# Fill in the line below: Read the file into a variable my_data

data2015 = pd.read_csv(my_filepath1)

data2016 = pd.read_csv(my_filepath2)

data2017 = pd.read_csv(my_filepath3)

data2018 = pd.read_csv(my_filepath4)

data2019 = pd.read_csv(my_filepath5)
#Add Year to each dataframe

data2015['Year']=2015

data2016['Year']=2016

data2017['Year']=2017

data2018['Year']=2018

data2019['Year']=2019



#Reduce df to contain interesting parts

data2015b=data2015.loc[:,['Country', 'Region', 'Happiness Rank', 'Happiness Score', 'Year']]

data2016b=data2016.loc[:,['Country', 'Region', 'Happiness Rank', 'Happiness Score', 'Year']]

data2017b=data2017.loc[:,['Country', 'Happiness.Rank', 'Happiness.Score', 'Year']]

data2018b=data2018.loc[:,['Country or region', 'Overall rank', 'Score', 'Year']]

data2019b=data2019.loc[:,['Country or region', 'Overall rank', 'Score', 'Year']]



#Use uniform column names for all years

data2015b.columns = ['Country', 'Region', 'Rank', 'Score', 'Year']

data2016b.columns = ['Country', 'Region', 'Rank', 'Score', 'Year']

data2017b.columns = ['Country', 'Rank', 'Score', 'Year']

data2018b.columns = ['Country', 'Rank', 'Score', 'Year']

data2019b.columns = ['Country', 'Rank', 'Score', 'Year']
#Create one df for all data

allyear = pd.concat([data2015b, data2016b, data2017b, data2018b, data2019b]).reset_index()

allyear.drop('index', axis=1, inplace=True)



#Fill in missing Region where applicable (some small groups have no region, so will be a few NaN left)

allyear['Region'] = allyear.groupby('Country').transform(lambda x: x.ffill().bfill())
#Prepare plot

fig, ax = plt.subplots()

fig.set_size_inches(18.5, 10.5)



#Plot average Score by Region per year for all data

g=sns.barplot(x=allyear['Region'], y=allyear['Score'], hue=allyear['Year'])

g.set_xticklabels(g.get_xticklabels(),rotation = 75, fontsize = 10, va='top', ha='right')