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

import matplotlib.pyplot as plt  

import statistics

import seaborn as sns; sns.set()

import matplotlib as plt

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from pandas.plotting import register_matplotlib_converters
#Selecting darkgrid style from seaborn library

sns.set_style("darkgrid")

plt.rcParams['figure.figsize'] = [15, 10]

register_matplotlib_converters()
#Read the .csv file 

gdpr_fines = pd.read_csv("/kaggle/input/gdpr-fines-in-eu-20182019-120-rows-8-columns/gdpr_fines.csv", index_col=0)

gdpr_fines.head()
#Looking at different attirbutes of the dataset: 
gdpr_fines.info()
#Looking at the shape of dataset: 121 rows, 7 columns (note! Python counts the first columns as 0)

gdpr_fines.shape
gdpr_fines = gdpr_fines.drop_duplicates()

gdpr_fines.shape
gdpr_fines.drop_duplicates(inplace=True)

gdpr_fines.shape
gdpr_fines.columns
#Cleaning the columns 

gdpr_fines.rename(columns={

    'Authority': 'authority',

    'Country': 'country',

    'Date': 'date', 'Fine [€]': 'fine', 'Controller/Processor': 'controller/processor', 'Quoted Art.': 'quoted article', 'Type': 'type', 'Infos': 'infos'

}, inplace=True)

gdpr_fines["date"] = gdpr_fines["date"].apply(lambda x: x.replace(",",""))

gdpr_fines["fine"] = gdpr_fines["fine"].apply(lambda x: x.replace(",",""))



gdpr_fines.columns
gdpr_fines.isnull().sum()
gdpr_fines.head(10)
fines = gdpr_fines['fine']

fines = fines.apply(lambda x: x.replace(",",""))

fines = fines.replace("Unknown", np.nan)

fines = pd.to_numeric(fines)
gdpr_fines.describe()
fines.mean()
import statistics

print(statistics.median(gdpr_fines["fine"]))
##Using describe() on an entire DataFrame we can get a summary of the distribution of continuous variables:

gdpr_fines.describe()
#Drop values that don't match the datetime format

gdpr_fines['date'] = pd.to_datetime(gdpr_fines['date'], errors='coerce')

gdpr_fines = gdpr_fines.dropna(subset=['date'])

gdpr_fines.describe()
#Drop the unknown values 

gdpr_fines = gdpr_fines[gdpr_fines["date"] != "Unknown"]

gdpr_fines = gdpr_fines[gdpr_fines["fine"] != "Unknown"]

gdpr_fines = gdpr_fines[gdpr_fines["quoted article"] != "Unknown"]
gdpr_fines["fine"] = pd.to_numeric(gdpr_fines["fine"])

gdpr_fines['date'] = gdpr_fines['date'].map(lambda date: pd.to_datetime(date, format="%Y.%m.%d"))

fines = gdpr_fines['fine']

fines
#.value_counts() can tell us the frequency of all values in a column:

gdpr_fines['fine'].value_counts().head(10)
gdpr_fines.describe()
date_and_fines = gdpr_fines.groupby(["date"]).sum()

date_and_fines.head()
ax1 = sns.lineplot(data=date_and_fines)

ax1.set(xlabel='Date when fine was given', ylabel='Amount of given fines')
article_and_date = gdpr_fines.groupby(["quoted article"]).sum()

article_and_date.sort_values(['fine'], ascending=[0], inplace=True)

article_and_date.head(10)
fine_by_country = gdpr_fines["fine"].groupby(["Country"]).count()

fine_by_country.head(10)