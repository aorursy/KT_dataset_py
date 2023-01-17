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
# Importing packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 
# After uploading the file, we are able to see the name of the file saved as "World_Happiness_2015_2017.csv"

# Use pd.read_csv() to read the file and assign it to variable call "data"

data = pd.read_csv('/kaggle/input/worldhapinees20152017/World_Happiness_2015_2017_.csv')



# We then use data.head() to see the first 5 rows of data

data.head()
# Then what I do next is look into shape using data.shape(). This will tell me how many rows and columns there are.

data.shape
# Now lets see data types using data.dtypes

data.dtypes
# Lets calculate the number of null values

data.isnull().sum()
g = sns.pairplot(data)

g.fig.suptitle('FacetGrid plot', fontsize = 20)

g.fig.subplots_adjust(top = 0.9);
# Creating a list of attributes we want (just copy the column name)

econ_happiness = ['Happiness Score','Economy (GDP per Capita)']



# Creating a dataframe that only contains these attributes

econ_corr = data[econ_happiness]



# Finding correlation

econ_corr.corr()
sns.regplot(data = econ_corr, x = 'Economy (GDP per Capita)', y = 'Happiness Score').set_title("Correlation graph for Happiness score vs Economy")
# Creating a list of attributes we want (just copy the column name)

econ_family = ['Happiness Score','Family']



# Creating a dataframe that only contains these attributes

econ_corr = data[econ_family]



# Finding correlation

econ_corr.corr()
sns.regplot(data = econ_corr, x = 'Family', y = 'Happiness Score').set_title("Correlation graph for Happiness score vs Family")
# Creating a list of attributes we want (just copy the column name)

econ_health = ['Happiness Score','Health (Life Expectancy)']



# Creating a dataframe that only contains these attributes

econ_corr = data[econ_health]



# Finding correlation

econ_corr.corr()
sns.regplot(data = econ_corr, x = 'Health (Life Expectancy)', y = 'Happiness Score').set_title("Correlation graph for Happiness score vs Health()")
econ_freedom = ['Happiness Score','Freedom']



# Creating a dataframe that only contains these attributes

econ_corr = data[econ_freedom]



# Finding correlation

econ_corr.corr()
sns.regplot(data = econ_corr, x = 'Freedom', y = 'Happiness Score').set_title("Correlation graph for Happiness score vs Freedom")
# Creating a list of attributes we want (just copy the column name)

econ_trust = ['Happiness Score','Trust (Government Corruption)']



# Creating a dataframe that only contains these attributes

econ_corr = data[econ_trust]



# Finding correlation

econ_corr.corr()
sns.regplot(data = econ_corr, x = 'Trust (Government Corruption)', y = 'Happiness Score').set_title("Correlation graph for Happiness score vs Trust (Government Corruption)")
# Creating a list of attributes we want (just copy the column name)

econ_generosity = ['Happiness Score','Generosity']



# Creating a dataframe that only contains these attributes

econ_corr = data[econ_generosity]



# Finding correlation

econ_corr.corr()
sns.regplot(data = econ_corr, x = 'Generosity', y = 'Happiness Score').set_title("Correlation graph for Happiness score vs Generosity")
# Creating a list of attributes we want (just copy the column name)

econ_dystopia = ['Happiness Score','Dystopia Residual']



# Creating a dataframe that only contains these attributes

econ_corr = data[econ_dystopia]



# Finding correlation

econ_corr.corr()
sns.regplot(data = econ_corr, x = 'Dystopia Residual', y = 'Happiness Score').set_title("Correlation graph for Happiness score vs Dystopia")