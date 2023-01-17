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
data = pd.read_csv("../input/countries of the world.csv")  #getting the data with pandas
data.info()  # learning more about the data
data.head(10)
data.describe()
def comma_to_dot(data):
    """This function is created for the replacement of comma with dot and changing the data type to float"""
    data = str(data);   # for any type of unexpected data types
    data = data.replace(",",".");
    data = float(data);
    return data;
print(data.columns)    # checking the full names of each column
data['Pop. Density (per sq. mi.)'] = list(map(comma_to_dot,data['Pop. Density (per sq. mi.)']))
data['Coastline (coast/area ratio)'] = list(map(comma_to_dot,data['Coastline (coast/area ratio)']))
data['Net migration'] = list(map(comma_to_dot,data['Net migration']))
data['Infant mortality (per 1000 births)'] = list(map(comma_to_dot,data['Infant mortality (per 1000 births)']))
data['GDP ($ per capita)'] = list(map(comma_to_dot,data['GDP ($ per capita)']))
data['Literacy (%)'] = list(map(comma_to_dot,data['Literacy (%)']))
data['Phones (per 1000)'] = list(map(comma_to_dot,data['Phones (per 1000)']))
data['Arable (%)'] = list(map(comma_to_dot,data['Arable (%)']))
data['Crops (%)'] = list(map(comma_to_dot,data['Crops (%)']))
data['Other (%)'] = list(map(comma_to_dot,data['Other (%)']))
data['Climate'] = list(map(comma_to_dot,data['Climate']))
data['Birthrate'] = list(map(comma_to_dot,data['Birthrate']))
data['Deathrate'] = list(map(comma_to_dot,data['Deathrate']))
data['Agriculture'] = list(map(comma_to_dot,data['Agriculture']))
data['Industry'] = list(map(comma_to_dot,data['Industry']))
data['Service'] = list(map(comma_to_dot,data['Service']))
data.head(10)   # to see results on dataset
data.describe()
data.info();
data.Climate.value_counts(dropna=False)   
# it is interesting that 2.0 and 3.0 is most common but the 2.5 is least common one even that it's just between most common values.
# also we have a lot of NaN values.
data = data.dropna();

data.info();   # we still have 180 countries that we can work on easily.  
data.boxplot(figsize=(20,8), column='Literacy (%)', by='GDP ($ per capita)', grid=False)
import seaborn as sns
import matplotlib.pyplot as plt
plt.clf()
plt.figure(figsize=(20,12))
sns.heatmap(data.corr(),annot=True,fmt='1.1f')
b = set(data.Region);
print(b)   # seems like we have a Lot of spaces, considering that they use storage unnecesarily: I want to delete whole spaces.
def SpaceRemover(data):
    """This function is created for the replacement of Space with empty"""
    data = str(data);   # for any type of unexpected data types
    data = data.replace(" ","");
    return data;
data['Region'] = list(map(SpaceRemover,data['Region']))
print(set(data.Region))  # now not super clear but still better.
plt.clf()
plt.figure(figsize=(15,10));
plt.scatter(data.Climate , data.Region, alpha=0.1, s=200, c="blue")  

# I made 0.1 opacity to see how often it repeats easily
# darker the blue, often the appearence of that climate.

plt.show()
melted_data = pd.melt(frame=data, id_vars="Country", value_vars=['Literacy (%)','Birthrate','Phones (per 1000)'])

melted_data  # the picked variables are all correlated, so when I pivot it will has more sense.
pivotted_melt = melted_data.pivot(index='Country',columns="variable",values='value')

pivotted_melt  # we can see that how dependently numbers change in each column easily.
# assert pivotted_melt.Birthrate.dtype == np.dtype(int)   ## would return an error.

assert pivotted_melt.Birthrate.dtype == np.dtype(float)   # returns no error because data type is really float
# assert data.columns[1] == "Efe"  ## would return an error 

assert data.columns[1] != "Efe" # in here I assert negatively. (not "Efe" is true)
dataHead = data.head(10);               dataTail = data.tail(10);

concated_data = pd.concat([dataHead,dataTail],axis=0,ignore_index=1)
concated_data   # Rigth Below we see that by "concat" operation of numpy we easliy created new data from 2 data(s). 
# this notebook is created for the 3rd homework of Data Science by DATAI on Udemy.
