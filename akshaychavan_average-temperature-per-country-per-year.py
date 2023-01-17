# Python Script

# coding: utf-8

# # Calculate the average temperature per year for every country
# Collect that into data frame where the 
# years take the index values & the
# countries take the column names

# In[ ]:

import pandas as pd
import numpy as np

# In[121]:

data = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')


# ## There are 2 ways the dt variable is formatted
# 1. YYYY-MM-DD
# 2. MM/DD/YYYY
# 
# ### Split the dt coulmn into 'year', 'month', and 'date' columns

# In[122]:

def splitDT(datadt):
    l1 = datadt.str.split('-').tolist();
    l2 = data.dt.str.split('/').tolist();

    l = [];
    for index in range(len(l1)):
        if( len(l1[index]) > len(l2[index]) ):
            l.append(l1[index]);
        else:
            elel2 = l2[index];
            elel2.insert(0, elel2.pop())
            l.append(elel2);
    return l;
            
ymd = pd.DataFrame( splitDT(data.dt), columns = ['year','month','date'] )


# ### Concat with the original data

# In[123]:

data = pd.concat([ymd, data], axis=1)


# ### Unique Countries

# In[124]:

uCountry = data.Country.unique()
len(uCountry)


# ### Unique Years

# In[125]:

uYear = data.year.unique()
len(uYear)


# ## Create a dataframe with 
# - a column 'year'
# - one column for each country
#   with average temp for each year across it

# In[126]:

uCountry = np.insert(uCountry, 0, 'year')
matdf = pd.DataFrame(columns=uCountry)
matdf.year = uYear[len(uYear)-14:len(uYear)]
matdf = matdf.set_index('year')
matdf.describe()


# ### Loop through every country and find the average temperature from the data given for that country

# This loop is very slow. 
# 
# I am pretty new to **pandas**.
# 
# *Would be happy to get suggestions on how calculate such a matrix efficiently.*

# In[127]:

for country in uCountry:
    avgTemp = []
    for ind in range(len(uYear)-14,len(uYear)):
        mCY = data.AverageTemperature[(data.Country == country) & (data.year == uYear[ind] )].mean()
        avgTemp.append(mCY)
    matdf[country] = avgTemp


# In[128]:

matdf.tail()


# In[129]:

matdf.to_csv('matYearCountry.csv')


