import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("darkgrid")



%matplotlib inline
data = pd.read_csv("../input/index.csv")

data.head()
data.isnull().sum()
inspect_inflation = data.groupby('Year')['Inflation Rate'].mean().to_frame().reset_index()

inspect_inflation.columns = ['Year', 'Mean_inflation']



inspect_inflation.head()
plt.plot(inspect_inflation['Year'], inspect_inflation['Mean_inflation'])
data['Inflation Rate'].isnull().sum()
#Looking at the plot above I think it will be a safe assumption to replace Nan of 1954-1957 to be around 3. 

data.loc[data['Year'] < 1958, 'Inflation Rate'] = 3.00
data['Inflation Rate'].isnull().sum()
Inflationrate_lookup = data.groupby('Year')['Inflation Rate'].mean().to_frame().reset_index()

Inflationrate_lookup.columns = ['Year', 'Mean_inflation']
Inflationrate_lookup.head()
def impute_Inflationrate(row):

    #You need to impute the row, takeing statistics from dataframe.

    global Inflationrate_lookup

    return_value = row['Inflation Rate']

    if(np.isnan(return_value)):

        return_value = Inflationrate_lookup.loc[Inflationrate_lookup.Year == row['Year'], "Mean_inflation"]

    return(return_value)

    

    

    

data['Inflation Rate'] = data.apply(lambda row: impute_Inflationrate(row) , axis = 1)

data['Inflation Rate'] = data['Inflation Rate'].astype("float64")

inspect_unemployment = data.groupby('Year')['Unemployment Rate'].mean().to_frame().reset_index()

inspect_unemployment.columns = ['Year', 'Mean_Unemployment']

plt.plot(inspect_unemployment['Year'], inspect_unemployment['Mean_Unemployment'], color = "red")
Unemploymentrate_lookup = data.groupby('Year')['Unemployment Rate'].mean().to_frame().reset_index()

Unemploymentrate_lookup.columns = ['Year', 'Mean_Unemployment']



data['Unemployment Rate'].isnull().sum()
def impute_Unemploymentrate(row):

    #You need to impute the row, takeing statistics from dataframe.

    global Unemploymentrate_lookup

    return_value = row['Unemployment Rate']

    if(np.isnan(return_value)):

        return_value = Unemploymentrate_lookup.loc[Unemploymentrate_lookup.Year == row['Year'], "Mean_Unemployment"]

    return(return_value)

    

    

    

data['Unemployment Rate'] = data.apply(lambda row: impute_Unemploymentrate(row) , axis = 1)

data['Unemployment Rate'] = data['Unemployment Rate'].astype("float64")

data['Unemployment Rate'].isnull().sum()
inspect_inflation = data.groupby('Year')['Inflation Rate'].mean().to_frame().reset_index()

inspect_inflation.columns = ['Year', 'Mean_inflation']



inspect_unemployment = data.groupby('Year')['Unemployment Rate'].mean().to_frame().reset_index()

inspect_unemployment.columns = ['Year', 'Mean_Unemployment']

#Doing the same as above. 

import matplotlib.patches as mpatches



plt.plot(inspect_unemployment['Year'], inspect_unemployment['Mean_Unemployment'])

blue_patch = mpatches.Patch(color='blue', label='Unemployment')





plt.plot(inspect_inflation['Year'], inspect_inflation['Mean_inflation'], color = "red")

red_patch = mpatches.Patch(color='red', label='Inflation')

plt.legend(handles=[blue_patch, red_patch])
