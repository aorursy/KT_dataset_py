#importing libraries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
#getting the dataset 



data = pd.read_csv('../input/heatmap.csv')

print(data.head(3))
#creating dataframe with required columns

df1 = data[['continent', 'year','lifeExp']]

print(df1.head())

#Pandas’ pivot_table function to spread the data from long form to tidy form

#To reshape the data such that take continent as rows and year on columns,specify index and columns variables accordingly

heatmap1_data = pd.pivot_table(df1, values='lifeExp', 

                     index=['continent'], 

                     columns='year')
#Plot heatmap and use color palette with “cmap” 

sns.heatmap(heatmap1_data, cmap="YlGnBu")