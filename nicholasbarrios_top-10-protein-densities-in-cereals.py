import pandas as pd               #for importing and manipulating data

import matplotlib.pyplot as plt   #for plotting data

import plotly.express as px       #for creating interactive charts
df = pd.read_csv("../input/80-cereals/cereal.csv") #Pandas allows a csv file to be read as a dataframe
df.head() #observing the head of the dataframe
df.shape #observing the shape of the dataframe
df.isnull().sum() #searching for null values in the dataset
df['prot_per_cal'] = df.protein / df.calories #following our definition of protein density 
df = df.sort_values(by = 'prot_per_cal', ascending = 1) #sorting the data makes the visualization less messy
plt.bar(df.name[-10:], df.prot_per_cal[-10:], #graphs only the 10 cereals with the highest protein density

        align='center', 

        alpha=.75)

plt.ylabel('g Protein per Cal')

plt.title('Protein Density of Cereals')

plt.xticks(rotation=40, ha='right')



plt.show()
fig = px.bar(df[-10:], x='name', y='prot_per_cal',

             hover_data=['fat', 'carbo'],               # displays fat and carbohydrate content when hovering the mouse over the bar

             labels={'prot_per_cal' : 'Protein (g) per Calorie', 'name':''},

             height=650)

fig.update_layout(

    title='Protein Density of Cereals',

    width=900)

fig.show()