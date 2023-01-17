import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import plotly.colors

import datetime

from statistics import mean
# Read csv file
df = pd.read_csv('../input/windows-store/msft.csv')
df.head(2)
# Check missing values
df.isnull().sum()
# Remove rows with null values
df.dropna(inplace=True)
# Manipulating data in Price column: replacing free with 0 and changing type to integer
df.Price = df.Price.str.replace('Free', '0')
df.Price = df.Price.str.replace(',', '')
df.Price = df.Price.str.replace('₹', '')
df.Price = df.Price.astype(float)
# Adding a column to dataframe
df['Free/Paid'] = np.where(df['Price'] == 0.0, 'Free', 'Paid')
df.head(2)
# Preparing data for bar Chart
data = pd.DataFrame(df.Category.value_counts())
data.reset_index(inplace=True)
data.rename(columns={'index':'Category', 'Category':'Count'}, inplace = True,)

# Plot bar graph
fig = px.bar(data, x='Count', y='Category', color='Count',
             title='Cateogories of App in Windows Store')
fig.show()
data = pd.DataFrame(df['Free/Paid'].value_counts())
data.reset_index(inplace=True)
data.rename(columns={'index':'Free/Paid', 'Free/Paid':'Count'}, inplace = True,)

fig = px.pie(data, values='Count', names='Free/Paid', 
             title='Free and Paid App in Windows Store', 
             color_discrete_sequence=px.colors.sequential.Rainbow[3:5])
fig.show()
data = df
data = data.groupby(['Category', 'Free/Paid'], as_index=False).size()

fig = px.bar(data, x='Category', y='size', color='Free/Paid', title='Free and Paid App in Each Category')
fig.show()
print('Average rating:', round(mean(df.Rating),3))
fig = px.histogram(df.Rating, x='Rating', title='Overall Ratings')
fig.show()
fig = px.box(df, x='Category', y='Rating', color='Category', title='Rating for Each Category')
fig.update_traces(quartilemethod='linear')
fig.show()
# Plot scatter plot
fig = px.scatter(df, x='Price', y='Rating', title='Rating and Price Trend')
fig.show()
fig = px.scatter(df, x='Category', y='Price', color='Category', title='Price for Each Category')
fig.show()
data = df.corr()
fig = px.imshow(data, title='Heatmap')
fig.show()