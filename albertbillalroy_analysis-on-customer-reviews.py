import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import matplotlib.pyplot as plt 

from wordcloud import WordCloud, STOPWORDS
df = pd.read_csv('../input/customer-review/Customer Review.csv')
df.head()
droplist = ['RetailerZip', 'RetailerState', 'ProductOnSale', 'ManufacturerRebate']



df = df.drop(columns=droplist)
df['UserOccupation'] = np.where(df['UserOccupation'].str.contains('Unknown'), np.nan, df['UserOccupation'])
df['ReviewDate'] = pd.to_datetime(df['ReviewDate'])
fig = px.bar(df['ProductModelName'].value_counts(), x=(df['ProductModelName'].value_counts()).index, y=(df['ProductModelName'].value_counts()).values, 

             color=(df['ProductModelName'].value_counts()).index)

fig.update_layout(title_text="Number of product",

                 xaxis_title="Name of product",

                 yaxis_title="Count")

fig.show()
fig = px.histogram(df, x="ReviewRating", histnorm='probability density')

fig.update_layout(title_text="Rating distribution",

                 xaxis_title="Rating",

                 yaxis_title="Count")

fig.show()
fig = px.histogram(df['ReviewDate'].value_counts(), x=(df['ReviewDate'].value_counts()).index, y=(df['ReviewDate'].value_counts()).values)

fig.update_layout(title_text="Review distribution",

                 xaxis_title="Year",

                 yaxis_title="Count")

fig.show()
stopwords = set(STOPWORDS)



comment_words = '' 



for val in df.ReviewText: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    comment_words += " ".join(tokens)+" "

  

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words)
# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
fig = px.pie(df['ProductCategory'].value_counts(), names=(df['ProductCategory'].value_counts()).index, values=(df['ProductCategory'].value_counts()).values)

fig.update_layout(title_text="Percentage of Number of product each category ")

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
fig = px.bar(df.groupby('ProductCategory')['ReviewRating'].mean().sort_values(ascending=True), y=(df.groupby('ProductCategory')['ReviewRating'].mean()).index, x=(df.groupby('ProductCategory')['ReviewRating'].mean()).values, 

             orientation='h', color=(df.groupby('ProductCategory')['ReviewRating'].mean()).index)

fig.update_layout(title_text="Average rating each category",

                 xaxis_title="Rating",

                 yaxis_title="Category")

fig.show()
fig = px.histogram(df, x="ProductPrice")

fig.update_layout(title_text="Price distribution",

                 xaxis_title="Price",

                 yaxis_title="Count")

fig.show()
fig = px.bar(df.groupby('RetailerName')['ProductModelName'].count(), x=(df.groupby('RetailerName')['ProductModelName'].count()).index, y=(df.groupby('RetailerName')['ProductModelName'].count()).values, 

             color=(df.groupby('RetailerName')['ProductModelName'].count()).index)

fig.update_layout(title_text="Number of product each retail",

                 xaxis_title="Retail",

                 yaxis_title="Count")

fig.show()
fig = px.bar(df.groupby('RetailerName')['ReviewRating'].mean().sort_values(ascending=True), y=(df.groupby('RetailerName')['ReviewRating'].mean()).index, x=(df.groupby('RetailerName')['ReviewRating'].mean()).values, 

             orientation='h', color=(df.groupby('RetailerName')['ReviewRating'].mean()).index)

fig.update_layout(title_text="Average rating each retail",

                 xaxis_title="Rating",

                 yaxis_title="Retail")

fig.show()
fig = px.bar(df.groupby('RetailerCity')['ProductModelName'].count(), x=(df.groupby('RetailerCity')['ProductModelName'].count()).index, y=(df.groupby('RetailerCity')['ProductModelName'].count()).values, 

             color=(df.groupby('RetailerCity')['ProductModelName'].count()).index)

fig.update_layout(title_text="Number of product each city",

                 xaxis_title="City",

                 yaxis_title="Count")

fig.show()
fig = px.bar(df.groupby('RetailerCity')['ReviewRating'].mean().sort_values(ascending=True), y=(df.groupby('RetailerCity')['ReviewRating'].mean()).index, x=(df.groupby('RetailerCity')['ReviewRating'].mean()).values, 

             orientation='h', color=(df.groupby('RetailerCity')['ReviewRating'].mean()).index)

fig.update_layout(title_text="Average rating each city",

                 xaxis_title="Rating",

                 yaxis_title="City")

fig.show()
fig = px.bar(df.groupby('ManufacturerName')['ProductModelName'].count(), x=(df.groupby('ManufacturerName')['ProductModelName'].count()).index, y=(df.groupby('ManufacturerName')['ProductModelName'].count()).values, 

             color=(df.groupby('ManufacturerName')['ProductModelName'].count()).index)

fig.update_layout(title_text="Number of product each manufacturer",

                 xaxis_title="Manufacturer",

                 yaxis_title="Count")

fig.show()
fig = px.bar(df.groupby('ManufacturerName')['ReviewRating'].mean().sort_values(ascending=True), y=(df.groupby('ManufacturerName')['ReviewRating'].mean()).index, x=(df.groupby('ManufacturerName')['ReviewRating'].mean()).values, 

             orientation='h', color=(df.groupby('ManufacturerName')['ReviewRating'].mean()).index)

fig.update_layout(title_text="Average rating each manufacturer",

                 xaxis_title="Rating",

                 yaxis_title="Manufacturer")

fig.show()
fig = px.histogram(df, x="UserAge")

fig.update_layout(title_text="Age distribution",

                 xaxis_title="Age",

                 yaxis_title="Count")

fig.show()
fig = px.histogram(df, x="UserGender")

fig.update_layout(title_text="Gender distribution",

                 xaxis_title="Gender",

                 yaxis_title="Count")

fig.show()
fig = px.pie(df['UserOccupation'].value_counts(), names=(df['UserOccupation'].value_counts()).index, values=(df['UserOccupation'].value_counts()).values)

fig.update_layout(title_text="Percentation each occupation")

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()