# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline




import os
print(os.listdir("../input"))



df = pd.read_csv('../input/zomato.csv',encoding="ISO-8859-1")
country = pd.read_excel('../input/Country-Code.xlsx')
data = pd.merge(df, country, on='Country Code')
data.head()

data.describe()
data.info()
#groupby country code
data_country = data.groupby(['Country'], as_index=False).count()[['Country', 'Restaurant ID']]
data_country.head()
data_country.columns = ['Country', 'No of Restaurant']
# No of restaurant that are on zomato in each country
plt.bar(data_country['Country'], data_country['No of Restaurant'])
plt.xlabel('Country', fontsize=20)
plt.rcParams['figure.figsize']=(30,20) 
plt.ylabel('No of Restaurant', fontsize=20)
plt.title('No of Restaurant', fontsize=30)
plt.xticks(rotation = 60)

data_City = data[data['Country'] =='India']
Total_city =data_City['City'].value_counts()
Total_city.plot.bar(figsize=(14, 7),fontsize=14)
plt.title('Restaurants by City', fontsize=30)                                             
plt.xlabel('City', fontsize=20)
plt.ylabel('No of Restaurants', fontsize=20)
plt.show()
data_Online = data[data['Has Online delivery'] =='Yes']
data_Online['Country'].value_counts().plot.bar(figsize=(14, 7), 
                                             fontsize=14)
plt.xlabel('Country', fontsize=20)
plt.ylabel('No of Restaurants', fontsize=20)
plt.title('Restaurants available for online orders', fontsize=30)
plt.show()
average_ratings = data.groupby(['Country'], as_index=False)
average_ratings_agg = average_ratings['Aggregate rating'].agg(np.mean)
plt.figure(figsize=(25,10))
plt.xlabel('Country', fontsize=20)
plt.ylabel('Average Ratings', fontsize=20)
plt.title('Average Ratings on Countries', fontsize=30)
plt.bar(average_ratings_agg['Country'], average_ratings_agg['Aggregate rating'])


Cuisine_data =data.groupby(['Cuisines'], as_index=False)['Restaurant ID'].count()
Cuisine_data.columns = ['Cuisines', 'Number of Resturants']
Top15= (Cuisine_data.sort_values(['Number of Resturants'],ascending=False)).head(15)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(Top15['Cuisines'], Top15['Number of Resturants'])
plt.xlabel('Cuisines', fontsize=20)
plt.ylabel('Number of Resturants', fontsize=20)
plt.title('Top 15 Cuisines on Zomato', fontsize=30)
plt.xticks(rotation = 90)
plt.show()


Cuisine_data_rating=(data.groupby(['Cuisines'], as_index=False)['Aggregate rating'].mean())
Cuisine_data_rating.columns = ['Cuisines', 'Rating']
Top30_ratings= (Cuisine_data_rating.sort_values(['Rating'],ascending=False)).head(30)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(Top30_ratings['Cuisines'], Top30_ratings['Rating'])
plt.title('Top Rated Cuisines on Zomato', fontsize=30)
plt.xlabel('Cuisines', fontsize=20)
plt.ylabel('Rating', fontsize=20)
plt.xticks(rotation = 90)
plt.show()
Cuisine_data_rating=(data.groupby(['Country'], as_index=False)['Price range'].mean())
Cuisine_data_rating.columns = ['Country', 'Price range' ]
Cuisine_data_rating.sort_values(['Price range'],ascending=False).head()
# Initialize the plotly
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import plotly.graph_objs as go


plot_data = [dict(
    type='scattergeo',
    lon = data['Longitude'],
    lat = data['Latitude'],
    text = data['Restaurant Name'],
    mode = 'markers',
    marker = dict(
    cmin = 0,
    color = data['Average Cost for two'],
    cmax = data['Average Cost for two'].max(),
    colorbar=dict(
                title="Average Cost for two"
            )
    )
    
)]
layout = dict(
    title = 'Restaurants Cost',
    hovermode='closest',
    geo = dict(showframe=False, countrywidth=1, showcountries=True,
               showcoastlines=True, projection=dict(type='Mercator'))
)
fig = go.Figure(data=plot_data, layout=layout)
iplot(fig)
data['Rating text'].value_counts()
Text_Rating_India = data_City.groupby(['City'], as_index=False).count()[['City', 'Restaurant ID']]
Text_Rating_India.head()
Text_Rating_India.columns = ['City', 'No of Restaurant']
Restaurant_text_rating=data_City.groupby(['City', 'Rating text'], as_index=False)['Restaurant ID'].count()
Total_Restaurant_text_rating_india = pd.merge(Text_Rating_India, Restaurant_text_rating, on='City')             
Total_Restaurant_text_rating_india['Percentage'] = (Total_Restaurant_text_rating_india['Restaurant ID']/Total_Restaurant_text_rating_india['No of Restaurant'])*100
Total_Restaurant_text_rating_india
sns.set(rc={'figure.figsize':(20,11)})
sns.barplot('City', 'Percentage', data=Total_Restaurant_text_rating_india, hue='Rating text')
plt.xticks(rotation = 90)
plt.xlabel('City', fontsize=20)
plt.title('No of Rating text', fontsize=30)
plt.ylabel('No of Ratings', fontsize=20)
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = (WordCloud(width=1440, height=1080, relative_scaling=0.5, stopwords=stopwords).generate_from_frequencies(data['Restaurant Name'].value_counts()))


fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
