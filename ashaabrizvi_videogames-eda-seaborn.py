import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/videogamesales/vgsales.csv')
data.head()
data.shape  # To check Number of Rows and Columns
data.isnull().sum() # To check Null Values
data.dropna(inplace = True) # To drop null values
# Preprocessing our Data for some Useful Insights 

data_decade = data[data.Year > 2010] 
data.corr() 
np.round(data.corr(),2)
# Heatmap 

plt.figure(dpi=125)

sns.heatmap(np.round(data.corr(),2),annot=True)

plt.show()
# Scatter Plot

plt.figure(dpi=125)

sns.regplot(x=data['NA_Sales'],y=data['Global_Sales'])

plt.xlabel('North America Sales')

plt.ylabel('Global Sales')

plt.title('Relationship between North America Sales and Global Sales')

plt.show()
plt.figure(dpi=125)

sns.regplot(x=data['EU_Sales'],y=data['Global_Sales'])

plt.xlabel('Europe Sales')

plt.ylabel('Global Sales')

plt.title('Relationship between Europe Sales and Global Sales')

plt.show()
plt.figure(dpi=125)

sns.regplot(x=data['JP_Sales'],y=data['Global_Sales'])

plt.xlabel('Japan Sales')

plt.ylabel('Global Sales')

plt.title('Relationship between Japan Sales and Global Sales')

plt.show()
plt.figure(dpi=125)

sns.regplot(x=data['Other_Sales'],y=data['Global_Sales'])

plt.xlabel('Other Sales')

plt.ylabel('Global Sales')

plt.title('Relationship between Other Country Sales and Global Sales')

plt.show()
# Count Plot

plt.figure(dpi=125)

sns.countplot(data.head(100)['Publisher'])

plt.xlabel('Publisher Name')

plt.xticks(rotation=90)

plt.ylabel('Count')

plt.title('Best Publisher in Top 100 Video Games')

plt.show()
# Count Plot

plt.figure(dpi=125)

sns.countplot(data_decade.head(100)['Publisher'])

plt.xlabel('Publisher Name')

plt.xticks(rotation=90)

plt.ylabel('Count')

plt.title('Best Publisher in Top 100 Video Games published in this Decade')

plt.show()
# Count Plot

plt.figure(dpi=125)

sns.countplot(data.head(100)['Genre'])

plt.xlabel('Genre Name')

plt.xticks(rotation=60)

plt.ylabel('Count')

plt.title('Best Genre in Top 100 Video Games')

plt.show()
# Count Plot

plt.figure(dpi=125)

sns.countplot(data_decade.head(100)['Genre'])

plt.xlabel('Genre Name')

plt.xticks(rotation=60)

plt.ylabel('Count')

plt.title('Best Genre in Top 100 Video Games published in this Decade ')

plt.show()
# Count Plot

plt.figure(dpi=125)

sns.countplot(data.head(100)['Platform'])

plt.xlabel('Platform Name')

plt.xticks(rotation=60)

plt.ylabel('Count')

plt.title('Best Platform in Top 100 Video Games')

plt.show()
# Count Plot

plt.figure(dpi=125)

sns.countplot(data_decade.head(100)['Platform'])

plt.xlabel('Platform Name')

plt.xticks(rotation=60)

plt.ylabel('Count')

plt.title('Best Platform in Top 100 Video Games published in this Decade')

plt.show()
# Year of Max Video Games Published

plt.figure(dpi=125)

sns.countplot(data.head(100)['Year'])

plt.xlabel('Year of Publish')

plt.xticks(rotation=60)

plt.ylabel('Count')

plt.title('In Which Year Max Video Games were Published')

plt.show()
from wordcloud import WordCloud
plt.subplots(figsize=(20,8))

wordcloud = WordCloud(background_color='White',width=1080,height=720).generate(" ".join(data.head(100)['Genre']))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
plt.subplots(figsize=(20,8))

wordcloud = WordCloud(background_color='White',width=1080,height=720).generate(" ".join(data_decade.head(100)['Genre']))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()