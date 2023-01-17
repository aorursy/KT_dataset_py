import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

netflix_shows = pd.read_csv("../input/netflix-shows/netflix_titles.csv")

print(netflix_shows.head())
print(netflix_shows.info())


content_ratings = netflix_shows[['show_id', 'type', 'rating']]
#print(content_ratings)

tv_or_movie = content_ratings.groupby('type').show_id.count().to_frame('count')
#print(tv_or_movie) 
##Movie     4265
##TV Show   1969

content_category = ["Movie", "TV Show"]
content_count = [4265, 1969]

# Creating the pie chart
plt.pie(content_count, labels=content_category, autopct='%d%%')
plt.axis('equal')
#plt.legend(content_category)
plt.title("Movies & TV Shows percentage")
plt.show()
rating_and_their_counts = content_ratings.groupby('rating').show_id.count().to_frame('count').reset_index()

rating_types = rating_and_their_counts['rating']
rating_types_counts = rating_and_their_counts['count']

#print(rating_and_their_counts)
#print(rating_types)
#print(rating_types_counts)


# creating the plot
plt.figure(figsize=(16, 8))
plt.bar(range(len(rating_types)), rating_types_counts, width=0.5)

ax = plt.subplot()
ax.set_xticks(range(len(rating_types)))
ax.set_xticklabels(rating_types)

plt.title("Rating catogeries on Netflix")
plt.xlabel("Rating")
plt.ylabel("Count of each catogery")
plt.grid()
ax.set_axisbelow(True)
plt.show()