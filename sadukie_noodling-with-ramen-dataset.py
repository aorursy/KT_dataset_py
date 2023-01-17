import pandas as pd
ramen_ratings = pd.read_csv('/kaggle/input/ramen-ratings/ramen-ratings.csv')
ramen_ratings.shape
ramen_ratings.info()
len(ramen_ratings.Brand.unique())
ramen_ratings.Brand.value_counts().nlargest(10).plot.bar()
brand_distribution = ramen_ratings.Brand.value_counts().nlargest(10)
import matplotlib.pyplot as plt
ax = brand_distribution.plot(kind="bar", title='Brand Distribution in the Ramen Ratings Dataset')
ax.set_xlabel('Brand')
ax.set_ylabel('Number of Entries in Dataset')
ax.set_title('Brands with Most Entries in Ramen Ratings Dataset')

plt.show()
ramen_ratings.Style.unique()
ramen_ratings.Variety.unique()
ramen_ratings.Stars.value_counts()
# Remove the not rated ramen
ramen_ratings = ramen_ratings[ramen_ratings.Stars != 'Unrated']
# Set the stars to a value we can plot
ramen_ratings.Stars = ramen_ratings.Stars.astype(float)

cup_ramen = ramen_ratings[ramen_ratings.Style == 'Cup']
bowl_ramen = ramen_ratings[ramen_ratings.Style == 'Bowl']
cup_ramen.nlargest(10,'Stars')
bowl_ramen.nlargest(10,'Stars')
top_countries_with_5_star_rated_ramen = ramen_ratings[ramen_ratings.Stars >= 5].groupby('Country')['Stars'].value_counts().nlargest(10)
import matplotlib.pyplot as plt
ax = top_countries_with_5_star_rated_ramen.plot(kind="barh", title='Top Countries with 5-Star Rated Ramen')
ax.set_xlabel('Number of 5-Star Ratings')
ax.set_ylabel('Country')

plt.show()
ax = ramen_ratings["Stars"].hist()
ax.set_title("Star Ratings Counts in Ramen Ratings")
ax.set_xlabel("Ratings")
ax.set_ylabel("Number of Ratings")
plt.show()
import seaborn as sns
sns.set_style('darkgrid')
ax = ramen_ratings["Stars"].hist()
ax.set_title("Star Ratings Counts in Ramen Ratings")
ax.set_xlabel("Ratings")
ax.set_ylabel("Number of Ratings")
plt.show()
ax = ramen_ratings["Stars"].plot.kde()
ax.set_title("Star Ratings Kernel Density Chart")
ax.set_ylabel("Density")
plt.show()
print(ramen_ratings[["Stars","Style"]].head())

sns.boxplot(x='Style', y='Stars', data=ramen_ratings)
ax = ramen_ratings.boxplot(column='Stars',by='Style')
plt.show()
ramen_ratings[ramen_ratings.Style == 'Bar']
ramen_ratings[ramen_ratings.Style == 'Can']
