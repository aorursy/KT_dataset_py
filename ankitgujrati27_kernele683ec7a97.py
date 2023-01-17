# Import Pandas:

import pandas as pd
import matplotlib.pyplot as plt
# Reading the CSV file:

cereal_data = pd.read_csv("../input/cereal.csv")
# Checking the top 5 rows:

cereal_data.head()
# Describing the data:

cereal_data.describe()
# Finding the total number of rows and columns and column names:

print (cereal_data.shape)
print (cereal_data.columns)
# Finding Calories per serving:

plt.hist(cereal_data['calories'], histtype = 'stepfilled', color = 'green', orientation = 'vertical', edgecolor = 'red')
plt.title('Calories')
plt.xlabel('Calories per Serving')
plt.ylabel('Number of Cereals')
# Analysing the Sodium content in the Cereal and its data distribution:

plt.hist(cereal_data['sodium'], bins = 30)
# Data above is not normalized, there are more cereals with sodium content below 50. Digging more into it.

# Importing Seaborn library:

import seaborn as sns
sns.set(color_codes = 'true')
sns.distplot(cereal_data['sodium'], )
# Out of 100, how many are cold and how many are hot Cereals?

# Getting the counts:

cereal_type_count = cereal_data['type'].value_counts()
print (cereal_type_count)
plt.pie(cereal_type_count, labels = ['Cold', 'Hot'], explode= (.1, .2), shadow = 'True', autopct = '%1.1f%%', colors = ('b','g'))
plt.axis('equal')
# Finding out the manufacturers:

manufacturer_data = cereal_data['mfr'].value_counts()
manufacturer_data
manufacturers = ['Kelloggs', 'General Mills', 'Post', 'Quaker Oats', 'Ralston Purina', 'Nabisco','American Home \n Food Products']
plt.pie(manufacturer_data, labels= manufacturers, explode = (.1,0,0,0,0,0,0), shadow='True', autopct = '%1.1f%%')
# Finding out the top manufacturer by average rating:

manufacturer_by_rating = cereal_data.groupby(['mfr']).mean()
print(manufacturer_by_rating)
manufacturer_by_rating['rating']

# Nabisco is on the top.
# Number of Cereals offered by each manufacturer:

sns.countplot(cereal_data.mfr)


