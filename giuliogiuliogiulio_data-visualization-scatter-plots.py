import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
# Set up code checking
import os
if not os.path.exists("../input/candy.csv"):
    os.symlink("../input/data-for-datavis/candy.csv", "../input/candy.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex4 import *
# Path of the file to read
candy_filepath = "../input/candy.csv"

# Fill in the line below to read the file into a variable candy_data
candy_data = pd.read_csv(candy_filepath, index_col='id')

# Run the line below with no changes to check that you've loaded the data correctly
step_1.check()
# Print the first five rows of the data
candy_data.head()
# Fill in the line below: Which candy was more popular with survey respondents:
# '3 Musketeers' or 'Almond Joy'?  (Please enclose your answer in single quotes.)
more_popular = '3 Musketeers'

# Fill in the line below: Which candy has higher sugar content: 'Air Heads'
# or 'Baby Ruth'? (Please enclose your answer in single quotes.)
more_sugar = 'Air Heads'
# Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'
plt.figure(figsize=(10, 10))
sns.scatterplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])
plt.title('Relationship between %sugar and %popularity')
# The correlation between the two variables appears to be very low, thus, the %sugar does not play a relevant role for the popularity of the candies.
# Scatter plot w/ regression line showing the relationship between 'sugarpercent' and 'winpercent'
plt.figure(figsize=(10, 10))
sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])
plt.title('Relationship between %sugar and %popularity')
# The regression line shows a slightly positive correlation between %sugar and %win. This means
# that the percentage of sugar has a slim impact on the popularity of the candies.
# Scatter plot showing the relationship between 'pricepercent', 'winpercent', and 'chocolate'
plt.figure(figsize=(10, 10))
sns.scatterplot(x=candy_data['pricepercent'], y=candy_data['winpercent'], hue=candy_data['chocolate'])
plt.title('Relationship between %sugar and popularity, by chocolate content')
# Color-coded scatter plot w/ regression lines
plt.figure(figsize=(10, 10))
sns.lmplot(x='pricepercent', y='winpercent', hue='chocolate', data=candy_data)
# Candies with chocolate tend to be more popular as their price increase.
# Candies without chocolate tend to be less popular as their price increase.
# Scatter plot showing the relationship between 'chocolate' and 'winpercent'
sns.swarmplot(x=candy_data['chocolate'], y=candy_data['winpercent'])
# Catehorical scatter plot is more appropriate since it convey the information in a simple and more intuitive way.