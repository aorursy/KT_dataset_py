# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Load data
data = pd.read_csv('../input/2016 Stack Overflow Survey Responses.csv')

# Subset
data = data.loc[:,['dogs_vs_cats','star_wars_vs_star_trek']]

# Clean
data = data.dropna()
data = data[data.dogs_vs_cats.isin(['Dogs','Cats'])]
data = data[data.star_wars_vs_star_trek.isin(['Star Wars','Star Trek'])]

# Check
data.head()
# Plot 
colors = ['faded green', 'windows blue']
sns.set(style='whitegrid')
sns.catplot(data=data,
            x='star_wars_vs_star_trek',
            hue='dogs_vs_cats',
            palette=sns.xkcd_palette(colors),
           kind='count',
           legend=False)
plt.title('Somehow I knew this would be true')
plt.xlabel('')
plt.ylabel('Count')
plt.legend(title='')