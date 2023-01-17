# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.colors import ListedColormap

from scipy import stats

from sklearn.preprocessing import StandardScaler



plt.style.use('ggplot')

sns.set(font_scale = 1.5)

%config InlineBackend.figure_format = 'retina'

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



# Pallets used for visualizations

color= "Spectral"

color_plt = ListedColormap(sns.color_palette(color).as_hex())

color_hist = 'teal'
df = pd.read_csv("/kaggle/input/all-products-available-on-sephora-website/sephora_website_dataset.csv")
df.head(2)
df.shape
df.info()
#visualize the missing data 

plt.figure(figsize = (10, 6))

sns.heatmap(data = df.isnull())
fig, ax = plt.subplots( figsize=(15, 6))

ax.hist(df['price'], bins = 300, color = color_hist)



ax.set_xlabel('Price')

ax.set_ylabel('Frequency')

fig.suptitle('The Distribution of products Price in Sephora website ', fontsize = 20)



ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))



plt.show()
fig, ax = plt.subplots( figsize=(15, 6))

ax.hist(df['rating'], color = color_hist)



ax.set_xlabel('rating')

ax.set_ylabel('Frequency')

fig.suptitle('The Distribution of products Rating in Sephora website ', fontsize = 20)



#ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))



plt.show()


fig, ax = plt.subplots( figsize=(15, 6))

ax.hist(df['number_of_reviews'], color = color_hist)



ax.set_xlabel('number_of_reviews')

ax.set_ylabel('Frequency')

fig.suptitle('The Distribution of number of reviews in each product in Sephora website ', fontsize = 20)



#ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))



plt.show()
ig, ax = plt.subplots( figsize = (12, 8))

ax = sns.scatterplot(x = 'rating', 

                     y = 'price', 

                     data = df, 

                     marker = 'o', s = 200, palette = color)



ax.set_ylabel('Price')

ax.set_xlabel('Rating')

fig.suptitle('The product Rating vs. Price', fontsize = 20)



ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.show()
ig, ax = plt.subplots( figsize = (12, 8))

ax = sns.scatterplot(x = 'number_of_reviews', 

                     y = 'price', 

                     data = df, 

                     marker = 'o', s = 200, palette = color)



ax.set_ylabel('Price')

ax.set_xlabel('Number of Reviews')

fig.suptitle('The Number of reviews for product vs. Price', fontsize = 20)



ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.show()
df.describe()
df.select_dtypes('object').columns
df.corr()['price'].sort_values(ascending = False)
fig, axs = plt.subplots(figsize = (16, 14)) 

mask = np.triu(np.ones_like(df.corr(), dtype = np.bool))

g = sns.heatmap(df.corr(), ax = axs, mask=mask, cmap = sns.diverging_palette(180, 10, as_cmap = True), square = True)



plt.title('Correlation between Features')



# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show()
# a function that takes a dataframe and transforms it into a standard form after dropping nun_numirical columns

def to_standard (df):

    

    num_df = df[df.select_dtypes(include = np.number).columns.tolist()]

    

    ss = StandardScaler()

    std = ss.fit_transform(num_df)

    

    std_df = pd.DataFrame(std, index = num_df.index, columns = num_df.columns)

    return std_df
ax, fig = plt.subplots(1, 1, figsize = (18, 18))

plt.title('The distribution of All Numeric Variable in the Dataframe', fontsize = 20) #Change please



sns.boxplot(y = "variable", x = "value", data = pd.melt(to_standard(df)), palette = color)

plt.xlabel('Range after Standarization', size = 16)

plt.ylabel('Attribue', size = 16)





# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values



plt.show()
numeric_feats = df.dtypes[df.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = df[numeric_feats.tolist()].apply(lambda x:stats.skew(x.dropna())).sort_values(ascending = False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew':skewed_feats})

skewness.head()