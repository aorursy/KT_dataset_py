# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# All imports done here 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set()



from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



from sklearn.mixture import GaussianMixture

import math

from IPython.display import display, HTML



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read the TSV (Tab separated values) into a pandas data frame

df = pd.read_csv('../input/en.openfoodfacts.org.products.tsv', low_memory=False, sep='\t')

# Dataframe info

df.info()
# Starter code for exploration - https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html ❤️ 

# Rows in a dataframe

print('Entries: '+str(len(df)))

# All the columns

print(df.columns)

# All the column types

print(df.dtypes)

# Some example data?

df.head()
#Filtering, sorting and slicing 

#1.Select

#2.Sort

#3.Slice

print(df[['product_name','energy_100g']].sort_values(by=['energy_100g'],ascending=False).head(20))

#Slice rows

print(df[0:5])
#Basic statistics

df.describe()
#Set the size of the figure

plt.figure(figsize=(5,30))

#Prepare the data

#Count and sort by the number of missing values.

counts_series = df.isnull().mean(axis=0).apply(lambda x: x*100).sort_values(ascending=False)

#Set the title and other labels if need be

plt.title("Proportion of missing values in each column")

plt.ylabel("column")

plt.xlabel("percentage")

#Plot the chart

plt.barh(counts_series.index,counts_series.values)

#Always show after plotting ⚡

plt.show()
# Change this to change the threshold for filtering - 0 to 100

cutoff = 30



def split_data_by_nan(dataframe,threshold):

    # percentage of null rows

    s = dataframe.isnull().mean(axis=0).apply(lambda x: x*100)

    # keep column if less nulls than threshold

    cols_of_interest = s[s<= threshold].index

    return cols_of_interest



# Filter columns by threshold

df_fs = df[split_data_by_nan(df,cutoff)]



print("Original number of features: " + str(df.shape[1]))

print("Number of features with less than "+ str(cutoff)+"% nans: " + str(df_fs.shape[1]))

print(df_fs.columns)
sns.set(context="paper", font_scale = 1.2)

# compute the correlation matrix for all the numeric columns

corrmat = df_fs.corr()

# size of the plot

f, ax = plt.subplots(figsize=(12, 12))

# set the plot heading

f.text(0.45, 0.93, "Pearson's correlation coefficients", ha='center', fontsize = 18)

# plot matrix as a heatmap

sns.heatmap(corrmat, square=True, linewidths=0.01, cmap="coolwarm")

plt.tight_layout()
# Features for modelling

filtered_columns = ['saturated-fat_100g','fat_100g','carbohydrates_100g','sugars_100g','proteins_100g','salt_100g','energy_100g']



# Cleaning if any values are empty in a row

df_fs["isempty"] = np.where(df_fs[filtered_columns].isnull().sum(axis=1) >= 1, 1, 0)

# Estimation of cleaning proportion

percentage = (df_fs.isempty.value_counts()[1] / df_fs.shape[0]) * 100

print("Percentage of incomplete tables: " + str(percentage))

print(df_fs.isempty.value_counts())

# Cleaning

df_cleaned = df_fs[df_fs.isempty==0]

# Check if cleaning is successful

df_cleaned.isnull().sum()
df_cleaned["reconstructed_energy"] = df_cleaned["fat_100g"] * 39 + df_cleaned["carbohydrates_100g"] * 17 + df_cleaned["proteins_100g"] * 17

df_cleaned.describe()

df_train = df_cleaned
'''

plt.figure(figsize = (10,10))

plt.scatter(df_cleaned["energy_100g"], df_cleaned["reconstructed_energy"])

plt.plot([0,5000], [0, 5000], color = 'black', linewidth = 2)

plt.xlabel("given energy")

plt.ylabel("reconstructed energy")

plt.show()

'''
'''

print(len(df_cleaned[df_cleaned["fat_100g"] + df_cleaned["carbohydrates_100g"] + df_cleaned["proteins_100g"] > 100]))

df_train = df_cleaned[df_cleaned["fat_100g"] + df_cleaned["carbohydrates_100g"] + df_cleaned["proteins_100g"] <= 100] 

'''
%%time

# change the number of clusters

n_clusters = 10



#replacing energy with reconstructed energy

features = ['saturated-fat_100g','fat_100g','carbohydrates_100g','sugars_100g','proteins_100g','salt_100g','reconstructed_energy']

X_train = df_train[features].values



#create the model

max_iter = 100

model = GaussianMixture(n_components=n_clusters, covariance_type="full", n_init = 5, max_iter = max_iter)

#fit model on data

model.fit(X_train)



#predict cluster number for the date

results = df_train

results["cluster"] = model.predict(X_train)
%%time

def make_word_cloud(data, cluster, subplotax, title):

    # Get words from the product name 

    words = data[data.cluster==cluster]["product_name"].apply(lambda l: l.lower().split() if type(l) == str else '')

    # Add to a pandas series

    cluster_words=words.apply(pd.Series).stack().reset_index(drop=True)

    # Split and join

    text = " ".join(w for w in cluster_words)



    # Create and generate a word cloud image:

    wordcloud = WordCloud(max_font_size=30, max_words=30, background_color="white", colormap="YlGnBu").generate(text)



    # Display the generated image:    

    subplotax.imshow(wordcloud, interpolation='bilinear')

    subplotax.axis("off")

    subplotax.set_title(title,fontweight="bold", size=20)

    return subplotax



rows = math.ceil(n_clusters/2)

fig, ax = plt.subplots(rows,2, figsize=(20,50))

for m in range(rows):

    for n in range(2):

        cluster = m*2+ n

        title = "Cluster " + str(cluster) 

        make_word_cloud(results, cluster, ax[m,n], title)
# Cluster ? has just one element

'''

results[results['cluster'] == 6]

'''
for cluster in range(0,n_clusters):

    display("Cluster "+str(cluster))

    display(results[results['cluster'] == cluster][features].describe())
# Change this to include the clusters you want

# Eg. 1 portion carbohydrates, 1 seasoning, 1 dairy product, 2 portions of proteins

meal_clusters = [1,2,3,9,9]



indices = []

results['rownum'] = results.index

for cluster in meal_clusters:

    indices+=results[results['cluster'] == cluster].sample(1)['rownum'].tolist()



meal = pd.DataFrame()

for index in indices:

    meal = meal.append(results[results['rownum'] == index])

meal