# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns
#reading in the data



penguins = pd.read_csv('/kaggle/input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv')

penguins.head(5)
#Checking for missing values

penguins.info()

penguins.isna().sum()

#missing values present in most columns, 'Sex' Column has the highest (10)
#exploring the data

sns.countplot(x = "species", data = penguins)
#Adelie are the highest in number followed by Gentoo and Chintrap
sns.countplot(x = "island", data = penguins)
#most of the Penguins belong to Biscoe island and least are from Torgersen
#culmen length and depth wrt species 



fig,axs = plt.subplots(ncols = 2)

fig.tight_layout()



sns.boxplot(y= 'culmen_length_mm', x = 'species', data = penguins, ax= axs[0])

sns.boxplot(y= 'culmen_depth_mm', x = 'species', data = penguins, ax= axs[1])
#flipper length by species



sns.boxplot(x = 'species', y = 'flipper_length_mm', data = penguins)
#Gentoo species has the longest flippers while Adelie has the shortest
#body mass of different Species



sns.boxplot(x = 'species', y = 'body_mass_g', data = penguins)
#Gentoo are the heaviest of the three species whereas Adelie and Chinstrap weigh around the same
#Species count by sex



sns.countplot(x='sex', data = penguins, hue = 'species')
#all species are evenly distributed across the genders
#Let us try to plot culmen length & depth. Let us see if we can get a clear grouping of species 



sns.scatterplot(x='culmen_length_mm', y='culmen_depth_mm', hue = 'species', data = penguins)
#we can observe that Adelie and Gentoo form two distinctive grouping with Chinstrap being a little more spread out
#Handling missing values using mean for numerical variables



penguins["culmen_length_mm"] = penguins["culmen_length_mm"].fillna(value = penguins["culmen_length_mm"].mean())

penguins["culmen_depth_mm"] = penguins["culmen_depth_mm"].fillna(value = penguins["culmen_depth_mm"].mean())

penguins["flipper_length_mm"] = penguins["flipper_length_mm"].fillna(value = penguins["flipper_length_mm"].mean())

penguins["body_mass_g"] = penguins["body_mass_g"].fillna(value = penguins["body_mass_g"].mean())



penguins.isna().sum()
#filling in missing values of categorical variable 'Sex'



penguins["sex"] = penguins["sex"].fillna("FEMALE")



penguins.isna().sum()
#encoding categorical variables

y = penguins["species"]

penguins_main = penguins.iloc[:,1:]

X = pd.get_dummies(penguins_main)

X.head()
#Let us try K-means clustering on the data



from sklearn.cluster import KMeans



model = KMeans(n_clusters=3)

model.fit(X)

# Using the predict method of KMeans to predict 3 clusters using the sample data



labels = model.predict(X)
# We plot a cross tab matrix to check how well has our K-Means model classified the Species



matrix = pd.DataFrame({'labels': labels, 'species': y})

ct = pd.crosstab(matrix['labels'], matrix['species'])

print(ct)
# It seems we are not making an accurate clustering



# So Let's try scaling the data to check, if it helps our KMeans to perform better
#importing StandardScaler & pipeline and initializing



from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline



scaler = StandardScaler()

pipeline = make_pipeline (scaler, model)

#using the pipeline object to fit and predict



pipeline.fit(X)

labels_new = pipeline.predict (X)
#plotting the Cross-tab matrix with the new labels



matrix_new = pd.DataFrame({'labels': labels_new, 'species': y})

ct_new = pd.crosstab(matrix_new['labels'], matrix_new['species'])

print(ct_new)
# We can see that we are able to classify 'Gentoo' perfectly now while work needs to be done more on 'Adelie' adn 'Chinstrap'
# Upvote if you find this notebook useful. Cheers!!