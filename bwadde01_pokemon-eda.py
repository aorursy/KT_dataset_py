# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pokemon_stats = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')

pokemon_images = pd.read_csv('/kaggle/input/pokemon-images-and-types/pokemon.csv')
pokemon_stats
# From examination, it appears that there are some pokemon in the stats dataset that are not in the images dataset (e.g. VenusaurMega)

pokemon_images
pokemon_stats['lower_names'] = pokemon_stats['Name'].apply(lambda x: x.lower())
# merge the images and the stats together so we can extract images where possible

pokemon_full = pokemon_stats.merge(pokemon_images,how='left',left_on='lower_names',right_on='Name')
pokemon_full
# here are all of the pokemon missing from the images set based on our join - looks like there are some that would have matches if we reformat the names to match

pokemon_full[pokemon_full['Name_y'].isna()]['Name_x'].tolist()
# in the meantime, let's filter out the megas to see how many remain -- here are all the unavailable pokemon

pokemon_missing = pd.DataFrame(pokemon_full[pokemon_full['Name_y'].isna() & ['mega' not in name for name in pokemon_full['lower_names'].tolist()]])

pokemon_full[pokemon_full['Name_y'].isna() & ['mega' not in name for name in pokemon_full['lower_names'].tolist()]]
# let's re-format these pokemon based on the observed naming convention being used in the files

# pokemon with more than one form make up the vast majority of this population; we can use regular expresisons to highlight these and update them to the 

# naming convention being used in the images dataset

pokemon_missing['reformatted_name'] = ["".join(re.split('([A-Z])',eg)[1:3]).lower()+'-'+"".join(re.split('([A-Z])',eg)[3:5]).lower()[:-1] if len(re.findall('[A-Z]',eg))>2 else eg for eg in pokemon_full[pokemon_full['Name_y'].isna() & ['mega' not in name for name in pokemon_full['lower_names'].tolist()]]['Name_x'].tolist()]

pokemon_missing
# here are the matches we were able to get from reformatting... looks like some of the alternate forms are not present in the images dataset, so we will leave those

# for now, but a few other ones which did not get picked up we will have to look at more closely

pokemon_missing.merge(pokemon_images,how='left',left_on='reformatted_name',right_on='Name')
# Below are the pokemon which require manual updating to be joined with the images dataset

# Nidoran♀ -> 'nidoran-m'

# Nidoran♂ -> 'nidoran-f'

# Farfetch'd -> 'farfetchd'

# Mr. Mime -> 'mr-mime'

# Mime Jr. -> 'mime-jr'

# Basculin? -> 'basculin-red-striped'

# Flabébé -> 'flabebe'

# MeowsticMale -> 'meowstick-male'

# Zygarde50% Forme -> 'zygarde-50'

# HoopaHoopa Confined -> 'hoopa-confined'
# for some of the names, we go in and manually update them in the missing df to have them align with the images df

pokemon_missing['manual_name']=pokemon_missing['Name_x'].map({'Nidoran♀':'nidoran-m','Nidoran♂':'nidoran-f','Farfetch\'d':'farfetchd','Mr. Mime':'mr-mime','Mime Jr.':'mime-jr','Basculin':'basculin-red-striped','Flabébé':'flabebe','MeowsticMale':'meowstick-male','Zygarde50% Forme':'zygarde-50','HoopaHoopa Confined':'hoopa-confined'})
# as you can see, the manual name has been added

pokemon_missing
# here, we add the manual name and reformatted name together to optimize the matching between this set and the pokemon images dataset (through matching with the full view on name)

pokemon_missing['final_name'] = [y if y is not np.nan else x  for x,y in zip(pokemon_missing['reformatted_name'].tolist(),pokemon_missing['manual_name'].tolist())]
pokemon_missing
pokemon_full = pokemon_full.merge(pokemon_missing,how='left',left_on='Name_x',right_on ='Name_x')
# same process as for the missing pokemon; 

pokemon_full['master_name'] = [y if y is not np.nan else x  for x,y in zip(pokemon_full['lower_names_x'].tolist(),pokemon_full['final_name'].tolist())]
# add the pokemon image path to the full df

pokemon_full['image'] = pokemon_full['master_name'].apply(lambda x: f"/kaggle/input/pokemon-images-and-types/images/images/{x}.png" if x in pokemon_images['Name'].tolist() else "")
# rename the columns back to the original values to clean things up

pokemon_full = pokemon_full[['#_x','Name_x','Type1_x','Type2_x','Total_x','HP_x','Attack_x','Defense_x','Sp. Atk_x','Sp. Def_x','Speed_x','Generation_x','Legendary_x','master_name','image']]

pokemon_full.columns = ['#','Name','Type 1','Type 2','Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','Legendary']+['master_name','image']
# filter out the pokemon which were not found in the images dataset

pokemon_full = pd.DataFrame(pokemon_full[pokemon_full['image']!=''])
pokemon_full
from IPython.display import Image 

pil_img = Image(filename=pokemon_full['image'].iloc[383])

display(pil_img)
# let's try to visualize the data, first by reducing its dimension to 2D

from sklearn.decomposition import PCA



dim_reducer = PCA()



dim_reducer.fit(pokemon_full[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']].values)
# the first two principal components have an explained variance ratio of 43.6% and 18.9% respectively

dim_reducer.explained_variance_ratio_
# now let's take the principal components and see what information they hold



result = dim_reducer.transform(pokemon_full[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']].values)

pokemon_full['pca-one'] = result[:,0]

pokemon_full['pca-two'] = result[:,1]

pokemon_full['pca-three'] = result[:,2]
# now let's visualize the pokemon based on their first two principal components

from PIL import Image

plt.figure(figsize=(32,20))

plt.xlim([-110,125])

plt.ylim([-110,150])



sns.scatterplot(

    x='pca-one', y='pca-two',

    data=pokemon_full,

    palette='bright',

    legend="full",

    alpha=0.3,

    zorder=0

)



for index,row in pokemon_full.iterrows():

    plt.imshow(Image.open(row.image).resize((50,50)),zorder=5,extent=(row['pca-one']-10,row['pca-one']+10,row['pca-two']-10,row['pca-two']+10))
# lookin also at the other groupings, we can see that those pokemon classified as legendary score much higher in the first principal component but possess

# similar variation in the second



plt.figure(figsize=(16,10))

sns.scatterplot(

    x='pca-one', y='pca-two',

    hue='Legendary',

    data=pokemon_full,

    palette='bright',

    legend="full",

    alpha=0.3,

    zorder=0

)
# as one might expect, there is significant diversity in the quality of pokemon across typings



plt.figure(figsize=(16,10))

sns.scatterplot(

    x='pca-one', y='pca-two',

    hue='Type 1',

    data=pokemon_full,

    palette='bright',

    legend="full",

    alpha=0.3

)
# as one might expect, there is significant diversity in the quality of pokemon across the generations in which they were released

plt.figure(figsize=(16,10))

sns.scatterplot(

    x='pca-one', y='pca-two',

    hue='Generation',

    data=pokemon_full,

    palette='bright',

    legend="full",

    alpha=0.3

)
# commonly, people have general archetypes that they use to classify pokemon, but this is not so formal.

# In this section, we try clustering to be able to identify typical groupings people use (bulky, glass cannon, pseudo-legendary, etc.)

from sklearn.cluster import KMeans



clustering = KMeans(n_clusters=6).fit(pokemon_full[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']].values) # tuned for number of clusters; 6 appeared best
#update the pokemon full view witht the clustering labels

pokemon_full['Cluster'] = clustering.labels_
# using the hue from the new Cluster column, let's see the groupings it produced



plt.figure(figsize=(16,10))

sns.scatterplot(

    x='pca-one', y='pca-two',

    hue='Cluster',

    data=pokemon_full,

    palette='bright',

    legend="full"

)
# grabbed list of all final evolutions from this site: https://bulbapedia.bulbagarden.net/wiki/List_of_fully_evolved_Pok%C3%A9mon_by_base_stats

# to be used to reduce noise in the visual and evaluate pokemon for their competitive usability (assuming people would not use earlier stages of a pokemon)

fully_evolved = pd.read_csv(r"/kaggle/input/pokemon-fully-evolved/fully_evolved.csv")

# remove duplicates in serial number from the fully evolved file

fully_evolved.drop_duplicates(inplace=True)
# now let's overlay the images of the pokemon to dive deeper into the clusters

from matplotlib.patches import Ellipse





plt.figure(figsize=(15,30))

sns.scatterplot(

    x='pca-one', y='pca-two',

    hue='Cluster',

    data=pokemon_full,

    palette='bright',

    legend="full"

)



plt.xlim([-110,125])

plt.ylim([-110,150])



for index,row in pokemon_full[[pokemon in fully_evolved['#'].tolist() for pokemon in pokemon_full['#'].tolist()]].iterrows():

    plt.imshow(Image.open(row.image).resize((50,50)),zorder=5,extent=(row['pca-one']-10,row['pca-one']+10,row['pca-two']-10,row['pca-two']+10))



centroids = clustering.cluster_centers_

for enum,color in zip(enumerate(centroids),['blue','orange','green','red','purple','brown']):

    ind = enum[0]

    i = enum[1]

    data_for_class = pokemon_full[pokemon_full['Cluster']==ind][['pca-one','pca-two']]

    

    # compute average in 2d space and diameter of class

    max_horiz_dist=np.max(data_for_class['pca-one'].values)-np.min(data_for_class['pca-one'].values)

    max_vert_dist=np.max(data_for_class['pca-two'].values)-np.min(data_for_class['pca-two'].values)

    avg_2d = [data_for_class['pca-one'].mean(),data_for_class['pca-two'].mean()]

    

    plt.gca().add_artist(Ellipse(avg_2d,max_horiz_dist,max_vert_dist, fill=False,zorder=10,color=color,linewidth=3))