import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
cartier = pd.read_csv('/kaggle/input/cartier-jewelry-catalog/cartier_catalog.csv')
cartier.head(5)
# Define tag_splitter // splits column  

def tag_spiliter(dataframe , col_name , delimiter , metal , first_gem , second_gem , third_gem , foruth_gem):

    dataframe['str_split'] = dataframe[col_name].str.split(delimiter)

    dataframe[metal] = dataframe.str_split.str.get(0).str.strip()

    dataframe[first_gem] = dataframe.str_split.str.get(1).str.strip()

    dataframe[second_gem] = dataframe.str_split.str.get(2).str.strip()

    dataframe[third_gem] = dataframe.str_split.str.get(3).str.strip()

    dataframe[foruth_gem] = dataframe.str_split.str.get(4).str.strip()

    dataframe.fillna(0 , inplace=True)

    del dataframe['str_split']
# Recall tag_splitter

tag_spiliter(cartier , 'tags' , ',' , 'metal' , 'gem' , 'second_gem' , 'third_gem' , 'foruth_gem')
# Drop redundant columns

cartier.drop(['ref' , 'image' , 'tags' , 'title' , 'description'] , axis  = 1 , inplace=True)

gems = pd.concat([cartier["gem"],cartier["second_gem"],cartier["third_gem"],cartier["foruth_gem"]], axis= 0)

gems_values = gems.value_counts()[1:].to_frame()

gems_values.reset_index(inplace=True)

gems_values.columns = ['gem_type' , 'count']
plt.figure(figsize=(15, 5))

sns.barplot(x= 'gem_type', y= "count", data= gems_values,

            palette= sns.cubehelix_palette(len(gems_values.gem_type), start=.5, rot=-.75, reverse= True))

plt.xlabel("Gems Type")

plt.ylabel("Count")

plt.title("Count of Gems")

plt.xticks(rotation= 90)

plt.show()
# Dictionary for costum color palette 

color_dict = {'yellow gold': "#fcc72d", 

              'platinum': "#e5e4e2", 

              'pink gold': "#e9cdd0", 

              'white gold': "#f9f3d1", 

              'non-rhodiumized white gold': "#C0C0C0"}
cartier_category_metal = cartier.groupby('categorie')['metal'].value_counts().to_frame()

cartier_category_metal.columns = ['count']

cartier_category_metal.reset_index(level = [0 , 1] , inplace=True)
plt.figure(figsize=(15, 7))

sns.barplot(x= "categorie", y= "count", hue= "metal", data= cartier_category_metal, 

            palette= color_dict)

plt.xlabel("Jewels Type")

plt.ylabel("Counts")

plt.legend(loc= "upper left")

plt.show()
cartier_gp1 = cartier.groupby(["categorie", "metal"])["price"].mean().round(2).to_frame()

cartier_gp1 = cartier_gp1.reset_index()
plt.figure(figsize=(15, 7))

sns.barplot(x= 'categorie', y= 'price', hue= 'metal', data= cartier_gp1 , palette = color_dict)

plt.xlabel('Jewels Type')

plt.ylabel('Mean Price in $')

plt.legend(loc= "upper left")

plt.show()
cartier_gp2 = cartier.groupby("metal")["price"].mean().round(2).to_frame()

cartier_gp2.reset_index(inplace=True)
plt.figure(figsize=(15, 7))

sns.barplot(x= "metal" , y = 'price', data=cartier_gp2 , palette = color_dict)

plt.xlabel('Metal')

plt.ylabel('Mean Price in $')

plt.show()
cartier_gp_gem = cartier.groupby('categorie')['gem'].value_counts().to_frame()

cartier_gp_gem.columns = ['count']

cartier_gp_gem.reset_index(level = [0 , 1] , inplace=True)

cartier_gp_gem = cartier_gp_gem[cartier_gp_gem["gem"] != 0]
plt.figure(figsize=(15, 7))

sns.barplot(x= 'categorie', y= 'count', hue= 'gem', data= cartier_gp_gem , palette = sns.color_palette("Set2"))

plt.xlabel('Jewels Type')

plt.ylabel('Counts')

plt.legend(ncol=4, loc= 'upper left')

plt.show()
cartier_gp1_gem = cartier.groupby(["categorie", "gem"])["price"].mean().round(2).to_frame()

cartier_gp1_gem = cartier_gp1_gem.reset_index()

cartier_gp1_gem = cartier_gp1_gem[cartier_gp1_gem["gem"] != 0]
plt.figure(figsize=(15, 7))

sns.barplot(x= 'categorie', y= 'price', hue= 'gem', data= cartier_gp1_gem , palette = sns.color_palette("Set2"))

plt.xlabel('Jewels Type')

plt.ylabel('Mean Price in $')

plt.legend(ncol=4, loc= 'upper left')

plt.show()
cartier_gp2_gem = cartier.groupby("gem")["price"].mean().round(2).to_frame()

cartier_gp2_gem.reset_index(inplace=True)

cartier_gp2_gem = cartier_gp2_gem[(cartier_gp2_gem['gem'] != 'white gold') &

                                  (cartier_gp2_gem['gem'] != 'yellow gold') & 

                                  (cartier_gp2_gem['gem'] != 0)]
plt.figure(figsize=(15, 8))

sns.barplot(x= 'gem' , y = 'price', data=cartier_gp2_gem ,  palette = sns.color_palette("Set2"))

plt.xlabel('Gem Type')

plt.ylabel('Mean Price in $')

plt.xticks(rotation=90)

plt.show()