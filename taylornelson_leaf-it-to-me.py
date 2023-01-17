#get started with some libraries, girl.

import pandas as pd

import numpy as np
#create df for the training data, "mmm data"

train = pd.read_csv('../input/train.csv')

train.describe()
#get proportions for each class, in case we need to over/under sample

#visualize with pyplot

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#probably an easier way to do this...

species_list = train['species'].unique() #all the species

y = [] #list to hold counts for each species



#print number of species

print (str(len(species_list)) + " species") 



#get counts for each species

for c in range(len(species_list)):

    y.append(train['species'][train['species']==species_list[c]].count())





    

fig = plt.figure(figsize=(12, 3), dpi=100)

y_pos = np.arange(len(species_list))



#sns.set_color_codes("muted")

#sns.barplot(x=y_pos, y=y,color="b")    

plt.bar(y_pos, y, align='center', width=0.5,alpha=0.5)

plt.xticks(y_pos, species_list,rotation='vertical')

plt.xlabel('Species')

plt.ylabel('Row Count')

plt.title('Species Dist')

plt.show()



#try this again, later, with seaborn
#I got 99 species... each with 10 rows / observations

#no imbalance issues but...

#completely uniform distribution. Not a lot of N per class.
#first, scale the predictors, since the values are all over the place

from sklearn.preprocessing import StandardScaler

train = train[train.columns[2:]]