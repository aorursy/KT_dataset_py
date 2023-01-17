import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn import preprocessing



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/cereal.csv")



# Print the head of df

print(df.head())



# Print the info of df

print(df.info())



# Print the shape of df

print(df.shape)
df.iloc[:,~df.columns.isin(['name','mfr','type','rating'])].describe()
pd.DataFrame(df['mfr'].value_counts(dropna=False))
# Visualization product ranking

df = df.sort_values(['rating'],ascending=False).reset_index(drop=True)

plt.figure(figsize=(20,20))

sns.barplot(x=df["rating"],y=df["name"])

plt.xlabel("Product Name",fontsize=15)

plt.ylabel("Rating",fontsize=15)

plt.title("Product Rating",fontsize=15)

plt.show()
# Barcharts to understand the categorical variables

f, axes = plt.subplots(1,2, figsize=(10, 5))

sns.countplot(x="mfr", data=df, ax=axes[0], palette="Set3")

sns.countplot(x="type", data=df, ax=axes[1], palette="Set2")
# Display the histogram to undestand the data

f, axes = plt.subplots(3,5, figsize=(20, 12))

sns.distplot( df["calories"], ax=axes[0,0])

sns.distplot( df["protein"], ax=axes[0,1])

sns.distplot( df["fat"], ax=axes[0,2])

sns.distplot( df["fiber"], ax=axes[0,3])

sns.distplot( df["sodium"], ax=axes[0,4])

sns.distplot( df["carbo"], ax=axes[1,0])

sns.distplot( df["sugars"], ax=axes[1,1])

sns.distplot( df["protein"], ax=axes[1,2])

sns.distplot( df["vitamins"], ax=axes[1,3])

sns.distplot( df["cups"], ax=axes[1,4])

sns.distplot( df["potass"], ax=axes[2,0])

sns.distplot( df["vitamins"], ax=axes[2,1])

sns.distplot( df["shelf"], ax=axes[2,2])

sns.distplot( df["weight"], ax=axes[2,3])
# Barcharts: are they comparable?

f, axes = plt.subplots(1,2, figsize=(10, 5))

sns.scatterplot(x="mfr", y="weight", data=df, ax=axes[0])

sns.scatterplot(x="mfr", y="cups", data=df, ax=axes[1])
#How to normalize the values

df.loc[df['name'].isin(['All-Bran','Lucky Charms','Puffed Wheat'])]
# Normalize with weight

cereals = df.iloc[:,~df.columns.isin(['name','mfr','type','rating'])].div(df.weight, axis=0)

cereals = pd.concat([df.iloc[:,df.columns.isin(['name','mfr','type','rating'])] , cereals], axis=1)

cereals.head()
# Compute the correlation matrix

corr=df.iloc[:,~cereals.columns.isin(['name','mfr','type'])].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.lmplot(x="sugars",y="rating",data=cereals)

sns.jointplot(x="calories", y="rating", data=cereals)
cereals_scale = cereals



scaler = preprocessing.StandardScaler()

columns =cereals.columns[3:]

cereals_scale[columns] = scaler.fit_transform(cereals_scale[columns])

cereals_scale.head()
#Finding the average of good and bad ingredients

cereals_scale['Good'] = cereals_scale.loc[:,['protein','fiber','vitamins']].mean(axis=1)

#Good: the bigger the better.

cereals_scale['Bad'] = cereals_scale.loc[:,['fat','sodium','potass', 'sugars']].mean(axis=1)

#Multiply by negative to make this indicator the bigger the worse.

cereals_scale.loc[cereals_scale['name'].isin(['All-Bran','Lucky Charms','Puffed Wheat'])]
#Visualize the relacionship between the good/bad ingredients measure

ax = sns.lmplot('Good', # Horizontal axis

           'Bad', # Vertical axis

           data=cereals_scale, # Data source

           fit_reg=True, # Don't fix a regression line

           height = 10,

           aspect =2 ) # size and dimension



plt.title('Cereals Plot')

# Set x-axis label

plt.xlabel('Good')

# Set y-axis label

plt.ylabel('Bad')





def label_point(x, y, val, ax):

    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)

    for i, point in a.iterrows():

        if point['val'] == 'Lucky Charms':

            ax.text(point['x']+.02, point['y'], str(point['val']),bbox=dict(facecolor='red', alpha=0.5))

        else:

            ax.text(point['x']+.02, point['y'], str(point['val']))



label_point(cereals_scale.Good, cereals_scale.Bad, cereals_scale.name, plt.gca())  
#Finding a proxy of Sharpe Ratio good/bad to make a new ranking.

cereals_scale['new_ranking'] = cereals_scale['Good']/cereals_scale['Bad']



# Visualization new ranking

new_cereals = cereals_scale.sort_values(['new_ranking'],ascending=False).reset_index(drop=True)

plt.figure(figsize=(20,20))

sns.barplot(x=new_cereals["new_ranking"],y=new_cereals["name"])

plt.xlabel("Product Name",fontsize=15)

plt.ylabel("New Rating",fontsize=15)

plt.title("Product Rating",fontsize=15)

plt.show()
#Finding the most characteristic ingredient in each cereal

def knownby (row):

    maxValue = max(map(abs, pd.Series.tolist(row)[4:12]))

    try:

        index = pd.Series.tolist(row).index(maxValue)

    except ValueError:

        index = pd.Series.tolist(row).index(-maxValue)        

    return index



cereals_scale['knowby']=cereals_scale.apply(lambda row: knownby (row),axis=1)

cereals_scale['knowby']=cereals_scale.columns[cereals_scale['knowby']]

cereals_scale.loc[cereals_scale['name'].isin(['All-Bran','Lucky Charms','Puffed Wheat'])]
# Count of ingredients prevalence

ax = sns.countplot(x="knowby", data=cereals_scale, palette="Set3")