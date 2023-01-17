import pandas as pd

import numpy as np

from sklearn import datasets
iris = datasets.load_iris()

type(iris)
print(iris.DESCR)
iris.feature_names
#making a dataframe with the data and use column headers as feature names

dfiris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# looking at the first 5 entries

dfiris.head()

# the target column "Species" is still missing!
# get the two dataframes with the species ID and the species name

species = pd.DataFrame(data = iris.target, columns=["species_id"])

species_names = pd.DataFrame(data = iris.target_names, columns=['Species'])
# join them into one dataframe

species = species.join(species_names, how='left', on='species_id')
species.head()
# and finally join them to the overall dataframe



dfiris = dfiris.join(species)
dfiris.head()
#renaming columns for easier use

dfiris.columns
# making a dictionary to rename columns

newnames = {'sepal length (cm)':'sepal_length', 

            'sepal width (cm)':'sepal_width', 

            'petal length (cm)': 'petal_length',

       'petal width (cm)': 'petal_width'}
# renaming the columns, and keeping the new names in the dataframe --> hence "inplace"

dfiris.rename(columns=newnames, inplace=True)
# beautiful

dfiris.head()
# Note: Practice for slicing is also in R with [] operator

dfiris[dfiris.sepal_length > 5]
spec_list = dfiris.Species.unique()

spec_list


#little function to return n random items from a dataframe

def random_n_items(df, n):

    return df.sample(n=n)



#making an empty df first

dfiris_random_n = pd.DataFrame()



#with loop

for s in spec_list:

    dfout = random_n_items(dfiris[dfiris.Species==s],5)

    if dfiris_random_n.empty:

        dfiris_random_n=dfout

    else:

        dfiris_random_n = dfiris_random_n.append(dfout)

        
dfiris_random_n
# copy pasting is easier than typing - here's the column names

dfiris.columns
# only showing the relevant columns

selected_cols = ['petal_length','Species']

dfiris_random_n[selected_cols]
dfiris["Small sepal width"]=dfiris.sepal_width>3

dfiris.head()
relcols = ['sepal_length','sepal_width','petal_length','petal_width','Species']

dfiris[relcols].groupby(by="Species").mean()
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(10,6))

sns.set_style("whitegrid")

fig = sns.boxplot(data=dfiris, x='Species', y='sepal_length')

plt.figure(figsize=(10,6))

sns.scatterplot(data=dfiris, x="sepal_width", y="petal_width", hue='Species')
# getting 3rd setosa by petal length as a series (not a dataframe!) 

dfiris[dfiris.Species=='setosa'].sort_values(by="petal_length", ascending=False).iloc[2]
#function to get nth highest item for any feature passed in (returns a Series)



def get_nth(df, feature, n):

    return df.sort_values(by=feature, ascending=False).iloc[n-1].to_frame().transpose()

    
df_nth_item = pd.DataFrame()



feat = 'petal_length'

nth = 3



for s in dfiris.Species.unique():

    dfout = get_nth(dfiris[dfiris.Species==s],feat,nth)

    if df_nth_item.empty:

        df_nth_item = dfout

    else:

        df_nth_item = df_nth_item.append(dfout)

df_nth_item
# shouldn't change values in original dataframe, but make a copy first to avoid warning

dfiris['Species'].loc[(dfiris.petal_width < 1.4) & (dfiris.Species=="versicolor")]="new Species"
#nevertheless it works ;-)

dfiris.Species.unique()
plt.figure(figsize=(10,6))

sns.scatterplot(data=dfiris, x='petal_length', y='petal_width', hue='Species')
dfiris_short=dfiris[['Species','sepal_width']]

dfiris_short.head()
dfiris_short.sepal_width = dfiris_short.sepal_width.astype(int)
dfiris_short
dfiris_short_grouped = dfiris_short.groupby(by=['Species','sepal_width'])['Species'].count().to_frame()
# grouped items end up in index, easier to reset it

dfiris_short_grouped = dfiris_short_grouped.rename(columns={'Species': 'Count'})

dfiris_short_grouped[dfiris_short_grouped.Count>=10]