import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv('/kaggle/input/child-labour-in-inida/Child Labour in India.csv')
df.head(3)
df.tail(3)
print('Number of rows in the dataset: ',df.shape[0])

print('Number of columns in the dataset: ',df.shape[1])
df.info()
df.rename(columns={

    'Category of States':'Category'

}, inplace=True)
# Renaming the values present in the category column

df['Category'] = df['Category'].replace(['Non Special Category states', 'Special Category States'],

                                      ['Non Special', 'Special'])
df['Manufacturing'] =df['Manufacturing'].replace('9. 9', '9.9')

df['Manufacturing'] = df['Manufacturing'].astype('float')
df.drop('Total', axis=1, inplace=True)
india = df.loc[df['Category'] == 'All India']

india
df =df[df['Category'] != 'All India']
df['States'].nunique()
special =len(df[df['Category'] == 'Special'])

non_special = len(df[df['Category']== 'Non Special'])



plt.figure(figsize=(8,6))



# Data to plot

labels = 'Special','Non Special'

sizes = [special, non_special]

colors = ['skyblue', 'yellowgreen']

explode = (0, 0.2)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True)

plt.title('Percentage of Special and Non Special States in the Dataset')

plt.axis('equal')

plt.show()
sns.set_style('darkgrid')

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16,8), sharey=True)

sns.distplot(df['Agriculture'], kde=False, bins=15,color='red', hist_kws=dict(edgecolor="black", linewidth=2),ax=ax[0][0])

sns.distplot(df['Manufacturing'], kde=False, bins=15,color='blue', hist_kws=dict(edgecolor="black", linewidth=2),ax=ax[0][1])

sns.distplot(df['Construction'], kde=False, bins=15,color='green', hist_kws=dict(edgecolor="black", linewidth=2),ax=ax[0][2])

sns.distplot(df['Trade Hotels & Restaurants'], kde=False, bins=15,color='purple', hist_kws=dict(edgecolor="black", linewidth=2),ax=ax[1][0])

sns.distplot(df['Community, Social and Personal Services'], kde=False, bins=15,color='aqua', hist_kws=dict(edgecolor="black", linewidth=2),ax=ax[1][1])

sns.distplot(df['Others'], kde=False, bins=15,color='gold', hist_kws=dict(edgecolor="black", linewidth=2),ax=ax[1][2])

plt.tight_layout()
# Function to Draw pie chart in terms of Special and Non Special Categories

def draw_piechart(feature):

    special =df[df['Category'] == 'Special'][feature].mean()

    non_special =df[df['Category']== 'Non Special'][feature].mean()



    plt.figure(figsize=(8,6))



    # Data to plot

    labels = 'Special','Non Special'

    sizes = [special, non_special]

    colors = ['skyblue', 'yellowgreen']

    explode = (0, 0.1)  # explode 1st slice



    # Plot

    plt.pie(sizes, explode=explode, labels=labels, colors=colors,

    autopct='%1.1f%%', shadow=True)

    plt.title('Percentage of children employed in ' +feature+' category in Special and Non Special category States')

    plt.axis('equal')

    plt.show()
#Function to draw a barchart in terms of states

def draw_barchart(feature):

    plt.figure(figsize=(15, 6))

    sns.barplot(x='States', y=feature, data=df, edgecolor='black', order=list(df.sort_values(by=feature,ascending=False)['States']))

    plt.title('Percentage of Children employed in ' + feature + ' category in various states')

    plt.xlabel('States')

    plt.ylabel('Percentage of children working')

    plt.xticks(rotation=90)

    plt.show()
draw_piechart('Agriculture')
draw_barchart('Agriculture')
draw_piechart('Manufacturing')
draw_barchart('Manufacturing')
draw_piechart('Construction')
draw_barchart('Construction')
draw_piechart('Trade Hotels & Restaurants')
draw_barchart('Trade Hotels & Restaurants')
draw_piechart('Community, Social and Personal Services')
draw_barchart('Community, Social and Personal Services')
draw_piechart('Others')
draw_barchart('Others')
fig,ax = plt.subplots(figsize=(17,8))

ax.bar(df['States'],df['Agriculture'],color='#70C1B3',label='Agriculture')

ax.bar(df['States'], df['Manufacturing'], bottom=df['Agriculture'], color='#247BA0', label='Manufacturing')

ax.bar(df['States'], df['Construction'], bottom=df['Agriculture']+df['Manufacturing'], color='#FFE066',label='Construction')

ax.bar(df['States'], df['Trade Hotels & Restaurants'], bottom=df['Agriculture']+df['Manufacturing']+df['Construction'], color='#F25F5C', label='Trade Hotels & Restaurants')

ax.bar(df['States'], df['Community, Social and Personal Services'], 

       bottom=df['Agriculture']+df['Manufacturing']+df['Construction'] + df['Trade Hotels & Restaurants'], color='#50514F', label='Community, Social and Personal Services')



ax.bar(df['States'], df['Others'], 

       bottom=df['Agriculture']+df['Manufacturing']+df['Construction'] + df['Trade Hotels & Restaurants']+df['Community, Social and Personal Services'],

       color='#A1CF6B', label='Others')



# ax.bar(df['States'],df['Other'])

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,

       ncol=3, mode="expand", borderaxespad=0.)

plt.xticks(rotation=90)

plt.ylim((0, 110))

plt.show()