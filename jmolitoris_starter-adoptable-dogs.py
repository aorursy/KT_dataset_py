# Importing relevant libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style = 'white')
df = pd.read_csv("../input/adoptable-dogs/ShelterDogs.csv")

df.head(10)
nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.isnull().sum()
# Pie chart of dogs' sexes

males = df.sex[df.sex=='male'].count()

females = df.sex[df.sex=='female'].count()



male_per = males/(males + females)*100

female_per = females/(males + females)*100



for i in set(df['size'].values):

    exec("{}_values = df['size'][df['size'] == i].count()".format(i))



small_per = small_values/len(df)*100

medium_per = medium_values/len(df)*100

large_per = large_values/len(df)*100



fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,8))

fig.tight_layout()

fig.suptitle("Percent Distribution of Sex and Size", fontsize = 24, y = 0.9)





colors1 = ['tab:blue', 'tab:pink']

ax1.pie([male_per,female_per], 

       pctdistance = 1.15, 

       autopct = '% .1f',

       shadow = True,

       textprops = {'color' : 'k', 'fontsize' : '12', 'ha' : 'center'},

       colors = colors1)

circle1 = plt.Circle(xy=(0,0), radius = 0.75, facecolor = 'w')

ax1.add_patch(circle1)

ax1.set_title('Sex of Dogs', fontsize = 18)



colors2 = ['tab:red', 'tab:purple', 'tab:blue']

ax2.pie([small_per, medium_per, large_per],

        pctdistance = 1.15, 

        autopct = '% .1f',

        shadow = True,

        textprops = {'color' : 'k', 'fontsize' : '12', 'ha' : 'center'},

        colors = colors2)

circle2 = plt.Circle(xy=(0,0), radius = 0.75, facecolor = 'w')

ax2.add_patch(circle2)

ax2.set_title('Size of Dogs', fontsize = 18)



ax1.legend(labels = ['Male', 'Female'], loc = 'center', edgecolor = 'w', facecolor = 'w', fontsize = 16)

ax2.legend(labels = ['Small', 'Medium', 'Large'], loc = 'center', edgecolor = 'w', facecolor = 'w', fontsize = 16)
bins = int(df['age'].max())



small =df['age'][df['size'] == 'small']

medium =df['age'][df['size'] == 'medium']

large =df['age'][df['size'] == 'large']



plt.figure(figsize = (8,6))

plt.hist([small, medium, large], 

         bins = bins, 

         stacked = True, 

         density = True,

         color = ["tab:red", "tab:purple", 'tab:blue'])

plt.xlabel('Age (in years)')

plt.ylabel('Density')

plt.legend(['Small', 'Medium', 'Large'], edgecolor = 'w', facecolor = 'w')

plt.title('Age Distribution of Dogs by their Size', fontsize = 16)
