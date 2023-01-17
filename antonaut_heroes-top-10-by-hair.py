# Env: https://github.com/kaggle/docker-python

import pandas as pd



df_marvel = pd.read_csv('../input/marvel-wikia-data.csv')

df_dc = pd.read_csv('../input/dc-wikia-data.csv')



# Merge and clean up

df_marvel['src'] = 'marvel'

df_dc['src'] = 'dc'



df_dc['Year'] = df_dc['YEAR']

df_dc = df_dc.drop(columns=('YEAR'))

df_dc.head()

df_marvel.head()



df = df_marvel.append(df_dc, sort=False)



df.ALIVE = df.ALIVE == 'Living Characters'

df.EYE = df.EYE.str.split().str.get(0).astype('category')

df.HAIR = df.HAIR.str.split().str.get(0).astype('category')

df.HAIR = df.HAIR.str.replace('No', 'No hair (not bald)')

df.SEX = df.SEX.str.split().str.get(0).astype('category')

df.ALIGN = df.ALIGN.str.split().str.get(0).astype('category')



# Drop things we don't need

df_hair = df.drop(columns=['name', 'urlslug',

                 'page_id', 'Year',

                 'ID', 'ALIGN',

                 'APPEARANCES', 'GSM',

                 'EYE', 'SEX',

                 'FIRST APPEARANCE', 'src'])



df_alive = df_hair[df_hair.ALIVE]



# Create a df for plotting

pdf = pd.DataFrame({'alive': df_alive.HAIR.value_counts(), 'total': df_hair.HAIR.value_counts()})

pdf.index.names = ['hair']

pdf.reset_index(inplace=True)



total = pdf.sort_values('total', ascending=False) # for sorting the bars in the plot
# Plot, plot and plot!

import matplotlib.pyplot as plt

import seaborn as sns



sns.set(style='whitegrid')

f, ax = plt.subplots(figsize=(12, 5))



sns.set_color_codes('pastel')

sns.barplot(x='total', label='Total', y='hair', data=pdf, order=total['hair'].head(10), color='b')



sns.set_color_codes('muted')

sns.barplot(x='alive', label='Alive', y='hair', data=pdf, order=total['hair'].head(10), color='b')



ax.legend(ncol=2, loc='lower right', frameon=True)

ax.set(xlim=(0, 5400), ylabel="",

       xlabel="#of superheroes")

ax.set_title('Top 10: #of alive superheroes by hair color', fontsize=20)

sns.despine(left=True, bottom=True)
