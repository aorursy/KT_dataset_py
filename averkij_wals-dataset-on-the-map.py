from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.basemap import Basemap



from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

lang_df = pd.read_csv('/kaggle/input/language.csv')
lang_df.shape
major_families = lang_df.groupby('family').size().reset_index(name='count').sort_values(by='count', ascending=False).head(20)
major_families
ax = major_families.plot(kind='pie', y='count', subplots=True, figsize=(16, 12), labels=major_families['family'], legend=False)

altaic_langs = lang_df[lang_df['family'] == 'Altaic']
plt.figure(figsize=(22, 14))

earth = Basemap()

earth.bluemarble(alpha=0.02)

earth.drawcoastlines()

earth.drawstates()

earth.drawcountries()

earth.drawcoastlines(color='#555566', linewidth=1)

plt.scatter(altaic_langs.longitude, altaic_langs.latitude, c='red',alpha=1, zorder=10)

plt.xlabel("Altaic languages")

plt.savefig('altaic.png', dpi=350)
ie_langs = lang_df[lang_df['family'] == 'Indo-European']
plt.figure(figsize=(22, 14))

earth = Basemap()

earth.bluemarble(alpha=0.02)

earth.drawcoastlines()

earth.drawstates()

earth.drawcountries()

earth.drawcoastlines(color='#555566', linewidth=1)

plt.scatter(ie_langs.longitude, ie_langs.latitude, c='red',alpha=1, zorder=10)

plt.xlabel("Indo European languages")

plt.savefig('ie_langs.png', dpi=350)
earth = Basemap()



plt.figure(figsize=(22,14))

earth.bluemarble(alpha=0.02)

earth.drawcountries()

earth.drawcoastlines(color='#888888', linewidth=1)



sns.scatterplot(x='longitude', y='latitude', hue='family', data =lang_df, legend=False)



plt.title("Language map")

plt.show()

earth = Basemap()



plt.figure(figsize=(22,14))

earth.bluemarble(alpha=0.95)

#earth.drawcoastlines(color='#ffffff', linewidth=1)



sns.scatterplot(x='longitude', y='latitude', hue='family', data =lang_df, legend=False)



plt.title("Language map")

plt.show()
