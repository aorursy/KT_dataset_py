import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

cereal = pd.read_csv("../input/cereal.csv")
cereal.describe(include = ['O'])
Manufacturer_of_cereals = cereal['mfr'].unique()

mfr_count = []

for mnfc in Manufacturer_of_cereals:

    mfr_count.append(cereal[cereal['mfr'] == mnfc]['mfr'].count())
x = np.arange(len(Manufacturer_of_cereals))

plt.figure(figsize=(10,8))

plt.bar(x, mfr_count, align='center', alpha=0.5, edgecolor = 'black')

plt.xticks(x, Manufacturer_of_cereals, fontsize=18 )

plt.ylabel('Number of cereals a manufacturer produces', fontsize=18 )

plt.title('Manufacturers of cereals ', fontsize=18)

 

plt.show()





explode = (0, 0, 0.1, 0, 0, 0, 0)  # explode 1st slice

 

plt.figure(figsize=(8,8))

plt.rcParams['font.size'] = 16

plt.pie(mfr_count, 

        explode=explode, 

        labels=Manufacturer_of_cereals, 

        autopct='%1.1f%%', 

        shadow=True, 

        startangle = 140 )

 

plt.axis('equal')

plt.tight_layout()

plt.show()

                 