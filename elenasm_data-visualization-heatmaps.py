import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



avocado_db = pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')

avocado_db.head()
avocado_db = avocado_db.groupby('year').sum()

avocado_db.round(0)

avocado_db = avocado_db.reset_index()

avocado_db = avocado_db[avocado_db['year'] < 2018]

avocado_db.head()

avocado_db.round(0)
year = avocado_db['year'].astype(str)

volume = avocado_db['Total Volume']/10000



plt.title('Volume by year')

plt.xlabel('year')

plt.ylabel('volume (/ 10k)')

plt.plot(year,volume)
plt.title('Volume by year')

plt.xlabel('year')

plt.ylabel('volume (/ 10k)')

plt.scatter(year,volume, alpha = 0.5)
plt.bar(year, volume, align='center', alpha=0.5)


from numpy import *

from matplotlib.pyplot import *

from numpy.random import *



plt.title('Sales per type of bag by year')

plt.xlabel('year')

plt.ylabel('volume (/ 10k)')

small_bags = avocado_db['Small Bags']/10000

large_bags = avocado_db['Large Bags']/10000

xlarge_bags = avocado_db['XLarge Bags']/10000

plt.plot(year,small_bags, c = 'k', label = 'small bags')

plt.plot(year,large_bags, c = 'b', label = 'large bags')

plt.plot(year,xlarge_bags, c = 'r', label = 'very large bags')

legend()
N = 3



ind = np.arange(N) 

width = 0.35       

plt.bar(ind, small_bags, width, label='small')

plt.bar(ind + width, large_bags, width,

    label='large')

plt.bar(ind + width, xlarge_bags, width,

    label='xlarge')





plt.ylabel('sales')

plt.title('sales')



plt.xticks(ind + width, ('2015', '2016', '2017'))

plt.legend(loc='best')

plt.show()
years = ['2015', '2016', '2017']



ind = [x for x, _ in enumerate(years)]



plt.bar(ind, small_bags, width=0.8, label='small', color='gold', bottom=large_bags+xlarge_bags)

plt.bar(ind, large_bags, width=0.8, label='large', color='silver', bottom=xlarge_bags)

plt.bar(ind, xlarge_bags, width=0.8, label='xlarge', color='#CD853F')



plt.xticks(ind, years)

plt.ylabel("sales")

plt.xlabel("years")

plt.legend(loc="upper left")

plt.title("Sales per year")



plt.show()
avocado_db2 = pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')

avocado_db2.dropna(thresh=1)

avocados_4225 = avocado_db2[['4225','Large Bags','Small Bags','XLarge Bags']]

avocados_4046 = avocado_db2[['4046','Large Bags','Small Bags','XLarge Bags']]

avocados_4770 = avocado_db2[['4770','Large Bags','Small Bags','XLarge Bags']]
import seaborn as sns

correlation = avocados_4225.corr(method = 'pearson')

ax = sns.heatmap(correlation)
correlation = avocados_4046.corr(method = 'pearson')

ax = sns.heatmap(correlation)
correlation = avocados_4770.corr(method = 'pearson')

ax = sns.heatmap(correlation)