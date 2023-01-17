%load_ext autoreload

%autoreload 2

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from numpy import log

import warnings

warnings.filterwarnings("ignore")





import pandas as pd



df = pd.concat([pd.read_csv("../input/200-financial-indicators-of-us-stocks-20142018/2014_Financial_Data.csv", index_col=0),

                pd.read_csv("../input/200-financial-indicators-of-us-stocks-20142018/2015_Financial_Data.csv", index_col=0),

                pd.read_csv("../input/200-financial-indicators-of-us-stocks-20142018/2016_Financial_Data.csv", index_col=0),

                pd.read_csv("../input/200-financial-indicators-of-us-stocks-20142018/2017_Financial_Data.csv", index_col=0)],

              sort=False).drop_duplicates()



df.head()
df.shape
df.describe()
plt.figure()



# to have a better view of the different entreprises, we will use the log scale



dflog = log(df[["Revenue","R&D Expenses"]] + 0.01)

dflog['Sector'] = df['Sector']

sns.catplot(x="Revenue", y="Sector", kind="box", data=dflog)

sns.catplot(x="R&D Expenses", y="Sector", kind="box", data=dflog)



plt.show()
# Price variation are the difference between a year and an other

X = df[['Sector', '2015 PRICE VAR [%]', '2016 PRICE VAR [%]', '2017 PRICE VAR [%]', '2018 PRICE VAR [%]']]

X=  X.rename(mapper={'2015 PRICE VAR [%]':'2015',

                     '2016 PRICE VAR [%]':'2016',

                     '2017 PRICE VAR [%]':'2017',

                     '2018 PRICE VAR [%]':'2018'}, axis=1)

X = X.fillna(0)

X.head()
g = sns.PairGrid(X[["2015","2016","2017","2018"]])

g.map(plt.scatter)
# just take a look at revenues between 0 and 1000:

x = X[X[["2015","2016","2017","2018"]]<1000]



g = sns.PairGrid(x[["2015","2016","2017","2018"]])

g.map(plt.scatter)
# Data are really ugly ^^
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=0).fit(X[["2015","2016","2017","2018"]])

X['label'] = kmeans.labels_
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure()

ax = Axes3D(fig, elev=48, azim=134)

ax.scatter(X["2015"], X["2016"], X["2017"],

               c=X['label'].astype(np.float), edgecolor='k')

ax.set_title("Price ")

ax.set_xlabel('2015')

ax.set_ylabel('2016')

ax.set_zlabel('2017')

plt.show()
g = sns.PairGrid(X[["2015","2016","2017","2018","label"]], hue="label")

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)

g.add_legend()

plt.show()