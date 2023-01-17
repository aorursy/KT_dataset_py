import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler, LabelEncoder

from scipy import stats

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from matplotlib import rcParams

from collections import defaultdict



sf = pd.read_csv("../input/train.csv")

sf.head()
sf.groupby("Category")["Category"].count().sort_values(ascending=False)
sf.groupby("PdDistrict")["PdDistrict"].count().sort_values(ascending=False)
sf.groupby("DayOfWeek")["DayOfWeek"].count().sort_values(ascending=False)
d = defaultdict(LabelEncoder)

sf_encode = sf.apply(lambda x: d[x.name].fit_transform(x))

sf_encode = sf_encode.drop(['X', 'Y'], axis=1)



corrmat = sf_encode.corr()

f, ax = plt.subplots(figsize=(12, 12))

plot2 =sns.heatmap(corrmat, vmax=.8);

plt.xticks(rotation=90)

plt.yticks(rotation=45)

plot2.axes.set_title('Correlation Heat Map')

sns.plt.show()
cmap1 = sns.cubehelix_palette(as_cmap=True)

k = 8

cols = corrmat.nlargest(k, 'Category')['Category'].index

cm = np.corrcoef(sf_encode[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True,cmap=cmap1, square=True, annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

hm.axes.set_title('Correlation Matrix')

plt.xticks(rotation=90)

plt.yticks(rotation=45)

plt.show()
most_dangerous_districts = sf.PdDistrict.value_counts()

_n_crime_plot = sns.barplot(x=most_dangerous_districts.index,y=most_dangerous_districts)

_n_crime_plot.set_xticklabels(most_dangerous_districts.index,rotation=90)
number_of_crimes = sf.Category.value_counts()



_n_crime_plot = sns.barplot(x=number_of_crimes.index,y=number_of_crimes)

_n_crime_plot.set_xticklabels(number_of_crimes.index,rotation=90)
pt = pd.pivot_table(sf,index="PdDistrict",columns="Category",aggfunc=len,fill_value=0)["Dates"]

_ = pt.loc[most_dangerous_districts.index,number_of_crimes.index]

ax = sns.heatmap(_)

ax.set_title("Number of Crimes per District")