import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt
data = pd.DataFrame(pd.read_csv("../input/world-happiness-data/world_happiness.csv"))

data.head()
data.columns = data.columns.str.strip().str.lower().str.replace('.', '_')

data.head(2)
data.shape
data.columns = ['continent','country', 'happiness_rank', 'happiness_score', 'whisker_high', 'whisker_low', 'economy','family','health','freedom','generosity','trust','dystopia_residual'] 

data.head(2)
data.drop(data.columns[[4,5]], axis = 1, inplace = True)

data.head(2)
data.shape
data.columns
data.describe()
data.info()
print('Number of Null values in Columns')

data.isnull().sum()
corrMatrix = data.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
data.drop(data.columns[[2]], axis = 1, inplace = True)

data.head(2)
corrMatrix = data.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
data.head(2)
#asia

asia = data["continent"] == "Asia"

asia_df = data[asia]

asia_data = asia_df.mean()

asia_mean = asia_data['happiness_score']

print("Mean of happiness score -", asia_mean)

asia_df.head(2)
#africa

africa = data["continent"] == "Africa"

africa_df = data[africa]

africa_data = africa_df.mean()

africa_mean = africa_data['happiness_score']

print("Mean of happiness score -", africa_mean)

africa_df.head(2)
#europe

europe = data["continent"] == "Europe"

europe_df = data[europe]

europe_data = europe_df.mean()

europe_mean = europe_data['happiness_score']

print("Mean of happiness score -", europe_mean)

europe_df.head(2)
#north america

nm = data["continent"] == "North America"

nm_df = data[nm]

nm_data = nm_df.mean()

nm_mean = nm_data['happiness_score']

print("Mean of happiness score -", nm_mean)

nm_df.head(2)
#south america

sm = data["continent"] == "South America"

sm_df = data[sm]

sm_data = sm_df.mean()

sm_mean = sm_data['happiness_score']

print("Mean of happiness score -", sm_mean)

sm_df.head(2)
#Australia

aus = data["continent"] == "Australia"

aus_df = data[aus]

aus_data = aus_df.mean()

aus_mean = aus_data['happiness_score']

print("Mean of happiness score -", aus_mean)

aus_df.head(2)
corr_asia = asia_df.corr()



mask = np.zeros_like(corr_asia)

mask[np.triu_indices_from(mask)] = True

ax = plt.axes()

with sns.axes_style("white"):

    p2 = sns.heatmap(corr_asia, mask=mask, square=True, annot = True, ax=ax)

ax.set_title('Happiness Matrix for Asia')
corr_africa = africa_df.corr()



mask = np.zeros_like(corr_africa)

mask[np.triu_indices_from(mask)] = True

ax = plt.axes()

with sns.axes_style("white"):

    p2 = sns.heatmap(corr_africa, mask=mask, square=True, annot = True, ax=ax)

ax.set_title('Happiness Matrix for Africa')
corr_europe = europe_df.corr()



mask = np.zeros_like(corr_europe)

mask[np.triu_indices_from(mask)] = True

ax = plt.axes()

with sns.axes_style("white"):

    p2 = sns.heatmap(corr_europe, mask=mask, square=True, annot = True, ax=ax)

ax.set_title('Happiness Matrix for Europe')
corr_nm = nm_df.corr()



mask = np.zeros_like(corr_nm)

mask[np.triu_indices_from(mask)] = True

ax = plt.axes()

with sns.axes_style("white"):

    p2 = sns.heatmap(corr_nm, mask=mask, square=True, annot = True, ax=ax)

ax.set_title('Happiness Matrix for North America')
corr_sm = sm_df.corr()



mask = np.zeros_like(corr_sm)

mask[np.triu_indices_from(mask)] = True

ax = plt.axes()

with sns.axes_style("white"):

    p2 = sns.heatmap(corr_sm, mask=mask, square=True, annot = True, ax=ax)

ax.set_title('Happiness Matrix for South America')
corr_aus = aus_df.corr()



mask = np.zeros_like(corr_aus)

mask[np.triu_indices_from(mask)] = True

ax = plt.axes()

with sns.axes_style("white"):

    p2 = sns.heatmap(corr_aus, mask=mask, square=True, annot = True, ax=ax)

ax.set_title('Happiness Matrix for Australia')
clr = {"Europe": "b", "North America": "g", "Australia":"r", "Asia": "c", "South America": "m", "Africa":"y"}
plt.scatter(data["continent"], data["happiness_score"], cmap=clr )

plt.xticks(rotation=15)
sns.boxplot( x=data["continent"], y=data["happiness_score"], palette=clr )

plt.xticks(rotation=15)
sns.violinplot( x=data["continent"], y=data["happiness_score"], width=25, palette=clr )

plt.xticks(rotation=15)
grid = sns.pairplot(data, hue = 'continent', diag_kind = 'kde',

             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},

             size = 4)

grid

grid.savefig('PairPlot.png')
data.head(2)
#ENCODING 

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data[data.columns[0]] = labelencoder.fit_transform(data[data.columns[0]])

data[data.columns[1]] = labelencoder.fit_transform(data[data.columns[1]])
data.head(2)
X = data[data.columns[[0,1,3,4,5,6,7,8,9]]]

y = data[data.columns[2]]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)



reg=LinearRegression()
reg.fit(X_train,y_train)
reg.score(X_train,y_train)
pdt = reg.predict(X_test)

pdt
from sklearn.metrics import mean_squared_error

from math import sqrt



rms = sqrt(mean_squared_error(y_test,pdt))

rms
plt.scatter(pdt, y_test, color = 'red') 



# plot predicted data 

plt.plot(pdt, y_test, color = 'blue') 