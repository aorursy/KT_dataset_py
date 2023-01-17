import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
dd = pd.read_csv('../input/train.csv')



dd.head()



dd_test=pd.read_csv('../input/test.csv')
dd.describe()
dd.info()
dd.columns
dd.isnull().sum()
#Data Visualisation

sns.set_style("whitegrid")

sns.scatterplot(x=dd['Total Volume'],y=dd['AveragePrice'])

plt.show()
no_volume=len(dd[(dd['Total Volume']>20000000)])

no_volume
sns.set_style("whitegrid")

sns.scatterplot(x=dd['Total Bags'],y=dd['AveragePrice'])

plt.show()
no_bags=len(dd[(dd['Total Bags']>4000000)])

no_bags
no_bagsvol=len(dd[(dd['Total Bags']>7500000)&(dd['Total Volume']>20000000)])

no_bagsvol
# Compute the correlation matrix

corr = dd.corr(method="kendall")



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)



plt.show()
g = sns.FacetGrid(data=dd,  col="year")

g = g.map(plt.scatter, "Total Volume", "AveragePrice", edgecolor="w")

plt.show()
h = sns.FacetGrid(data=dd,  col="type")

h = h.map(plt.scatter, "Total Volume", "AveragePrice", edgecolor="w")

plt.show()
g = sns.FacetGrid(data=dd,  col="year")

g = g.map(plt.scatter, "Total Bags", "AveragePrice", edgecolor="w")

plt.show()
#Remove non-necessary columns

dd_new=dd[["id","Total Volume","Total Bags","type","AveragePrice"]]



dd_test_new=dd_test[["id","Total Volume","Total Bags","type"]]



dd_new.head()
dd_test_new.head()
from sklearn.preprocessing import StandardScaler

m = StandardScaler()

m.fit_transform(dd_new)

X = dd_new.drop(['AveragePrice'],axis=1)

Y = dd_new['AveragePrice'].tolist()
#Split Training data into test, train

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=4)
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=1,n_estimators=500,max_depth=25)

rfr.fit(x_train,y_train)
#Performance Metrics

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import explained_variance_score



def performance_metrics(y_true,y_pred):

    rmse = mean_absolute_error(y_true,y_pred)

    r2 = r2_score(y_true,y_pred)

    explained_var_score = explained_variance_score(y_true,y_pred)

    

    return rmse,r2,explained_var_score
y_pred = rfr.predict(x_test)



rmse,r2,explained_var_score = performance_metrics(y_test,y_pred)



print("Root mean squared error:{} \nR2-score:{} \nExplained variance score:{}".format(rmse,r2,explained_var_score))
pred4=rfr.predict(dd_test_new)

pred4
df = pd.DataFrame({"id": dd_test_new["id"], "AveragePrice": pred4})

new_df = df[['id', 'AveragePrice']]



new_df.to_csv("try4.csv", index=False)