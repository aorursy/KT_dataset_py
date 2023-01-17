import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#create a dataframe from the wine csv
wine = pd.DataFrame.from_csv('../input/winemag-data-130k-v2.csv')
wine.shape
wine.head(5)
# reduce the data set for the fields relevant to the linear program
wine_reduced = wine.filter(['variety','winery','title', 'country','province','region_1','region_2','points','price' ], axis=1)
wine_reduced.head()
#lets see the distribution of points to price

plt.figure(figsize = (10, 6))
box = sns.boxplot(x='points', y='price', data=wine_reduced)
plt.setp(box.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)
box

#Based on the boxplot, there does seem to be an overall trend that higher rated wines also cost more.
#scatterplot will also help us understand the relationship between points and price

x = wine_reduced['points']
y = wine_reduced['price']
plt.scatter(x, y)
plt.show()


wine_reduced.corr()
#Lets cleanup the dataset. Because we need price and points. We will have to drop missing values before we do that, lets determine if there are duplicate rows
wine_reduced['dupes'] = wine_reduced.duplicated()
pd.value_counts(wine_reduced['dupes'].values, sort=False)
#Result of the dupes
wine_reduced.loc[wine_reduced['dupes'] == True]
#remove any duplicates from variety, winery and title fields
wine_dedup = wine_reduced.drop_duplicates(['variety', 'winery', 'title'])
#lets see if we eliminated all the dupes. We should only see False in the array
wine_dedup.dupes.unique()
#lets remove the NAs. 
wine_dedup.replace('', np.nan)
wine_clean = wine_dedup.dropna(subset=['points','price'], how = 'any')
wine_clean.head(20)
#lets drop the dupes field
wine_clean = wine_clean.drop('dupes',1)
wine_clean[:5]

#test model on first 50 rows
model_test = wine_clean[:50].copy()
model_test['price'] = model_test.price.astype(int)
#What is the dimension of our final data frame
model_test.shape
from scipy import optimize
# Note that since linprog only solves minimization problems, that sign of the cost function is inverted.
result = optimize.linprog(
    c = model_test['points']*-1, 
    A_ub=model_test['price'], 
    b_ub=[100],
    bounds=(0,1),
    method='simplex'
)
result.message
model_test['buy'] = result.x
model_test[model_test['buy']==1]
print ("Total Monies Spent: " + " " + str(model_test[model_test['buy']==1].price.sum()))
print ("The number of bottles of wine purchased:" + " " + str(len(model_test[model_test['buy']==1].index)))
cali = wine_clean[wine_clean['province']=='California']
cali = cali[cali['points'] >= 95]
cali.head(10)
#select the wine varities
cali = cali.loc[(cali['variety']=='Chardonnay') | (cali['variety'] == "Cabernet Sauvignon") |(cali['variety'] == "Pinot Gris") |  
           (cali['variety'] == "Pinot Grigio") | (cali['variety'] == "Red Blend") | (cali['variety'] == "Red Blends") | 
           (cali['variety'] == "Merlot") | (cali['variety'] == "Pinot Noir") | (cali['variety'] == "Rosé") ]
cali.variety.unique()
cali.shape
#lets set up our model and set a budget for 1000. This is going to be an awesome party!
result = optimize.linprog(
    c = cali['points']*-1, 
    A_ub=cali['price'], 
    b_ub=[1000],
    bounds=(0,1),
    method='simplex'
)
result.message
cali['buy'] = result.x
print ("Total Monies Spent :" + " " + str(cali[cali['buy']==1].price.sum()))
cali[cali['buy']==1]
print ("The number of bottles of wine purchased:" + " " + str(len(cali[cali['buy']==1].index)))