

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from apyori import apriori

#from mlxtend.frequent_patterns import apriori, association_rules

from scipy.stats import norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input/house-prices-advanced-regression-techniques"))



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

%matplotlib inline



# Read files

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)



# From EDA obvious outliers

train = train[train.GrLivArea < 4500]

train.reset_index(drop=True, inplace=True)

outliers = [30, 88, 462, 631, 1322]

train = train.drop(train.index[outliers])





print (train.columns)

print(test.columns)

print(train.shape,test.shape)

 
#Resumen 

train.describe()
#Primeras filas

train.head(10)
#Resumen de analisis estadistico variable Precio de Venta

train['SalePrice'].describe()

#Histograma

sns.distplot(train['SalePrice']);
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#Grafico totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#Grafico de bigotes overallqual/saleprice

var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(12, 10))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 12))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=600000);

plt.xticks(rotation=90);
corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 3)

plt.show();
var = 'YearBuilt'

df = pd.concat([train['SalePrice'], train[var]], axis=1)



for k in range (1, 11):

 

	# Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.

	kmeans_model = KMeans(n_clusters=k, random_state=1).fit(df)

	

	# These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.

	labels = kmeans_model.labels_

 

	# Sum of distances of samples to their closest cluster center

	interia = kmeans_model.inertia_

	print( "k:",k, " cost:", interia)

    

kmeans = KMeans(n_clusters=5).fit(df)

centroids = kmeans.cluster_centers_

plt.scatter(df['SalePrice'], df['YearBuilt'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)

plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

plt.show()
df1 = pd.concat([train['SalePrice'], train['OverallQual'], train['GrLivArea'], train['GarageCars'], train['TotalBsmtSF'], train['FullBath'], train['YearBuilt']], axis=1)

records = [] 

for i in range(0, 1450):

    records.append([str(df1.values[i,j]) for j in range(0, 7)])

association_rules = apriori(records, min_support=0.2, min_confidence=0.2, min_lift=1, min_length=2)

association_results = list(association_rules)

for item in association_results:



    # first index of the inner list

    # Contains base item and add item

    pair = item[0] 

    items = [x for x in pair]

    print("Rule: " + items[0] + " -> " + items[0])



    #second index of the inner list

    print("Support: " + str(item[1]))



    #third index of the list located at 0th

    #of the third index of the inner list



    print("Confidence: " + str(item[2][0][2]))

    print("Lift: " + str(item[2][0][3]))

    print("=====================================")
