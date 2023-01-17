import pandas as pd



import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, linear_model, preprocessing, svm

from sklearn.preprocessing import StandardScaler, Normalizer

import math

import matplotlib

import seaborn as sns

def category_values(dataframe, categories):

    for c in categories:

        print('\n', dataframe.groupby(by=c)[c].count().sort_values(ascending=False))

        print('Nulls: ', dataframe[c].isnull().sum())



def plot_correlation_map( df ):

    corr = df.corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )



df = pd.read_csv('../input/autos.csv', sep=',', header=0, encoding='cp1252')

#df = pd.read_csv('autos.csv.gz', sep=',', header=0, compression='gzip',encoding='cp1252')

df.sample(10)
df.describe()
print(df.seller.unique())

print(df.offerType.unique())

print(df.abtest.unique())

print(df.nrOfPictures.unique())
df.drop(['seller', 'offerType', 'abtest', 'dateCrawled', 'nrOfPictures', 'lastSeen', 'postalCode', 'dateCreated'], axis='columns', inplace=True)

print("Too new: %d" % df.loc[df.yearOfRegistration >= 2017].count()['name'])

print("Too old: %d" % df.loc[df.yearOfRegistration < 1950].count()['name'])

print("Too cheap: %d" % df.loc[df.price < 100].count()['name'])

print("Too expensive: " , df.loc[df.price > 150000].count()['name'])

print("Too few km: " , df.loc[df.kilometer < 5000].count()['name'])

print("Too many km: " , df.loc[df.kilometer > 200000].count()['name'])

print("Too few PS: " , df.loc[df.powerPS < 10].count()['name'])

print("Too many PS: " , df.loc[df.powerPS > 500].count()['name'])

print("Fuel types: " , df['fuelType'].unique())

#print("Offer types: " , df['offerType'].unique())

#print("Sellers: " , df['seller'].unique())

print("Damages: " , df['notRepairedDamage'].unique())

#print("Pics: " , df['nrOfPictures'].unique()) # nrOfPictures : number of pictures in the ad (unfortunately this field contains everywhere a 0 and is thus useless (bug in crawler!) )

#print("Postale codes: " , df['postalCode'].unique())

print("Vehicle types: " , df['vehicleType'].unique())

print("Brands: " , df['brand'].unique())



# Cleaning data

#valid_models = df.dropna()



#### Removing the duplicates

dedups = df.drop_duplicates(['name','price','vehicleType','yearOfRegistration'

                         ,'gearbox','powerPS','model','kilometer','monthOfRegistration','fuelType'

                         ,'notRepairedDamage'])



#### Removing the outliers

dedups = dedups[

        (dedups.yearOfRegistration <= 2016) 

      & (dedups.yearOfRegistration >= 1950) 

      & (dedups.price >= 100) 

      & (dedups.price <= 150000) 

      & (dedups.powerPS >= 10) 

      & (dedups.powerPS <= 500)]



print("-----------------\nData kept for analisys: %d percent of the entire set\n-----------------" % (100 * dedups['name'].count() / df['name'].count()))

dedups.isnull().sum()
dedups['notRepairedDamage'].fillna(value='not-declared', inplace=True)

dedups['fuelType'].fillna(value='not-declared', inplace=True)

dedups['gearbox'].fillna(value='not-declared', inplace=True)

dedups['vehicleType'].fillna(value='not-declared', inplace=True)

dedups['model'].fillna(value='not-declared', inplace=True)
dedups.isnull().sum()
categories = ['gearbox', 'model', 'brand', 'vehicleType', 'fuelType', 'notRepairedDamage']



for i, c in enumerate(categories):

    v = dedups[c].unique()

    

    g = dedups.groupby(by=c)[c].count().sort_values(ascending=False)

    r = range(min(len(v), 5))



    print( g.head())

    plt.figure(figsize=(5,3))

    plt.bar(r, g.head()) 

    #plt.xticks(r, v)

    plt.xticks(r, g.index)

    plt.show()
dedups['namelen'] = [min(70, len(n)) for n in dedups['name']]



ax = sns.jointplot(x='namelen', 

                   y='price',

                   data=dedups[['namelen','price']], 

#                   data=dedups[['namelen','price']][dedups['model']=='golf'], 

                    alpha=0.1, 

                    size=8)

labels = ['name', 'gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']

les = {}



for l in labels:

    les[l] = preprocessing.LabelEncoder()

    les[l].fit(dedups[l])

    tr = les[l].transform(dedups[l]) 

    dedups.loc[:, l + '_feat'] = pd.Series(tr, index=dedups.index)



labeled = dedups[ ['price'

                        ,'yearOfRegistration'

                        ,'powerPS'

                        ,'kilometer'

                        ,'monthOfRegistration'

                        , 'namelen'] 

                    + [x+"_feat" for x in labels]]

len(labeled['name_feat'].unique()) / len(labeled['name_feat'])
labeled.drop(['name_feat'], axis='columns', inplace=True)
plot_correlation_map(labeled)

labeled.corr()
labeled.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]


Y = labeled['price']

X = labeled.drop(['price'], axis='columns', inplace=False)





matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"1. Before":Y, "2. After":np.log1p(Y)})

prices.hist()



Y = np.log1p(Y)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score, train_test_split



def cv_rmse(model, x, y):

    r = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = 5))

    return r



# Percent of the X array to use as training set. This implies that the rest will be test set

test_size = .33



#Split into train and validation

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state = 3)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)



r = range(2003, 2017)

km_year = 10000



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



rf = RandomForestRegressor()



param_grid = { "criterion" : ["mse"]

              , "min_samples_leaf" : [3]

              , "min_samples_split" : [3]

              , "max_depth": [10]

              , "n_estimators": [500]}



gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1)

gs = gs.fit(X_train, y_train)

print(gs.best_score_)

print(gs.best_params_)

 
bp = gs.best_params_

forest = RandomForestRegressor(criterion=bp['criterion'],

                              min_samples_leaf=bp['min_samples_leaf'],

                              min_samples_split=bp['min_samples_split'],

                              max_depth=bp['max_depth'],

                              n_estimators=bp['n_estimators'])

forest.fit(X_train, y_train)

# Explained variance score: 1 is perfect prediction

print('Score: %.2f' % forest.score(X_val, y_val))

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]

# Print the feature ranking

print("Feature ranking:")



for f in range(X.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



print(X_train.columns.values)

# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center",tick_label = X_train.columns.values)

plt.xticks(range(X.shape[1]), indices)

plt.xlim([-1, X.shape[1]])

plt.show()


