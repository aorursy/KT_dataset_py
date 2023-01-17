import pandas as pd



import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, linear_model, preprocessing, svm

from sklearn.preprocessing import StandardScaler, Normalizer

import math

import matplotlib

import seaborn as sns
df = pd.read_csv('../input/autos.csv', sep=',', header=0, encoding='cp1252')

df.head()
print("Too new: %d" % df.loc[df.yearOfRegistration >= 2017].count()['name'])

print("Too old: %d" % df.loc[df.yearOfRegistration < 1950].count()['name'])

print("Too cheap: %d" % df.loc[df.price < 100].count()['name'])

print("Too expensive: " , df.loc[df.price > 150000].count()['name'])

print("Too few km: " , df.loc[df.kilometer < 5000].count()['name'])

print("Too many km: " , df.loc[df.kilometer > 200000].count()['name'])

print("Too few PS: " , df.loc[df.powerPS < 10].count()['name'])

print("Too many PS: " , df.loc[df.powerPS > 500].count()['name'])

print("Fuel types: " , df['fuelType'].unique())

print("Offer types: " , df['offerType'].unique())

print("Sellers: " , df['seller'].unique())

print("Damages: " , df['notRepairedDamage'].unique())

print("Pics: " , df['nrOfPictures'].unique()) # nrOfPictures : number of pictures in the ad (unfortunately this field contains everywhere a 0 and is thus useless (bug in crawler!) )

print("Postale codes: " , df['postalCode'].unique())

print("Vehicle types: " , df['vehicleType'].unique())

print("Brands: " , df['brand'].unique())



#print("tourans: " , df[df['model']=='touran'][['name','vehicleType','powerPS','yearOfRegistration']])



# Cleaning data

valid_models = df.dropna()



#### Removing the duplicates

dedups = valid_models.drop_duplicates(['name','seller','offerType','price','abtest','vehicleType','yearOfRegistration'

                         ,'gearbox','powerPS','model','kilometer','monthOfRegistration','fuelType'

                         ,'notRepairedDamage','postalCode'])



#### Removing the outliers

no_outliers = dedups[

        (valid_models.yearOfRegistration <= 2016) 

      & (valid_models.yearOfRegistration >= 1950) 

      & (valid_models.price >= 100) 

      & (valid_models.price <= 150000) 

      & (valid_models.powerPS >= 10) 

      & (valid_models.powerPS <= 500)]



print("\nData kept for analisys: %d percent of the entire set" % (100 * no_outliers['name'].count() / df['name'].count()))
p = no_outliers

#plt.scatter(p['powerPS'], p['price'], alpha=.1, marker='o', color='b')

ax = sns.jointplot(x='powerPS', y='price',data=p[['powerPS','price']], alpha=0.1, size=8)

plt.title('Price distribution for car power')

#plt.xlabel('Power in HP')

#plt.ylabel('Price (logarithmic)')

#plt.yscale('log')



plt.xlim(0, 501)

plt.ylim(0, 160000)
p = no_outliers

#plt.scatter(p['yearOfRegistration'], p['price'], alpha=.1, marker='o', color='b')

ax = sns.jointplot(x='yearOfRegistration', 

                   y='price',

                   data=p[['yearOfRegistration','price']], 

                   alpha=0.1, size=8)



plt.title('Price distribution for year')

#plt.xlabel('Year')

#plt.ylabel('Price (logarithmic)')

#plt.yscale('log')



plt.ylim(0, 160000)
p = no_outliers

#plt.scatter(p['kilometer'], p['price'], alpha=.1, marker='o', color='b')

ax = sns.jointplot(x='kilometer', y='price',data=p[['kilometer','price']], alpha=0.1, size=8)



plt.title('Price distribution for kilometers')

#plt.xlabel('Kilometers')

#plt.ylabel('Price (logarithmic)')

#plt.yscale('log')

plt.ylim(0, 160000)

plt.xticks([5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,125000,150000], rotation='vertical')


# only relevant columns

rel_cols = no_outliers[['price'

                        ,'yearOfRegistration'

                        ,'gearbox'

                        ,'powerPS'

                        ,'model'

                        ,'kilometer'

                        ,'fuelType'

                        ,'vehicleType'

                        ,'monthOfRegistration'

                        ,'brand'

                        ,'notRepairedDamage']]

rel_cols.head()
labels = ['gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']

les = {}



for l in labels:

    les[l] = preprocessing.LabelEncoder()

    les[l].fit(rel_cols[l])

    #print(les[l].classes_)

    tr = les[l].transform(rel_cols[l]) 

    rel_cols.loc[:, l + '_feat'] = pd.Series(tr, index=rel_cols.index)



df_autos = rel_cols[ ['price'

                        ,'yearOfRegistration'

                        ,'powerPS'

                        ,'kilometer'

                        ,'monthOfRegistration'] 

                    + [x+"_feat" for x in labels]]

autos = df_autos.values.astype(float)



Y = autos[:,0]

X = autos[:,1:]
plt.title('Pearson Correlation')

sns.heatmap(df_autos.astype(float).corr(),

            linewidths=0.1,

            vmax=1.0, 

            square=True, 

            cmap="PuBuGn", 

            linecolor='w', 

            annot=False)
p = df_autos

#plt.scatter(p['powerPS'], p['price'], alpha=.1, marker='o', color='b')

ax = sns.jointplot(x='brand_feat', y='price',data=p[['brand_feat','price']], alpha=0.1, size=8)

plt.title('Price distribution for brands')

#plt.xlabel('Power in HP')

#plt.ylabel('Price (logarithmic)')

#plt.yscale('log')



brands = zip(range(0, len(les['brand'].classes_)), les['brand'].classes_)



s = ''

for b in brands:

    s = s + ' ' + str(b)



print("brands: " + s)



plt.xlim(0, 41)

plt.ylim(0, 160000)
p = df_autos[df_autos['brand_feat']==37]

#plt.scatter(p['powerPS'], p['price'], alpha=.1, marker='o', color='b')

ax = sns.jointplot(x='model_feat', y='price',data=p[['model_feat','price']], alpha=0.1, size=8)

plt.title('Price distribution for VW models')

#plt.xlabel('Power in HP')

#plt.ylabel('Price (logarithmic)')

#plt.yscale('log')



model = zip(range(0, len(les['model'].classes_)), les['model'].classes_)



s = ''

for b in model:

    s = s + ' ' + str(b)



print("models: " + s)



#plt.xlim(0, 41)

plt.ylim(0, 160000)
p = df_autos

f = plt.figure()



JG1 = sns.jointplot(x='fuelType_feat', y='price',data=p[['fuelType_feat','price']], alpha=0.1, size=4)

JG2 = sns.jointplot(x='vehicleType_feat', y='price',data=p[['vehicleType_feat','price']], alpha=0.1, size=4)

JG3 = sns.jointplot(x='gearbox_feat', y='price',data=p[['gearbox_feat','price']], alpha=0.1, size=4)

JG4 = sns.jointplot(x='notRepairedDamage_feat', y='price',data=p[['notRepairedDamage_feat','price']], alpha=0.1, size=4)



fuelTypes = zip(range(0, len(les['fuelType'].classes_)), les['fuelType'].classes_)

s = ''

for b in fuelTypes:

    s = s + ' ' + str(b)

print("Fuels: " + s)



vehicleTypes = zip(range(0, len(les['vehicleType'].classes_)), les['vehicleType'].classes_)

s = ''

for b in vehicleTypes:

    s = s + ' ' + str(b)

print("vehicleType: " + s)



gearboxes = zip(range(0, len(les['gearbox'].classes_)), les['gearbox'].classes_)

s = ''

for b in gearboxes:

    s = s + ' ' + str(b)

print("gearbox: " + s)



fuelTypes = zip(range(0, len(les['notRepairedDamage'].classes_)), les['notRepairedDamage'].classes_)

s = ''

for b in fuelTypes:

    s = s + ' ' + str(b)

print("notRepairedDamage: " + s)



#plt.xlim(0, 41)

plt.ylim(0, 160000)
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"1. Before":Y, "2. After":np.log1p(Y)})

prices.hist()



Y = np.log1p(Y)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score, train_test_split





def cv_rmse(model, x, y):

    r = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = 5))

    return r



r = range(2003, 2017)

km_year = 10000
linear = linear_model.LinearRegression()



# Percent of the X array to use as training set. This implies that the rest will be test set

test_size = .2



#Split into train and validation

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state = 3)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

linear.fit(X_train, y_train)



# Explained variance score: 1 is perfect prediction

score = linear.score(X_test, y_test)

print('Variance score: %.2f' % score)



print("rmse on validation set", cv_rmse(linear, X_test, y_test).mean())


def calc_values(model, year, brand, car_model, vehicle, gearbox, fuel, powerPS, km, not_repaired_damage):

    sample = [year

          , powerPS

          , km

          , 1 # month

          , les['gearbox'].transform([gearbox])

          , les['notRepairedDamage'].transform([not_repaired_damage])

          , les['model'].transform([car_model])

          , les['brand'].transform([brand])

          , les['fuelType'].transform([fuel])

          , les['vehicleType'].transform([vehicle])]

    s_predict = np.expm1(model.predict([sample]))

    return s_predict



# [nan 'coupe' 'suv' 'kleinwagen' 'limousine' 'cabrio' 'bus' 'kombi' 'andere']

vw1 = calc_values(linear, 2016, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 105, 10000, 'nein')

bmw  = calc_values(linear, 2016, 'bmw', '3er', 'limousine', 'manuell', 'diesel', 150, 10000, 'nein')

fiat  = calc_values(linear, 2016, 'fiat', 'punto', 'kleinwagen', 'manuell', 'benzin', 60, 10000, 'nein')

vw2 = calc_values(linear, 2016, 'volkswagen', 'tiguan', 'suv', 'manuell', 'diesel', 130, 10000, 'nein')



print(vw1)

print(bmw)

print(fiat)

print(vw2)



values_linear  = [calc_values(linear, y, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 120, km_year * (2017-y), 'nein') for y in r]

values_linear2  = [calc_values(linear, y, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 120, km_year * 2 * (2017-y), 'nein') for y in r]

values_linear3  = [calc_values(linear, y, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 120, km_year * 3 * (2017-y), 'nein') for y in r]

plt.plot(r, values_linear, label='linear')

plt.plot(r, values_linear2, label='linear 20000km_x_y')

plt.plot(r, values_linear3, label='linear 30000km_x_y')



#plt.grid()

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


#alphas = [0.005,0.05, 0.1, 0.3, 0.5, 0.7, 1, 2, 3, 5, 10, 20, 25, 30, 50]

#cv_lasso = [cv_rmse(Lasso(alpha = alpha), X_train, y_train).mean() for alpha in alphas]



#cv_lasso = pd.Series(cv_lasso, index = alphas)

#cv_lasso.plot(title = "Validation", figsize=(5,5))

#plt.xlabel("alpha")

#plt.ylabel("rmse")



#model_lasso = Lasso(alpha = cv_lasso.min()).fit(X_train, y_train)



# Explained variance score: 1 is perfect prediction

#print('Variance score: %.2f' % model_lasso.score(X_test, y_test))



#coef = pd.Series(model_lasso.coef_, index = df_autos.iloc[:,1:].columns)



#imp_coef = pd.concat([coef.sort_values()])



#print(imp_coef)



#print(np.sqrt(np.mean((y_test - model_lasso.predict(X_test))**2)))

#print("rmse on validation set", cv_rmse(model_lasso, X_test, y_test).mean())
#vw1 = calc_values(model_lasso, 2016, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 105, 10000, 'nein')

#bmw  = calc_values(model_lasso, 2016, 'bmw', '3er', 'limousine', 'manuell', 'diesel', 150, 10000, 'nein')

#fiat  = calc_values(model_lasso, 2016, 'fiat', 'punto', 'kleinwagen', 'manuell', 'benzin', 60, 10000, 'nein')

#vw2 = calc_values(model_lasso, 2016, 'volkswagen', 'tiguan', 'suv', 'manuell', 'diesel', 130, 10000, 'nein')



#print(vw1)

#print(bmw)

#print(fiat)

#print(vw2)



#values_lasso  = [calc_values(model_lasso, y, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 120, km_year * (2017-y), 'nein') for y in r]

#values_lasso2  = [calc_values(model_lasso, y, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 120, km_year * 2 * (2017-y), 'nein') for y in r]

#values_lasso3  = [calc_values(model_lasso, y, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 120, km_year * 3 * (2017-y), 'nein') for y in r]

#plt.plot(r, values_lasso, label='lasso')

#plt.plot(r, values_lasso2, label='lasso 20000km_x_y')

#plt.plot(r, values_lasso3, label='lasso 30000km_x_y')



#plt.grid()

#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


alphas = [-10, -5, -2, -1, 0, 0.005,0.05, 0.1, 0.3, 1, 3, 5, 10, 20, 25, 30, 50]

cv_ridge = [cv_rmse(Ridge(alpha = alpha), X_train, y_train).mean() for alpha in alphas]



cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation", figsize=(5,5))

plt.xlabel("alpha")

plt.ylabel("rmse")



model_ridge = Ridge(alpha = cv_ridge.min()).fit(X_train, y_train)



# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % model_ridge.score(X_test, y_test))



coef = pd.Series(model_ridge.coef_, index = df_autos.iloc[:,1:].columns)



imp_coef = pd.concat([coef.sort_values()])



print(imp_coef)



print(np.sqrt(np.mean((y_test - model_ridge.predict(X_test))**2)))

print("rmse on validation set", cv_rmse(model_ridge, X_test, y_test).mean())
vw1 = calc_values(model_ridge, 2016, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 105, 10000, 'nein')

bmw  = calc_values(model_ridge, 2016, 'bmw', '3er', 'limousine', 'manuell', 'diesel', 150, 10000, 'nein')

fiat  = calc_values(model_ridge, 2016, 'fiat', 'punto', 'kleinwagen', 'manuell', 'benzin', 60, 10000, 'nein')

vw2 = calc_values(model_ridge, 2016, 'volkswagen', 'tiguan', 'suv', 'manuell', 'diesel', 130, 10000, 'nein')



print(vw1)

print(bmw)

print(fiat)

print(vw2)



values_ridge  = [calc_values(model_ridge, y, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 120, km_year * (2017-y), 'nein') for y in r]

values_ridge2  = [calc_values(model_ridge, y, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 120, km_year * 2 * (2017-y), 'nein') for y in r]

values_ridge3  = [calc_values(model_ridge, y, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 120, km_year * 3 * (2017-y), 'nein') for y in r]

plt.plot(r, values_ridge, label='ridge')

plt.plot(r, values_ridge2, label='ridge 20000km_x_y')

plt.plot(r, values_ridge3, label='ridge 30000km_x_y')



#plt.grid()

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators = 20)

forest.fit(X_train, y_train)
# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % forest.score(X_test, y_test))



print(np.sqrt(np.mean((y_test - forest.predict(X_test))**2)))

print("rmse on validation set", cv_rmse(forest, X_test, y_test).mean())
vw1 = calc_values(forest, 2016, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 105, 10000, 'nein')

bmw  = calc_values(forest, 2016, 'bmw', '3er', 'limousine', 'manuell', 'diesel', 150, 10000, 'nein')

fiat  = calc_values(forest, 2016, 'fiat', 'punto', 'kleinwagen', 'manuell', 'benzin', 60, 10000, 'nein')

vw2 = calc_values(forest, 2016, 'volkswagen', 'tiguan', 'suv', 'manuell', 'diesel', 130, 10000, 'nein')



print(vw1)

print(bmw)

print(fiat)

print(vw2)



values_forest = [calc_values(forest     , y, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 120, km_year * (2017-y), 'nein') for y in r]

values_forest2 = [calc_values(forest     , y, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 120, km_year* 2 * (2017-y), 'nein') for y in r]

values_forest3 = [calc_values(forest     , y, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 120, km_year* 3 * (2017-y), 'nein') for y in r]



plt.plot(r, values_forest, label='forest')

plt.plot(r, values_forest2, label='forest 20000km_x_y')

plt.plot(r, values_forest3, label='forest 30000km_x_y')



#plt.grid()

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
from xgboost import XGBRegressor

import xgboost as xgb

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold



clf = XGBRegressor()

clf.fit(X_train, y_train)
xgb.plot_importance(clf)



#f0 = year

#f1 = powerPS

#f2 = km

#f3 = month

#f4 = gearbox

#f5 = notRepairedDamage

#f6 = model

#f7 = brand

#f8 = fuelType

#f9 = vehicleType



# make predictions for test data

y_pred = clf.predict(X_test)

print(np.sqrt(np.mean((y_test - y_pred)**2)))
values_clf  = [calc_values(clf, y, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 115, km_year * (2017-y), 'nein') for y in r]

values_clf2 = [calc_values(clf, y, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 115, km_year * 2 * (2017-y), 'nein') for y in r]

values_clf3 = [calc_values(clf, y, 'volkswagen', 'touran', 'bus', 'manuell', 'diesel', 115, km_year * 3 * (2017-y), 'nein') for y in r]

plt.plot(r, values_clf , label='xgboost')

plt.plot(r, values_clf2, label='xgboost 20000km_x_y')

plt.plot(r, values_clf3, label='xgboost 30000km_x_y')



#plt.grid()

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)