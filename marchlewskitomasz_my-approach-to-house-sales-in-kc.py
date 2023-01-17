from __future__ import division

import pandas as pd

import sklearn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import eli5
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
%matplotlib inline

plt.rcParams['figure.figsize'] = 12, 8 # universal plot size

pd.options.mode.chained_assignment = None  # default='warn', disables pandas warnings about assigments

njobs = 2 # number of jobs

sbr_c = "#1156bf" # seaborn plot color
data = pd.read_csv('../input/kc_house_data.csv', iterator=False, parse_dates=['date'])
data.head(10) # to see the columns and first 10 rows
data.info() # overview of the data
data['date'].dt.year.hist() 

plt.title('Year of pricing distribution')

plt.show()
data.describe() # overview of the data
data['price'].hist(xrot=30, bins=500) 

plt.title('Price distribution')

plt.show()
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize = (12, 15))

sns.stripplot(x = "grade", y = "price", data = data, jitter=True, ax = ax1, color=sbr_c)

sns.stripplot(x = "view", y = "price", data = data, jitter=True, ax = ax2, color=sbr_c)

sns.stripplot(x = "bedrooms", y = "price", data = data, jitter=True, ax = ax3, color=sbr_c)

sns.stripplot(x = "bathrooms", y = "price", data = data, jitter=True, ax = ax4, color=sbr_c)

sns.stripplot(x = "condition", y = "price", data = data, jitter=True, ax = ax5, color=sbr_c)

sns.stripplot(x = "floors", y = "price", data = data, jitter=True, ax = ax6, color=sbr_c)

ax4.set_xticklabels(ax4.get_xticklabels(), rotation=60)

for i in range(1,7):

    a = eval('ax'+str(i))

    a.set_yscale('log')

plt.tight_layout()
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize = (12, 12))

sns.regplot(x = 'sqft_living', y = 'price', data = data, ax = ax1, fit_reg=False, scatter_kws={"s": 1})

sns.regplot(x = 'sqft_lot', y = 'price', data = data, ax = ax2, fit_reg=False, scatter_kws={"s": 1})

sns.regplot(x = 'yr_built', y = 'price', data = data, ax = ax5, fit_reg=False, scatter_kws={"s": 1})

sns.regplot(x = 'sqft_basement', y = 'price', data = data, ax = ax6, fit_reg=False, scatter_kws={"s": 1})

sns.regplot(x = 'lat', y = 'price', data = data, ax = ax3, fit_reg=False, scatter_kws={"s": 1})

sns.regplot(x = 'long', y = 'price', data = data, ax = ax4, fit_reg=False, scatter_kws={"s": 1})

ax6.set_xlim([-100, max(data['sqft_basement'])]) # 6th plot has broken xscale

for i in range(1,7):

    a = eval('ax'+str(i))

    a.set_yscale('log')

plt.tight_layout()
corrmat = data.corr() # correlations between features

f, ax = plt.subplots(figsize=(16,16))

sns.heatmap(corrmat, square = True, cmap = 'RdBu_r', vmin = -1, vmax = 1, annot=True, fmt='.2f', ax = ax)
# selecting house with 33 bedrooms

myCase = data[data['bedrooms']==33]

myCase
# data without '33 bedrooms' house

theOthers = data[data['bedrooms']!=33]

theOtherStats = theOthers.describe()

theOtherStats
newDf = theOthers[['bedrooms', 'bathrooms', 'sqft_living']]

newDf = newDf[(newDf['bedrooms'] > 0) & (newDf['bathrooms'] > 0)]

newDf['bathrooms/bedrooms'] = newDf['bathrooms']/newDf['bedrooms']

newDf['sqft_living/bedrooms'] = newDf['sqft_living']/newDf['bedrooms']
newDf['bathrooms/bedrooms'].hist(bins=20)

plt.title('bathrooms/bedrooms ratio distribution')

plt.show()
newDf['sqft_living/bedrooms'].hist(bins=20)

plt.title('sqft_living/bedrooms ratio distribution')

plt.show()
# values for other properties

othersMeanBB = np.mean(newDf['bathrooms/bedrooms']) # mean bathroom/bedroom ratio

othersStdBB = np.std(newDf['bathrooms/bedrooms']) # std of bathroom/bedroom ratio



# values for suspicious house: myCase - real data; myCase2 - if there would be 3 bedrooms

myCaseBB = float(myCase['bathrooms'])/float(myCase['bedrooms'])

myCase2BB = float(myCase['bathrooms'])/3. # if there would be 3 bedrooms



print ('{:10}: {:6.3f} bathroom per bedroom'.format('"33" case', myCaseBB))

print ('{:10}: {:6.3f} bathroom per bedroom'.format('"3" case', myCase2BB))

print ('{:10}: {:6.3f} (std: {:.3f}) bathroom per bedroom'.format('The others', othersMeanBB, othersStdBB))
# values for other properties

othersMeanSB = np.mean(newDf['sqft_living/bedrooms']) # mean sqft_living/bedroom ratio

othersStdSB = np.std(newDf['sqft_living/bedrooms']) # std of sqft_living/bedroom ratio



# values for suspicious house: myCase - real data; myCase2 - if there would be 3 bedrooms

myCaseSB = float(myCase['sqft_living'])/float(myCase['bedrooms'])

myCase2SB = float(myCase['sqft_living'])/3. # if there would be 3 bedrooms



print ('{:10}: {:6.0f} sqft per bedroom'.format('"33" case', myCaseSB))

print ('{:10}: {:6.0f} sqft per bedroom'.format('"3" case', myCase2SB))

print ('{:10}: {:6.0f} (std: {:.0f}) sqft per bedroom'.format('The others', othersMeanSB, othersStdSB))
toDropIndex = myCase.index
data.drop(toDropIndex, inplace=True)
stats = data.describe()

stats
data2 = data[np.abs(data['price'] - stats['price']['mean']) <= (3*stats['price']['std'])] # cutting 'price'
data2.describe()
sns.regplot(x = "sqft_living", y = "price", data = data2, fit_reg=False, scatter_kws={"s": 2})
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize = (12, 15))

sns.stripplot(x = "grade", y = "price", data = data2, jitter=True, ax = ax1, color=sbr_c)

sns.stripplot(x = "view", y = "price", data = data2, jitter=True, ax = ax2, color=sbr_c)

sns.stripplot(x = "bedrooms", y = "price", data = data2, jitter=True, ax = ax3, color=sbr_c)

sns.stripplot(x = "bathrooms", y = "price", data = data2, jitter=True, ax = ax4, color=sbr_c)

sns.stripplot(x = "condition", y = "price", data = data2, jitter=True, ax = ax5, color=sbr_c)

sns.stripplot(x = "floors", y = "price", data = data2, jitter=True, ax = ax6, color=sbr_c)

ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)

for i in range(1,7):

    a = eval('ax'+str(i))

    a.set_yscale('log')

plt.tight_layout()
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize = (12, 12))

sns.regplot(x = 'sqft_living', y = 'price', data = data2, ax = ax1, fit_reg=False, scatter_kws={"s": 1})

sns.regplot(x = 'sqft_lot', y = 'price', data = data2, ax = ax2, fit_reg=False, scatter_kws={"s": 1})

sns.regplot(x = 'yr_built', y = 'price', data = data2, ax = ax5, fit_reg=False, scatter_kws={"s": 1})

sns.regplot(x = 'sqft_basement', y = 'price', data = data2, ax = ax6, fit_reg=False, scatter_kws={"s": 1})

sns.regplot(x = 'lat', y = 'price', data = data2, ax = ax3, fit_reg=False, scatter_kws={"s": 1})

sns.regplot(x = 'long', y = 'price', data = data2, ax = ax4, fit_reg=False, scatter_kws={"s": 1})

ax6.set_xlim([-100, max(data2['sqft_basement'])]) # 6th plot has broken xscale

for i in range(1,7):

    a = eval('ax'+str(i))

    a.set_yscale('log')

plt.tight_layout()
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize = (12, 6))

sns.regplot(x = 'sqft_basement', y = 'sqft_living', data = data2, ax = ax1, fit_reg=False, scatter_kws={"s": 1})

sns.regplot(x = 'sqft_above', y = 'sqft_living', data = data2, ax = ax2, fit_reg=False, scatter_kws={"s": 1})

plt.tight_layout()
data['basement'] = data['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)

data2['basement'] = data2['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)
data2.head(10)
# removing unnecessary features

dataRaw = data.copy(deep=True)

dataRaw.drop(['date', 'id'], axis = 1, inplace=True)

dataSel1 = data[['price', 'basement', 'bathrooms', 'bedrooms', 'grade', 'sqft_living', 'sqft_lot', 'waterfront', 'view']]

dataSel2 = data2[['price', 'basement', 'bathrooms', 'bedrooms', 'grade', 'sqft_living', 'sqft_lot', 'waterfront', 'view']]
# random_state=seed fixes RNG seed. 80% of data will be used for training, 20% for testing.

seed = 2

splitRatio = 0.2



# data with outliers, only columns selected manually

train, test = train_test_split(dataSel1, test_size=splitRatio, random_state=seed) 

Y_trn1 = train['price'].tolist()

X_trn1 = train.drop(['price'], axis=1)

Y_tst1 = test['price'].tolist()

X_tst1 = test.drop(['price'], axis=1)



# data without outliers, only columns selected manually

train2, test2 = train_test_split(dataSel2, test_size=splitRatio, random_state=seed)

Y_trn2 = train2['price'].tolist()

X_trn2 = train2.drop(['price'], axis=1)

Y_tst2 = test2['price'].tolist()

X_tst2 = test2.drop(['price'], axis=1)



# data with outliers and all meaningful columns (date and id excluded)

trainR, testR = train_test_split(dataRaw, test_size=splitRatio, random_state=seed)

Y_trnR = trainR['price'].tolist()

X_trnR = trainR.drop(['price'], axis=1)

Y_tstR = testR['price'].tolist()

X_tstR = testR.drop(['price'], axis=1)
X_trnR.head()
modelLRR = LinearRegression(n_jobs=njobs)

modelLR1 = LinearRegression(n_jobs=njobs)

modelLR2 = LinearRegression(n_jobs=njobs)
modelLRR.fit(X_trnR, Y_trnR)

modelLR1.fit(X_trn1, Y_trn1)

modelLR2.fit(X_trn2, Y_trn2)
scoreR = modelLRR.score(X_tstR, Y_tstR)

score1 = modelLR1.score(X_tst1, Y_tst1)

score2 = modelLR2.score(X_tst2, Y_tst2)



print ("R^2 score: {:8.4f} for {}".format(scoreR, 'Raw data'))

print ("R^2 score: {:8.4f} for {}".format(score1, 'Dataset 1 (with outliers)'))

print ("R^2 score: {:8.4f} for {}".format(score2, 'Dataset 2 (without outliers)'))
lrDict = {'Dataset': ['Raw data', 'Dataset 1', 'Dataset 2'], 

         'R^2 score': [scoreR, score1, score2],

         'Best params': [None, None, None]}

pd.DataFrame(lrDict)
lr = LinearRegression(n_jobs=njobs, normalize=True)

lr.fit(X_trnR, Y_trnR)
weights = eli5.explain_weights_df(lr) # weights of LinearRegression model for RawData

rank = [int(i[1:]) for i in weights['feature'].values[1:]]

labels = ['BIAS'] + [X_trnR.columns[i] for i in rank]

weights['feature'] = labels

weights
tuned_parameters = {'n_neighbors': range(1,21), 'weights': ['uniform', 'distance']}

knR = GridSearchCV(KNeighborsRegressor(), tuned_parameters, n_jobs=njobs)

kn1 = GridSearchCV(KNeighborsRegressor(), tuned_parameters, n_jobs=njobs)

kn2 = GridSearchCV(KNeighborsRegressor(), tuned_parameters, n_jobs=njobs)
knR.fit(X_trnR, Y_trnR)

kn1.fit(X_trn1, Y_trn1)

kn2.fit(X_trn2, Y_trn2)
scoreR = knR.score(X_tstR, Y_tstR)

score1 = kn1.score(X_tst1, Y_tst1)

score2 = kn2.score(X_tst2, Y_tst2)

parR = knR.best_params_

par1 = kn1.best_params_

par2 = kn2.best_params_



print ("R^2: {:6.4f} {:12} | Params: {}".format(scoreR, 'Raw data', parR))

print ("R^2: {:6.4f} {:12} | Params: {}".format(score1, 'Dataset 1', par1))

print ("R^2: {:6.4f} {:12} | Params: {}".format(score2, 'Dataset 2', par2))
knDict = {'Dataset': ['Raw data', 'Dataset 1', 'Dataset 2'], 

         'R^2 score': [scoreR, score1, score2],

         'Best params': [parR, par1, par2]}

pd.DataFrame(knDict)
tuned_parameters = {'n_estimators': [10,20,50,100], 'max_depth': [10,20,50]}

rfR = GridSearchCV(RandomForestRegressor(), tuned_parameters, n_jobs=njobs)

rf1 = GridSearchCV(RandomForestRegressor(), tuned_parameters, n_jobs=njobs)

rf2 = GridSearchCV(RandomForestRegressor(), tuned_parameters, n_jobs=njobs)
rfR.fit(X_trnR, Y_trnR)

rf1.fit(X_trn1, Y_trn1)

rf2.fit(X_trn2, Y_trn2)
scoreR = rfR.score(X_tstR, Y_tstR)

score1 = rf1.score(X_tst1, Y_tst1)

score2 = rf2.score(X_tst2, Y_tst2)

parR = rfR.best_params_

par1 = rf1.best_params_

par2 = rf2.best_params_



print ("R^2: {:6.4f} {:12} | Params: {}".format(scoreR, 'Raw data', parR))

print ("R^2: {:6.4f} {:12} | Params: {}".format(score1, 'Dataset 1', par1))

print ("R^2: {:6.4f} {:12} | Params: {}".format(score2, 'Dataset 2', par2))
rfDict = {'Dataset': ['Raw data', 'Dataset 1', 'Dataset 2'], 

         'R^2 score': [scoreR, score1, score2],

         'Best params': [parR, par1, par2]}

pd.DataFrame(rfDict)
rf = RandomForestRegressor(n_estimators=100, max_depth=50, n_jobs=njobs)
rf.fit(X_trnR, Y_trnR)
importances = rf.feature_importances_



# calculating std by collecting 'feature_importances_' from every tree in forest

rfStd = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

indices = np.argsort(importances)[::-1] # descending order
xlabels = [X_trnR.columns[i] for i in indices]
plt.title("Random Forest: Mean feature importances with STD")

plt.bar(range(len(xlabels)), importances[indices],

       color="#1156bf", yerr=rfStd[indices], align="center", capsize=8)

plt.xticks(rotation=45)

plt.xticks(range(len(xlabels)), xlabels)

plt.xlim([-1, len(xlabels)])

plt.show()
# feature importance for RandomForest with the best params tunnedy by GridSearchCV calculated by eli5

weights = eli5.explain_weights_df(rf) 

rank = [int(i[1:]) for i in weights['feature'].values]

labels = [X_trnR.columns[i] for i in rank]

weights['feature'] = labels

weights
resDict = {'lr' : lrDict, 'kn' : knDict, 'rf' : rfDict}
dict_of_df = {k: pd.DataFrame(v) for k,v in resDict.items()}

resDf = pd.concat(dict_of_df, axis=0)

resDf
toPlot = resDf.sort_values(by=['R^2 score'], ascending=False)

fig, axes = plt.subplots(ncols=1, figsize=(12, 8))

toPlot['R^2 score'].plot(ax=axes, kind='bar', title='R$^{2}$ score', color="#1153ff")

plt.ylabel('R$^{2}$', fontsize=20)

plt.xlabel('Model & Dataset', fontsize=20)

plt.xticks(rotation=45)

plt.show()
toPlot = resDf.sort_values(by=['R^2 score'], ascending=False)

fig, axes = plt.subplots(ncols=1, figsize=(12, 8))

toPlot.loc['lr']['R^2 score'].plot(ax=axes, kind='bar', title='R$^{2}$ score for Linear Regression', color="#1153ff")

plt.ylabel('R$^{2}$', fontsize=20)

plt.xlabel('Dataset', fontsize=20)

plt.xticks(rotation=45)

plt.xticks(range(3), [toPlot.loc['lr']['Dataset'][i] for i in range(3)])

plt.show()
toPlot = resDf.sort_values(by=['R^2 score'], ascending=False)

fig, axes = plt.subplots(ncols=1, figsize=(12, 8))

toPlot.loc['kn']['R^2 score'].plot(ax=axes, kind='bar', title='R$^{2}$ score for KNeighbors', color="#1153ff")

plt.ylabel('R$^{2}$', fontsize=20)

plt.xlabel('Dataset', fontsize=20)

plt.xticks(rotation=45)

plt.xticks(range(3), [toPlot.loc['kn']['Dataset'][i] for i in range(3)])

plt.show()
toPlot = resDf.sort_values(by=['R^2 score'], ascending=False)

fig, axes = plt.subplots(ncols=1, figsize=(12, 8))

toPlot.loc['rf']['R^2 score'].plot(ax=axes, kind='bar', title='R$^{2}$ score for Random Forest', color="#1153ff")

plt.ylabel('R$^{2}$', fontsize=20)

plt.xlabel('Dataset', fontsize=20)

plt.xticks(rotation=45)

plt.xticks(range(3), [toPlot.loc['rf']['Dataset'][i] for i in range(3)])

plt.show()