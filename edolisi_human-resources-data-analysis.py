import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

import pandas as pd

import seaborn as sbn

%matplotlib inline



HRdata = pd.read_csv('../input/HR_comma_sep.csv')

HRdata.describe()
mydims = (10, 5)

fig, ax = plt.subplots(figsize=mydims)

sbn.barplot(HRdata['sales'], HRdata['satisfaction_level'], ax=ax)
fig, ax = plt.subplots(figsize=mydims)

sbn.barplot(HRdata['salary'], HRdata['satisfaction_level'], ax=ax)
hrnum = HRdata

hrnum = hrnum.drop('sales', 1) # drop 'sales'

numsalary = {"low":1 ,"medium":2,"high":3} # turn 'salary' to numerical

hrnum['salary'] = hrnum['salary'].apply(numsalary.get).astype(float)



mydimssq = (6, 6)

fig, ax2 = plt.subplots(figsize=mydims)

sbn.heatmap(hrnum.corr(), vmax=.75, square=True, annot=True, fmt='.2f', ax=ax2)
hrnum = np.asarray(hrnum) # turn the pandas dataframe into a numpy array



X = hrnum.copy() # X is the normalized version of the dataset

for i in range(0,9):

    X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.sqrt(np.var(X[:,i]))



pca = PCA() # apply PCA

pca = pca.fit(X)

Xpca = pca.transform(X) # 'Xpca' is the PCA version of X; its columns should be independent

pd.DataFrame(Xpca[0:5], columns=['c1','c2','c3','c4','c5','c6','c7','c8','c9']) # show the first 5 rows of Xpca
PCAratio = np.empty([9,1])

PCAratio[0] = pca.explained_variance_ratio_[0]

for i in range(1,9): # compute the cumulative variance explained by components

    PCAratio[i] = PCAratio[i-1] + pca.explained_variance_ratio_[i]

pd.DataFrame(np.transpose(PCAratio), columns=['c1','c2','c3','c4','c5','c6','c7','c8','c9'])
numclusts = range(2, 9) # range of numbers of clusters k

km = [KMeans(n_clusters=k) for k in numclusts] # compute k-means for each k in the range

scores = [km[k].fit(X).score(X) for k in range(len(km))] # extract the scores



fig, ax = plt.subplots(figsize=mydims)

plt.plot(numclusts, scores)

plt.xlabel('number of k-means clusters')

plt.ylabel('score')
kmeans = KMeans(n_clusters=5, random_state=0).fit(X) # fit the k-means clustering with k=5



mydimssqbig = (9, 9)

fig, ax = plt.subplots(figsize=mydimssqbig) # plot the first two PCA components with colours corresponding to clusters

plt.scatter(Xpca[:,0], Xpca[:,1], c=kmeans.labels_, s=6.0, cmap='Set3')

plt.xlabel('PCA 1')

plt.ylabel('PCA 2')
infomat = np.empty([5,7]) # produce a summary table of the 5 clusters

for i in range(0,5):

    clust_i = HRdata[kmeans.labels_==i]

    infomat[i, 0] = clust_i.shape[0]

    infomat[i, 1] = np.mean(clust_i['satisfaction_level'])

    infomat[i, 2] = np.mean(clust_i['last_evaluation'])

    infomat[i, 3] = np.mean(clust_i['average_montly_hours'])

    infomat[i, 4] = np.mean(clust_i['left'])

    infomat[i, 5] = np.mean(clust_i['Work_accident'])

    infomat[i, 6] = np.mean(clust_i['promotion_last_5years'])

pd.DataFrame(infomat, columns=['Size','Satisfaction','Evaluation','Avg hours','Left','Accident','Promotion'])
mydimside3 = (14, 4)

fig, ax = plt.subplots(1,3, figsize=mydimside3) # histograms of the 3 continuous variables

ax[0].hist(HRdata['satisfaction_level'], bins=20,  rwidth=0.9)

ax[0].set_xlabel('Satisfaction')

ax[1].hist(HRdata['last_evaluation'], bins=20,  rwidth=0.9)

ax[1].set_xlabel('Last evaluation')

ax[2].hist(HRdata['average_montly_hours'], bins=20,  rwidth=0.9)

ax[2].set_xlabel('Avg. monthly hours')
mydimside = (12, 4)

fig, ax = plt.subplots(1,2, figsize=mydimside)

ax[0].plot(HRdata['last_evaluation'],HRdata['satisfaction_level'], '.', markersize=5.0)

ax[0].set_xlabel('Evaluation')

ax[0].set_ylabel('Satisfaction')

ax[1].plot(HRdata['average_montly_hours'],HRdata['satisfaction_level'], '.', markersize=5.0)

ax[1].set_xlabel('Monthly hours')

ax[1].set_ylabel('Satisfaction')
fig, ax = plt.subplots(1,2, figsize=mydimside)

sbn.barplot(HRdata['number_project'], HRdata['satisfaction_level'], ax=ax[0])

sbn.barplot(HRdata['time_spend_company'], HRdata['satisfaction_level'], ax=ax[1])
HRshuffle = hrnum[np.random.choice(14999,14999,replace=False),:] # randomly shuffle the dataset

HRtrain = HRshuffle[0:14000,:] # produce testing and training subsets

HRtest = HRshuffle[14000:14999,:]



y = HRtrain[:,0]

Xtrain = HRtrain[:,1:9]

Xtest = HRtest[:,1:9]

ytest = HRtest[:,0]



RF = RandomForestRegressor(n_estimators=100) # apply the random forest with 100 trees on the training subset

RF.fit(Xtrain, y)

ypred = RF.predict(Xtest) # predict the testing response



fig, ax = plt.subplots(figsize=mydimssqbig) # plot true satifaction vs predicted satisfaction

plt.plot(ytest, ypred, '.', markersize=7.0)

plt.xlabel('True satisfaction')

plt.ylabel('Predicted satisfaction')



MSE = mean_squared_error(ytest,ypred) # calculate and print MSE and correlation between true and predicted satisfaction

corr = np.corrcoef(ytest,ypred)[0,1]

print('MSE:',MSE,'; corr:',corr)
featurenames = np.array(['Evaluation','Projects','Monthly Hours','Time','Accident','Left','Promotion','Salary'])

pd.DataFrame([featurenames,RF.feature_importances_])