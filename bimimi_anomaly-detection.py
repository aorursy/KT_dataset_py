import pandas as pd 



df = pd.read_excel('../input/superstore/Superstore.xls')

df.head()
import numpy as np 

import matplotlib.pyplot as plt 

plt.style.use(['fivethirtyeight', 'dark_background'])

import seaborn as sns 
df.Sales.describe()
sns.distplot(df['Sales'])

plt.title('SLS dist')

sns.despine()
print('Kurtosis: %f' % df['Sales'].kurt())

print('Skewness: %f' % df['Sales'].skew())
sns.distplot(df['Profit'])

plt.title('PFT dist')

sns.despine()
from sklearn.ensemble import IsolationForest



# algo for detecting anomalies 
# Specify model 



# Train on sales     

ifo = IsolationForest(n_estimators = 100)



ifo.fit(df['Sales'].values.reshape(-1, 1))



# Store in NumPy array 

min_max = np.linspace(df['Sales'].min(),

                      df['Sales'].max(), 

                      len(df)).reshape(-1, 1)



# Compute anomaly score for each observation 

    # input score is computed as the mean score of all trees 

anomaly_score = ifo.decision_function(min_max)



# Classify each observation as outlier or not

outliers = ifo.predict(min_max)
plt.plot(min_max, anomaly_score, label = 'anomaly_score')



plt.fill_between(min_max.T[0], 

                 np.min(anomaly_score), 

                 np.max(anomaly_score),

                 where = outliers == -1, 

                 color = 'm',

                 alpha = .5, 

                 label = 'outlier area')



plt.legend()

plt.xlabel('SLS')

plt.ylabel('anomaly score')

plt.show();
df.loc[df['Sales'] > 1000].head()
# Check index 10 

df.iloc[10]
ifo = IsolationForest(n_estimators = 100)



ifo.fit(df['Profit'].values.reshape(-1, 1))



min_max = np.linspace(df['Profit'].min(),

                      df['Profit'].max(),

                      len(df)).reshape(-1, 1)



anomaly_score = ifo.decision_function(min_max)



outliers = ifo.predict(min_max)
plt.plot(min_max, anomaly_score, label = 'anomaly_score')

plt.fill_between(min_max.T[0], 

                 np.min(anomaly_score),

                 np.max(anomaly_score),

                 where = outliers == -1,

                 color = 'r',

                 alpha = 0.5, 

                 label = 'outlier area')



plt.legend()

plt.xlabel('PFT')

plt.ylabel('anomaly score')

plt.show();
df.loc[df['Profit'] > 100].head()
df.iloc[54]
df.loc[df['Profit'] < -100].head()
df.iloc[27]
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



!pip install pyod

from pyod.models.abod import ABOD

from pyod.models.cblof import CBLOF



from pyod.models.feature_bagging import FeatureBagging

from pyod.models.hbos import HBOS

from pyod.models.iforest import IForest

from pyod.models.knn import KNN

from pyod.models.lof import LOF



from scipy import stats
from numpy import percentile

import matplotlib
df = df.rename(columns = {'Sales': 'SLS', 'Profit': 'PFT'})

df.head()
cols = ['SLS', 'PFT']

df[cols].head()
# Scale features,

    # lest create error Contour levels must be increasing

    

minmax = MinMaxScaler(feature_range = (0, 1))

df[['SLS','PFT']] = minmax.fit_transform(df[['SLS','PFT']])

df[['SLS','PFT']].head()
# Plot corr

sns.regplot(x = 'SLS', 

            y = 'PFT', data = df,

            color = 'y')



sns.despine();
X1 = df['SLS'].values.reshape(-1,1)

X2 = df['PFT'].values.reshape(-1,1)



X = np.concatenate((X1,X2),axis=1)
outliers_fraction = 0.01

xx , yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

clf = KNN(contamination=outliers_fraction)

clf.fit(X)

scores_pred = clf.decision_function(X) * -1

y_pred = clf.predict(X)

n_inliers = len(y_pred) - np.count_nonzero(y_pred)

n_outliers = np.count_nonzero(y_pred == 1)

plt.figure(figsize=(8, 8))



# Prediction dataframe

df1 = df

df1['outlier'] = y_pred.tolist()

    

# sales - inlier feature 1,  profit - inlier feature 2

inliers_sales = np.array(df1['SLS'][df1['outlier'] == 0]).reshape(-1,1)

inliers_profit = np.array(df1['PFT'][df1['outlier'] == 0]).reshape(-1,1)

    

# sales - outlier feature 1, profit - outlier feature 2

outliers_sales = df1['SLS'][df1['outlier'] == 1].values.reshape(-1,1)

outliers_profit = df1['PFT'][df1['outlier'] == 1].values.reshape(-1,1)

         

print('OUTLIERS: ',n_outliers,'INLIERS: ',n_inliers)



threshold = percentile(scores_pred, 100 * outliers_fraction)



Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1

Z = Z.reshape(xx.shape)



plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)

a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')

plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')

b = plt.scatter(inliers_sales, inliers_profit, c='white',s=20, edgecolor='k')    

c = plt.scatter(outliers_sales, outliers_profit, c='black',s=20, edgecolor='k')  



plt.axis('tight')  

plt.legend([a.collections[0], b,c], ['learned decision function', 'inliers','outliers'],

           prop=matplotlib.font_manager.FontProperties(size=20),loc='lower right')      

plt.xlim((0, 1))

plt.ylim((0, 1))

plt.title('K Nearest Neighbors (KNN)')

plt.show();
outliers_fraction = 0.01

xx , yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

clf = IForest(contamination=outliers_fraction,random_state=0)

clf.fit(X)

scores_pred = clf.decision_function(X) * -1



y_pred = clf.predict(X)

n_inliers = len(y_pred) - np.count_nonzero(y_pred)

n_outliers = np.count_nonzero(y_pred == 1)

plt.figure(figsize=(8, 8))



df1 = df

df1['outlier'] = y_pred.tolist()

    

# sales - inlier feature 1,  profit - inlier feature 2

inliers_sales = np.array(df1['SLS'][df1['outlier'] == 0]).reshape(-1,1)

inliers_profit = np.array(df1['PFT'][df1['outlier'] == 0]).reshape(-1,1)

    

# sales - outlier feature 1, profit - outlier feature 2

outliers_sales = df1['SLS'][df1['outlier'] == 1].values.reshape(-1,1)

outliers_profit = df1['PFT'][df1['outlier'] == 1].values.reshape(-1,1)

         

print('OUTLIERS: ',n_outliers,'INLIERS: ',n_inliers)



threshold = percentile(scores_pred, 100 * outliers_fraction)

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1

Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)

a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')

plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')

b = plt.scatter(inliers_sales, inliers_profit, c='white',s=20, edgecolor='k')

    

c = plt.scatter(outliers_sales, outliers_profit, c='black',s=20, edgecolor='k')

       

plt.axis('tight')

plt.legend([a.collections[0], b,c], ['learned decision function', 'inliers','outliers'],

           prop=matplotlib.font_manager.FontProperties(size=20),loc='lower right')      

plt.xlim((0, 1))

plt.ylim((0, 1))

plt.title('Isolation Forest')

plt.show();
outliers_fraction = 0.01



xx , yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))



clf = CBLOF(contamination = outliers_fraction,

            check_estimator = False, random_state = 0)

clf.fit(X)



# predict raw anomaly score

scores_pred = clf.decision_function(X) * -1

        

# Identify point as outlier or inlier

y_pred = clf.predict(X)

n_inliers = len(y_pred) - np.count_nonzero(y_pred)

n_outliers = np.count_nonzero(y_pred == 1)
# Features 

df1 = df

df1['outlier'] = y_pred.tolist() # store predictions in df

    

# 

inliers_sales = np.array(df1['SLS'][df1['outlier'] == 0]).reshape(-1,1)

inliers_profit = np.array(df1['PFT'][df1['outlier'] == 0]).reshape(-1,1)

    

# sales - outlier feature 1, profit - outlier feature 2

outliers_sales = df1['SLS'][df1['outlier'] == 1].values.reshape(-1,1)

outliers_profit = df1['PFT'][df1['outlier'] == 1].values.reshape(-1,1)

         

print('OUTLIERS:',n_outliers,

      'INLIERS:',n_inliers)
plt.figure(figsize=(8, 8))



# Threshold val for identifying outs and ins 

threshold = percentile(scores_pred, 100 * outliers_fraction)

        

# decision function calculates the raw anomaly score for every point

Z = clf.decision_function(np.c_[xx.ravel(), 

                                yy.ravel()]) * -1

Z = Z.reshape(xx.shape)

# fill blue map colormap from minimum anomaly score to threshold value

plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 

                                           threshold, 7),

             cmap = plt.cm.Blues_r)



# red contour line where anomaly score =thresold

a = plt.contour(xx, yy, Z, 

                levels=[threshold],

                linewidths = 2, colors = 'red')

        

# orange contour lines where range of anomaly score covers threshold val to max anomaly score

plt.contourf(xx, yy, Z, 

             levels = [threshold, Z.max()],

             colors = 'orange')



b = plt.scatter(inliers_sales, inliers_profit, 

                c = 'white',s = 20, edgecolor = 'k')

    

c = plt.scatter(outliers_sales, outliers_profit, 

                c = 'black',s=20, edgecolor = 'k')

       

plt.axis('tight')   

plt.legend([a.collections[0], b,c],

           ['learned decision function', 'inliers','outliers'],

           prop = matplotlib.font_manager.FontProperties(size = 20),

           loc = 'lower right')

      

plt.xlim((0, 1))

plt.ylim((0, 1))

plt.title('Cluster-based Local Outlier Factor (CBLOF)')

plt.show();
outliers_fraction = 0.01

xx , yy = np.meshgrid(np.linspace(0, 1, 100), 

                      np.linspace(0, 1, 100))

clf = HBOS(contamination = outliers_fraction)

clf.fit(X)



scores_pred = clf.decision_function(X) * -1

y_pred = clf.predict(X)



n_inliers = len(y_pred) - np.count_nonzero(y_pred)

n_outliers = np.count_nonzero(y_pred == 1)



plt.figure(figsize = (8, 8))



df1 = df

df1['outlier'] = y_pred.tolist()

    

inliers_sales = np.array(df1['SLS'][df1['outlier'] == 0]).reshape(-1,1)

inliers_profit = np.array(df1['PFT'][df1['outlier'] == 0]).reshape(-1,1)

    



outliers_sales = df1['SLS'][df1['outlier'] == 1].values.reshape(-1,1)

outliers_profit = df1['PFT'][df1['outlier'] == 1].values.reshape(-1,1)

         

print('OUTLIERS:',n_outliers,'INLIERS:',n_inliers)

threshold = percentile(scores_pred, 100 * outliers_fraction)



Z = clf.decision_function(np.c_[xx.ravel(), 

                                yy.ravel()]) * -1

Z = Z.reshape(xx.shape)



plt.contourf(xx, yy, Z, 

             levels = np.linspace(Z.min(), 

                                           threshold, 7),

             cmap = plt.cm.Blues_r)



a = plt.contour(xx, yy, Z, levels = [threshold],

                linewidths=2, colors='red')



plt.contourf(xx, yy, Z, 

             levels = [threshold, Z.max()],

             colors = 'orange')



b = plt.scatter(inliers_sales, inliers_profit, 

                c ='white',s=20, edgecolor='k')

    

c = plt.scatter(outliers_sales, outliers_profit, 

                c ='black',s=20, edgecolor='k')

       

plt.axis('tight')      

plt.legend([a.collections[0], b,c], 

           ['learned decision function', 'inliers','outliers'],

           prop = matplotlib.font_manager.FontProperties(size = 20),

           loc='lower right')      



plt.xlim((0, 1))

plt.ylim((0, 1))

plt.title('Histogram-base Outlier Detection (HBOS)')

plt.show();
# Predicted outliers 

df1.loc[df1['outlier'] == 1].tail()
# Reload to view features at original scale 

df = pd.read_excel('../input/superstore/Superstore.xls')
df.iloc[9649]
df.iloc[9741]
df.iloc[9774]
df.iloc[9270]
df.iloc[9857]