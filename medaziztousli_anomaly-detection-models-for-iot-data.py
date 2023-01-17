import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from matplotlib.pyplot import axes, title

import numpy as np
# Load the data set

df0 = pd.read_csv("../input/data_tunis.csv")



#Visualize the DataFrame df0

df0
# Fill some null values

df0 = df0.bfill(axis=0)



# Remove null acquisations

df0 = df0.dropna()



# Replace min_temp and max_temp with avg_temp

df0['temp'] = (df0['Tmin'] + df0['Tmax']) / 2

df0 = df0.drop('Tmin',axis=1)

df0 = df0.drop('Tmax',axis=1)



df0
# Create empty dataframes for each month

df_jan = pd.DataFrame(columns=df0.columns)

df_feb = pd.DataFrame(columns=df0.columns)

df_mar = pd.DataFrame(columns=df0.columns)

df_apr = pd.DataFrame(columns=df0.columns)

df_may = pd.DataFrame(columns=df0.columns)

df_jun = pd.DataFrame(columns=df0.columns)

df_jul = pd.DataFrame(columns=df0.columns)

df_aug = pd.DataFrame(columns=df0.columns)

df_sep = pd.DataFrame(columns=df0.columns)

df_oct = pd.DataFrame(columns=df0.columns)

df_nov = pd.DataFrame(columns=df0.columns)

df_dec = pd.DataFrame(columns=df0.columns)



# Fill the dataframes of each month

for i in range(len(df0)):

    if df0.loc[i]['date'][3:5] == '01':

        df_jan = df_jan.append(df0.loc[i])

    if df0.loc[i]['date'][3:5] == '02':

        df_feb = df_feb.append(df0.loc[i])

    if df0.loc[i]['date'][3:5] == '03':

        df_mar = df_mar.append(df0.loc[i])

    if df0.loc[i]['date'][3:5] == '04':

        df_apr = df_apr.append(df0.loc[i])

    if df0.loc[i]['date'][3:5] == '05':

        df_may = df_may.append(df0.loc[i])

    if df0.loc[i]['date'][3:5] == '06':

        df_jun = df_jun.append(df0.loc[i])

    if df0.loc[i]['date'][3:5] == '07':

        df_jul = df_jul.append(df0.loc[i])

    if df0.loc[i]['date'][3:5] == '08':

        df_aug = df_aug.append(df0.loc[i])

    if df0.loc[i]['date'][3:5] == '09':

        df_sep = df_sep.append(df0.loc[i])

    if df0.loc[i]['date'][3:5] == '10':

        df_oct = df_oct.append(df0.loc[i])

    if df0.loc[i]['date'][3:5] == '11':

        df_nov = df_nov.append(df0.loc[i])

    if df0.loc[i]['date'][3:5] == '12':

        df_dec = df_dec.append(df0.loc[i])
# Choose a month to work with; for example: January

df_curr = df_jan



# Remove the columns 'date' and 'index'

df_curr = df_curr.reset_index().drop(['date','index'],axis=1)



# Get xdata, ydata and zdata to make 3D plot later on

xdata = df_curr['vent']

ydata = df_curr['pluie']

zdata = df_curr['temp']
# Create empty dataframe for reference points

df_anomaly_reference = pd.DataFrame(columns=df_curr.columns)



# Define thresholds that will present anomalies; for example: January will present these values

MAX_PLUIE = 20

MAX_VENT = 60

MAX_TEMP = 30

MIN_TEMP =  6



# # Fill the dataframe of reference points

for i in range(len(df_curr)):

    if df_curr.loc[i, 'pluie'] > MAX_PLUIE or df_curr.loc[i, 'vent'] > MAX_VENT or df_curr.loc[i, 'temp'] > MAX_TEMP or df_curr.loc[i, 'temp'] < MIN_TEMP:

        df_anomaly_reference = df_anomaly_reference.append(df_curr.loc[i])



print("The number of anomalies reference is: " + str(len(df_anomaly_reference)))
# 1st algorithm: DBSCAN

from sklearn.cluster import DBSCAN

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 

outlier_detection = DBSCAN(eps = .2, metric="euclidean", min_samples = 5, n_jobs = -1)

num2 = scaler.fit_transform(df_curr)

num2 = pd.DataFrame(num2, columns = df_curr.columns)

clusters = outlier_detection.fit_predict(num2)



# 2D Plot: pluie = f(vent)

from matplotlib import cm

cmap = cm.get_cmap('Set1')

df_curr.plot.scatter(x='vent',y='pluie', c=clusters, cmap=cmap, colorbar = False)



# 2D Plot: pluie = f(temp)

from matplotlib import cm

cmap = cm.get_cmap('Set1')

df_curr.plot.scatter(x='temp',y='pluie', c=clusters, cmap=cmap, colorbar = False)



# PS: This will be the only time that we show 2D plots
# 3D Plot: pluie = f(temp, vent)

ax = axes(projection='3d')

ax.scatter3D(xdata, ydata, zdata, c=clusters, cmap=cmap)

ax.set_xlabel('vent',fontsize=20)

ax.set_ylabel('pluie',fontsize=20)

ax.set_zlabel('temp',fontsize=20)

title('DBSCAN — Anomalies')



true_predicted = 0

mistakes = 0



# Get True Negative and False Positive 

for i in range(len(clusters)):

    if clusters[i] == -1 and i in df_anomaly_reference.index:

        true_predicted += 1

    if clusters[i] == -1 and i not in df_anomaly_reference.index:

        mistakes += 1



print("DBSCAN  | True Negative = " + str(true_predicted))

print("DBSCAN  | False Positive = " + str(mistakes))
# 2nd algorithm: Isolation Forests

from sklearn.ensemble import IsolationForest

rs=np.random.RandomState(0)

clf = IsolationForest(max_samples=100,random_state=rs, contamination=.01) 

clf.fit(df_curr)

if_scores = clf.decision_function(df_curr)

if_anomalies=clf.predict(df_curr)

if_anomalies=pd.Series(if_anomalies).replace([-1,1],[1,0])

if_anomalies=df_curr[if_anomalies==1]



# 3D Plot

ax = plt.axes(projection='3d')

ax.scatter3D(xdata, ydata, zdata, c='white',s=20,edgecolor='k')

ax.scatter3D(if_anomalies['vent'], if_anomalies['pluie'], if_anomalies['temp'], c='red')

ax.set_xlabel('vent',fontsize=20)

ax.set_ylabel('pluie',fontsize=20)

ax.set_zlabel('temp',fontsize=20)

plt.title('Isolation Forests — Anomalies')



true_predicted = 0

mistakes = 0



# Get True Negative and False Positive 

for i in range(len(clusters)):

    if i in if_anomalies.index and i in df_anomaly_reference.index:

        true_predicted += 1

    if i in if_anomalies.index and i not in df_anomaly_reference.index:

        mistakes += 1



print("Isolation Forests | True Negative = " + str(true_predicted))

print("Isolation Forests | False Positive = " + str(mistakes))
# 3rd algorithm: Local Outlier Factor

from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(n_neighbors=30, contamination=.01)

y_pred = clf.fit_predict(df_curr)

LOF_Scores = clf.negative_outlier_factor_

LOF_pred=pd.Series(y_pred).replace([-1,1],[1,0])

LOF_anomalies=df_curr[LOF_pred==1]



# 3D Plot

ax = plt.axes(projection='3d')

ax.scatter3D(xdata, ydata, zdata, c='white',s=20,edgecolor='k')

ax.scatter3D(LOF_anomalies['vent'], LOF_anomalies['pluie'], LOF_anomalies['temp'], c='red')

ax.set_xlabel('vent',fontsize=20)

ax.set_ylabel('pluie',fontsize=20)

ax.set_zlabel('temp',fontsize=20)

plt.title('Local Outlier Factor — Anomalies')



true_predicted = 0

mistakes = 0



# Get True Negative and False Positive 

for i in range(len(clusters)):

    if i in LOF_anomalies.index and i in df_anomaly_reference.index:

        true_predicted += 1

    if i in LOF_anomalies.index and i not in df_anomaly_reference.index:

        mistakes += 1



print("Local Outlier Factor | True Negative = " + str(true_predicted))

print("Local Outlier Factor | False Positive = " + str(mistakes))
# 4th algorithm: Elliptic Envelope

from sklearn.covariance import EllipticEnvelope

clf = EllipticEnvelope(contamination=0.005,random_state=0)

clf.fit(df_curr)

ee_scores = pd.Series(clf.decision_function(df_curr)) 

ee_pred = clf.predict(df_curr)

ee_anomalies = df_curr[ee_pred==-1]



# 3D Plot

ax = plt.axes(projection='3d')

ax.scatter3D(xdata, ydata, zdata, c='white',s=20,edgecolor='k')

ax.scatter3D(ee_anomalies['vent'], ee_anomalies['pluie'], ee_anomalies['temp'], c='red')

ax.set_xlabel('vent',fontsize=20)

ax.set_ylabel('pluie',fontsize=20)

ax.set_zlabel('temp',fontsize=20)

plt.title('Elliptic Envelope — Anomalies')



true_predicted = 0

mistakes = 0



# Get True Negative and False Positive 

for i in range(len(clusters)):

    if i in ee_anomalies.index and i in df_anomaly_reference.index:

        true_predicted += 1

    if i in ee_anomalies.index and i not in df_anomaly_reference.index:

        mistakes += 1



print("Elliptic Envelope | True Negative = " + str(true_predicted))

print("Elliptic Envelope | False Positive = " + str(mistakes))
# 5th algorithm: One-Class Support Vector Machines

from sklearn import svm

clf=svm.OneClassSVM(nu=.02,kernel='rbf',gamma=.005)

clf.fit(df_curr)

y_pred=clf.predict(df_curr)



# 3D Plot

ax = plt.axes(projection='3d')

ax.scatter3D(xdata, ydata, zdata, c=y_pred, cmap=cmap)

ax.set_xlabel('vent',fontsize=20)

ax.set_ylabel('pluie',fontsize=20)

ax.set_zlabel('temp',fontsize=20)

plt.title('One-Class Support Vector Machines — Anomalies')



true_predicted = 0

mistakes = 0



# Get True Negative and False Positive 

for i in range(len(clusters)):

    if y_pred[i] == -1 and i in df_anomaly_reference.index:

        true_predicted += 1

    if y_pred[i] == -1 and i not in df_anomaly_reference.index:

        mistakes += 1



print("One-Class Support Vector Machines | True Negative = " + str(true_predicted))

print("One-Class Support Vector Machines | False Positive = " + str(mistakes))