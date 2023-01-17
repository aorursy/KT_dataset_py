# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/baldwin_pump_data.csv')
data.columns
#Know about the missing data
data.isnull().sum()
import regex as re
import datetime
date=[]
for i in data['Unnamed: 0']:
    if re.search('/',i):
        date.append(datetime.datetime.strptime(i.replace('/','-'),'%m-%d-%y %H:%M'))
    else:
        date.append(datetime.datetime.strptime(i,'%m-%d-%y %H:%M'))
df=pd.DataFrame(date,columns=['Time'])    
New_df=pd.concat([df,data.iloc[:,1:]],axis=1)
New_df.describe()
#as datetime features can increase useful features. so date, month, year, time is segregated to create new features.
dff = pd.DataFrame({"month": New_df['Time'].dt.month,
              "day": New_df['Time'].dt.day,
              "hour": New_df['Time'].dt.hour,
              "minute": New_df['Time'].dt.minute
             })
complete_data = pd.concat([New_df.iloc[:,:], dff],axis=1)
New_df_dt=New_df.iloc[:,0]
New_df_dt.describe()
complete_data=complete_data.set_index("Time") 
complete_data
data['BFPT_B_LP_SPEED_INPUT__1__'].describe()
#The time index will be used in plot Y denote time space
Y=complete_data.index
Y
#We have seen that data has high misiingness in a few features.
#We have few Machine Learning based imputation algorithm (KNN, EM) and a few statistical method (iterative imputer and iterativeSVD). 
#In past Machine learning method has good imputation result. Here, using KNN for imputation

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
complete_data_impu = imputer.fit_transform(complete_data)
from sklearn import preprocessing
mm_scaler = preprocessing.MinMaxScaler()
complete_data_impu = mm_scaler.fit_transform(complete_data_impu)
complete_data_impu
complete_data_impu_df=pd.DataFrame(complete_data_impu)
complete_data_impu_df.columns=complete_data.columns
complete_data_impu_df.describe()

complete_data_impu_df["BFPT_B_LP_SPEED_INPUT__1__"].describe()
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(complete_data_impu_df["BFPT_B_LP_SPEED_INPUT__1__"], label='speed_input1', color='blue', animated = True, linewidth=1)
ax.plot(complete_data_impu_df["BFPT_B_LP_SPEED_INPUT__2__"], label='speed_inpu2', color='red', animated = True, linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Motor speed curve', fontsize=16)
plt.show()

complete_data_impu_df['BFPT_B_LP_SPEED_INPUT__1reformed'] = 0
threshold=0.2
complete_data_impu_df.loc[complete_data_impu_df["BFPT_B_LP_SPEED_INPUT__1__"] > threshold, "BFPT_B_LP_SPEED_INPUT__1reformed"] = 1
complete_data_impu_df=complete_data_impu_df.drop(["BFPT_B_LP_SPEED_INPUT__1__"], axis=1)
complete_data_impu_df

import seaborn as sns

sns.distplot( complete_data_impu_df["BFP_SEAL_WATER_FLOW"] , color="skyblue", label="Sepal Length")
sns.distplot( complete_data_impu_df["BFP_SUCT_HDR_PRESS________"] , color="red", label="Sepal Width")
sns.distplot( complete_data_impu_df["BFPT_B_LP_SPEED_INPUT__1reformed"], color="black" , label="Sepal Length")




#histogram plot for represantation of all features distribution so that we can know about anomaly visually if there is kink in regular value.
#show plot

complete_data_impu_df.iloc[:,:46].hist(bins=50,figsize=(30,30))
plt.show()
complete_data_impu_df
#heat map to show coorelation amongst the variable
sns.heatmap(complete_data_impu_df.iloc[:,:46].corr(),fmt='.2g')
from numpy.random import seed
import tensorflow as tf


from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from sklearn.model_selection import train_test_split
#spliting the data for training and testing
X_train, X_test,ytrain,ytest= train_test_split(complete_data_impu,Y,test_size=0.33, random_state=42)
# reshape inputs for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print("Training data shape:", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print("Test data shape:", X_test.shape)
print("Test data shape:", ytest.shape)
# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model
 #create the autoencoder model
model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
model.summary()
nb_epochs = 200
batch_size = 30
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.05).history

# plot the training losses
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()
# plot the loss distribution of the training set
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=complete_data.columns)
X_pred.index = ytrain

scored = pd.DataFrame(index=ytrain)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
plt.xlim([0.0,.5])
# calculate the loss on the test set
X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=complete_data.columns)
X_pred.index = ytest

scored = pd.DataFrame(index=ytest)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
scored['Threshold'] = 0.15
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
scored.head()
# calculate the same metrics for the training set 
# and merge all data in a single dataframe for plotting
X_pred_train = model.predict(X_train)
X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
X_pred_train = pd.DataFrame(X_pred_train, columns=complete_data.columns)
X_pred_train.index = ytrain

scored_train = pd.DataFrame(index=ytrain)
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-Xtrain), axis = 1)
scored_train['Threshold'] = 0.15
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
scored = pd.concat([scored_train, scored])
# plot bearing failure time plot
scored.plot(logy=True,  figsize=(16,9), ylim=[1e-2,1e2], color=['blue','red'])
# save all model information, including weights, in h5 format
model.save("Anomaly_model.h5")
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
n_cluster = range(1, 20)
kmeans = [KMeans(n_clusters=i).fit(complete_data_impu_df) for i in n_cluster]
scores = [kmeans[i].score(complete_data_impu_df) for i in range(len(kmeans))]

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(n_cluster, scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show();
X = complete_data_impu_df.reset_index(drop=True)
km = KMeans(n_clusters=10)
km.fit(X)
km.predict(X)
labels = km.labels_
#Plotting
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X.iloc[:,0], X.iloc[:,1], X.iloc[:,2],
          c=labels.astype(np.float), edgecolor="k")
ax.set_xlabel("BFP_SEAL_WATER_FLOW")
ax.set_ylabel("BFP_SUCT_HDR_PRESS________")
ax.set_zlabel("AUX_CDSR_2B_PRESS_________")
plt.title("K Means", fontsize=14);
mean_vec = np.mean(complete_data_impu_df.iloc[:,:], axis=0)
cov_mat = np.cov(complete_data_impu_df.iloc[:,1:].T)
eig_val, eig_vec = np.linalg.eig(cov_mat)
eig_pair = [ (np.abs(eig_val[i]),eig_vec[:,i]) for i in range(len(eig_val))]
eig_pair.sort(key = lambda x: x[0], reverse= True)
tot = sum(eig_val)
var_exp = [(i/tot)*100 for i in sorted(eig_val, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

plt.figure(figsize=(20, 10))
plt.bar(range(len(var_exp)), var_exp, alpha=0.3, align='center', label='individual explained variance', color = 'r')
plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show();
def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.at[i]= np.linalg.norm(Xa-Xb)
    return distance

outliers_fraction = 0.01
# get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly
distance = getDistanceByPoint(complete_data_impu_df, kmeans[8])
number_of_outliers = int(outliers_fraction*len(distance))
threshold = distance.nlargest(number_of_outliers).min()
# anomaly1 contain the anomaly result of the above method Cluster (0:normal, 1:anomaly) 
df['anomaly1'] = (distance >= threshold).astype(int)

# visualisation of anomaly with cluster view
fig, ax = plt.subplots(figsize=(10,6))
colors = {0:'blue', 1:'red'}
ax.scatter(complete_data_impu_df['BFP_SEAL_WATER_FLOW'], complete_data_impu_df['BFP_SUCT_HDR_PRESS________'], c=df["anomaly1"].apply(lambda x: colors[x]))
plt.xlabel('BFP_SEAL_WATER_FLOW')
plt.ylabel('BFP_SUCT_HDR_PRESS________')
plt.show();