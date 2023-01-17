import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = 'metro-bike-share-trip-data.csv'
data = pd.read_csv("../input/metro-bike-share-trip-data.csv")
data.head()
start_time = data['Start Time']
ncolum=start_time.shape[0]
start_time_formatted = np.zeros(ncolum)    
for i in range(ncolum):
    temp1_time =  start_time[i].split('T')[1]
    temp2_time = temp1_time.split(":")
    temp_hour = float(temp2_time[0])
    temp_min = float(temp2_time[1])
    start_time_formatted[i] = temp_hour +temp_min/60.0

start_time_formatted
num_bins = 24
plt.figure(figsize=(10,10))
n, bins, patches = plt.hist(start_time_formatted, num_bins, facecolor='mediumblue', alpha=0.5)
plt.xlim(0,24)
plt.xticks(np.arange(0,24,2), fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel('Time', fontsize = 20)
plt.ylabel('# of trips', fontsize = 20)
plt.show()
duration = data['Duration']
planduration = data['Plan Duration']
plt.figure(figsize=(10,10))
plt.scatter(duration, planduration)
plt.xlabel('real duration', fontsize = 20)
plt.ylabel('plan duration', fontsize = 20)
plt.yticks(fontsize = 15)
plt.xticks(fontsize = 15)
duration.head(10)
planduration.head(10)
start_time = data['Start Time']
ncolum=start_time.shape[0]
start_time_formatted = np.zeros(ncolum)   
for i in range(ncolum):
    temp1_time =  start_time[i].split('T')[1]
    temp2_time = temp1_time.split(":")
    temp_hour = float(temp2_time[0])
    temp_min = float(temp2_time[1])
    start_time_formatted[i] = temp_hour +temp_min/60.0

start_time = start_time_formatted

data['Start Time'] = start_time_formatted

station_list = list(data.groupby(['Starting Station ID']).groups.keys())

data_ML=np.zeros((len(station_list)*24, 4))

for i in range(len(station_list)):
    data_temp = data[data['Starting Station ID']==station_list[i]]
    hist, bin_edges= np.histogram(data_temp['Start Time'], bins = range(25))
    hist.shape
    data_ML[24*i:(i+1)*24,3]=hist
    data_ML[24*i:(i+1)*24,2]=bin_edges[1:]
    data_ML[24*i:(i+1)*24,0]=np.array(data_temp['Starting Station Latitude'])[0]
    data_ML[24*i:(i+1)*24,1]=np.array(data_temp['Starting Station Longitude'])[0]
    
data_ML[0:24,0]=np.mean(data_ML[25,0])
data_ML[0:24,1]=np.mean(data_ML[25,1])
data_ML[data_ML[:,0]==0,0]=np.mean(data_ML[25,0])
data_ML[data_ML[:,1]==0,1]=np.mean(data_ML[25,1])
data_ML[0:24,0]=np.mean(data_ML[25,0])
data_ML[0:24,1]=np.mean(data_ML[25,1])
data_ML[data_ML[:,0]==0,0]=np.mean(data_ML[25,0])
data_ML[data_ML[:,1]==0,1]=np.mean(data_ML[25,1])

print(np.amin(data_ML[:,3]), np.amax(data_ML[:,3]),np.amin(data_ML[:,1]),np.amax(data_ML[:,1]))

#data_ML[:,0]=(data_ML[:,0]-np.amin(data_ML[:,0]))/(np.amax(data_ML[:,0])-np.amin(data_ML[:,0]))*1
#data_ML[:,1]=(data_ML[:,1]-np.amin(data_ML[:,1]))/(np.amax(data_ML[:,1])-np.amin(data_ML[:,1]))*1
#data_ML[:,2]=(data_ML[:,2]-np.amin(data_ML[:,2]))/(np.amax(data_ML[:,2])-np.amin(data_ML[:,2]))*1
#data_ML[:,3]=(data_ML[:,3]-np.amin(data_ML[:,3]))/(np.amax(data_ML[:,3])-np.amin(data_ML[:,3]))*10
#data_ML_temp = data_ML[data_ML[:,1]>0.8]
data_ML
from sklearn import tree
features = data_ML[:,:-1]
labels = data_ML[:,-1]
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.50)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#model selection
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#evaluation metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification



models=[RandomForestRegressor(),AdaBoostRegressor(),BaggingRegressor(),SVR(kernel='rbf'),KNeighborsRegressor()]
model_names=['RandomForestRegressor','AdaBoostRegressor','BaggingRegressor','SVR','KNeighborsRegressor']
rmsle=[]
accuracy = []
d={}
for model in range (len(models)):
    clf=models[model]
    clf.fit(features_train,labels_train)
    test_pred=clf.predict(features_test)
    rmsle.append(np.sqrt(mean_squared_log_error(test_pred,labels_test)))
d={'Modelling Algo':model_names,'RMSLE':rmsle}   
d
