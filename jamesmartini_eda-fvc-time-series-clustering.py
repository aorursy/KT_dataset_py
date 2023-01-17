!pip install '/kaggle/input/tslearn-wheel37/tslearn-0.4.1-cp37-cp37m-manylinux1_x86_64.whl'
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')
train.head()
train.set_index('Weeks',inplace=True)

train = train[['Patient','FVC']]

train
from tslearn.clustering import TimeSeriesKMeans

from tslearn.utils import to_time_series_dataset
FVC_series = []

for patient in train['Patient'].unique():

#     FVC_series.append(train[train['Patient'] == patient][['FVC', 'Percent', 'Age', 'Ex-smoker', 'Never smoked', 'Male']])

    FVC_series.append(train[train['Patient'] == patient][['FVC']])

formatted_data = to_time_series_dataset(FVC_series)
model = TimeSeriesKMeans(n_clusters=4, metric='dtw', max_iter=10,random_state=1)

model.fit(formatted_data)
sns.countplot(model.labels_)

plt.xlabel('Cluster')

plt.ylabel('Number of time series')

plt.title('Number of time series versus cluster')
xvec=[]

yvec=[]

color_label = []

colors = []

locs = [0]

count = 0



#Collect the time series based on label

for label in pd.Series(model.labels_).unique(): #For each cluster label

    for labeled_patient_number in np.where(model.labels_ == label)[0]: #Grab the patients corresponding to that particular label

        x,y = FVC_series[labeled_patient_number].index, FVC_series[labeled_patient_number]['FVC'] #Save the FVC measurements

        xvec.append(x)

        yvec.append(y)

        color_label.append(label) #Assign a colour based on label

        count+=1

    locs.append(count) #Plot legend location for first occurence of cluster

    

#Assign colors for the four different clusters

for j in range(len(color_label)):

    if color_label[j] == 0:

        colors.append('tab:blue')

    if color_label[j] == 1:

        colors.append('tab:orange')

    if color_label[j] == 2:

        colors.append('tab:green')

    if color_label[j] == 3:

        colors.append('tab:red')

        

fig = plt.figure(figsize=(11,7)); ax = fig.add_subplot(1, 1, 1)





for j in range(len(xvec)):

    ax.plot(xvec[j],yvec[j],color=colors[j],label="Time Series Cluster {0}".format(color_label[j]) if j in locs else "")



# legend_handles, legend_labels = ax.get_legend_handles_labels()

# ax.legend(legend_handles, legend_labels)

ax.legend()

plt.xlabel('Weeks relative to CT scan')

plt.ylabel('FVC value')

plt.title('FVC Time Series Clustering for each series in the training data')

fig.show()
test = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')

test = test.set_index('Weeks')

test = test[['Patient','FVC']]
fig = plt.figure(figsize=(11,7)); ax = fig.add_subplot(1, 1, 1)



#Plot the time series

for j in range(len(xvec)):

    ax.plot(xvec[j],yvec[j],color=colors[j],label="Time Series Cluster {0}".format(color_label[j]) if j in locs else "")

    

#Plot the test point measurements

for j in range(len(test)):

    plt.plot(test['FVC'].index[j],test['FVC'].iloc[j],'mo',markersize=15,label="Test Points".format(color_label[j]) if j ==0 else "")



# legend_handles, legend_labels = ax.get_legend_handles_labels()

# ax.legend(legend_handles, legend_labels)

ax.legend()

plt.xlabel('Weeks relative to CT scan')

plt.ylabel('FVC value')

plt.title('Time Series Clustering for the training data (test points indicated)')

fig.show()
n_features=1

predictions = model.predict(test.drop('Patient',axis=1).values.reshape(len(test),1,n_features))
predictions
test