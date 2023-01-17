"""

import os

import glob

import pandas as pd

#os.chdir("/mydir")





files = [i for i in glob.glob('/kaggle/input/one-year-industrial-component-degradation/*.{}'.format('csv'))]

files





extension = 'csv'

all_filenames = [i for i in glob.glob('/kaggle/input/one-year-industrial-component-degradation/*[mode1].{}'.format(extension))] + \

                [i for i in glob.glob('/kaggle/input/one-year-industrial-component-degradation/oneyeardata/*[mode1].{}'.format(extension))]

#print(all_filenames)



#combine all files in the list

df = pd.concat([pd.read_csv(f) for f in all_filenames ])

#export to csv

df.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')



"""
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
"""

filenames = os.listdir('/kaggle/input/one-year-industrial-component-degradation/')

filenames = [i.strip(".csv") for i in filenames]

filenames.sort()

filenames.remove('oneyeardata')



parsed_filenames = []

for name in filenames:

    temp = name.split("T")

    month, date = temp[0].split("-")

    rhs = temp[1].split("_")

    hours, minutes, seconds = rhs[0][:2], rhs[0][2:4], rhs[0][4:]

    sample_no = rhs[1]

    mode = rhs[2][-1]

    # Now we have Month, Date, Hours, Minutes, Seconds, Sample Number, Mode 

    parsed_filenames.append([month, date, hours, minutes, seconds, sample_no, mode])

    

parsed_filenames = pd.DataFrame(parsed_filenames, columns=["Month", "Date", "Hours", "Minutes", "Seconds", "Sample Number", "Mode"])



for i in parsed_filenames.columns:

    parsed_filenames[i] = pd.to_numeric(parsed_filenames[i], errors='coerce')







path = '/kaggle/input/one-year-industrial-component-degradation/'

df = pd.DataFrame()

#f = pd.read_csv(path+filenames[0]+".csv")

#f = f.join(parsed_filenames[0:1], how='left')

#f = f.fillna(method='ffill')

#f

for ind, file in enumerate(filenames):

    file_content = pd.read_csv(path+file+".csv")

    file_content = file_content.join(parsed_filenames[ind:ind+1], how='left')

    file_content.fillna(method='ffill', inplace=True)

    

    if df.empty:

        df = file_content

        df.fillna(method='ffill', inplace=True)

    else:

        df = df.append(file_content, ignore_index=True)

        df.fillna(method='ffill', inplace=True)



        

for i in ['Mode', 'Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month']:

    df[i] = pd.to_numeric(df[i], downcast='integer')

df.info()



"""
"""

if not os.path.exists('/kaggle/working/compiled_df'):

    os.makedirs('/kaggle/working/compiled_df')

#Saves dataframe to a csv file, removes a index

df.to_csv('/kaggle/working/compiled_df/Combined.csv', index=False)



"""
df = pd.read_csv("../input/combineddataset/Combined.csv")

for i in ['Mode', 'Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month']:

    df[i] = pd.to_numeric(df[i], downcast='integer')

df.info()
df.head(10)
data = df.copy()

data = data[:10000]

data=data.drop(['Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month'],axis=1)



#df=pd.get_dummies(data)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 

num2 = scaler.fit_transform(data.drop(['timestamp'],axis=1))

num2 = pd.DataFrame(num2, columns = data.drop(['timestamp'],axis=1).columns)
from sklearn.cluster import DBSCAN

outlier_detection = DBSCAN(

 eps = .2, 

 metric='euclidean', 

 min_samples = 5,

 n_jobs = -1)

clusters = outlier_detection.fit_predict(num2)
clusters.shape
data['anomaly'] = pd.Series(clusters)

data.head()
data['anomaly'].unique()
X_anomaly = data[data['anomaly'] == -1]

X_normal = data[data['anomaly'] != -1]

print(X_anomaly.shape, X_normal.shape)

#from matplotlib import cm

#cmap = cm.get_cmap('Set1’)

#data.plot.scatter(x='Spend_Score',y='Income', c=clusters, cmap=cmap, colorbar = False)
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
data.columns
"""

cols = ['pCut::Motor_Torque',

       'pCut::CTRL_Position_controller::Lag_error',

       'pCut::CTRL_Position_controller::Actual_position',

       'pCut::CTRL_Position_controller::Actual_speed',

       'pSvolFilm::CTRL_Position_controller::Actual_position',

       'pSvolFilm::CTRL_Position_controller::Actual_speed',

       'pSvolFilm::CTRL_Position_controller::Lag_error', 'pSpintor::VAX_speed',

       'Mode']



pca = PCA(n_components=2)

data_pca = pca.fit_transform(data[cols].values)



#data['pca-one'] = data_pca[:,0]

#data['pca-two'] = data_pca[:,1] 



print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

"""
anomaly_ind = data[data['anomaly']==-1].index

normal_ind = data[data['anomaly']!=-1].index
anomaly_pca = pd.DataFrame(data_pca[anomaly_ind])

normal_pca = pd.DataFrame(data_pca[normal_ind])

anomaly_pca
data.columns
import matplotlib.pyplot as plt

import seaborn as sns



features = ['pCut::Motor_Torque',

       'pCut::CTRL_Position_controller::Lag_error',

       'pCut::CTRL_Position_controller::Actual_position',

       'pCut::CTRL_Position_controller::Actual_speed',

       'pSvolFilm::CTRL_Position_controller::Actual_position',

       'pSvolFilm::CTRL_Position_controller::Actual_speed',

       'pSvolFilm::CTRL_Position_controller::Lag_error', 'pSpintor::VAX_speed',

       'Mode']



for feature in features:

    plt.figure(figsize=(15,7))

    plt.plot(data[feature], color='blue', label = 'normal')

    plt.scatter(x=data.iloc[anomaly_ind].index, y=data.iloc[anomaly_ind][feature], color='red', label = 'anomalous')

    #plt.scatter(x=normal_pca[0], y=normal_pca[1], color='blue')

    plt.title(feature)

    plt.legend()
data = df.copy()

data = data[:10000]

data=data.drop(['timestamp', 'Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month'],axis=1)

from sklearn.ensemble import IsolationForest

rs=np.random.RandomState(0)

clf = IsolationForest(max_samples=100,random_state=rs, contamination=.1) 

clf.fit(data)

if_scores = clf.decision_function(data)

if_anomalies=clf.predict(data)

#print(if_anomalies)

if_anomalies=pd.Series(if_anomalies).replace([-1,1],[1,0])

#print(if_anomalies)

#if_anomalies=num[if_anomalies==1];
plt.figure(figsize=(12,8))

plt.hist(if_scores);

plt.title('Histogram of Avg Anomaly Scores: Lower => More Anomalous');
anomaly_ind = if_anomalies[if_anomalies==1].index
features = ['pCut::Motor_Torque',

       'pCut::CTRL_Position_controller::Lag_error',

       'pCut::CTRL_Position_controller::Actual_position',

       'pCut::CTRL_Position_controller::Actual_speed',

       'pSvolFilm::CTRL_Position_controller::Actual_position',

       'pSvolFilm::CTRL_Position_controller::Actual_speed',

       'pSvolFilm::CTRL_Position_controller::Lag_error', 'pSpintor::VAX_speed',

       'Mode']



for feature in features:

    plt.figure(figsize=(15,7))

    #cmap=np.array(['white','red'])

    plt.scatter(data.index,data[feature],c='green', label = 'normal')

    plt.scatter(anomaly_ind,data.iloc[anomaly_ind][feature],c='red', label='anomaly')

    plt.ylabel(feature)

    plt.title(feature)

    plt.legend()
data = df.copy()

data = data[:10000]

data=data.drop(['timestamp', 'Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month'],axis=1)
from sklearn import svm

clf=svm.OneClassSVM(nu=.1,kernel='rbf', gamma='auto')

clf.fit(data)

y_pred=clf.predict(data)
y_pred = pd.Series(y_pred).replace([-1,1],[1,0])
anomaly_ind = y_pred[y_pred==1].index
anomaly_ind
features = ['pCut::Motor_Torque',

       'pCut::CTRL_Position_controller::Lag_error',

       'pCut::CTRL_Position_controller::Actual_position',

       'pCut::CTRL_Position_controller::Actual_speed',

       'pSvolFilm::CTRL_Position_controller::Actual_position',

       'pSvolFilm::CTRL_Position_controller::Actual_speed',

       'pSvolFilm::CTRL_Position_controller::Lag_error', 'pSpintor::VAX_speed',

       'Mode']



for feature in features:

    plt.figure(figsize=(15,7))

    #cmap=np.array(['white','red'])

    plt.scatter(data.index,data[feature],c='green', label = 'normal')

    plt.scatter(anomaly_ind,data.iloc[anomaly_ind][feature],c='red', label='anomaly')

    plt.ylabel(feature)

    plt.title(feature)

    plt.legend()