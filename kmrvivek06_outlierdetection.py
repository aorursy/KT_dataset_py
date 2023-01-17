import pandas as pd

import os

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import stats

%matplotlib inline

import matplotlib.font_manager
from pyod.models.abod import ABOD

from pyod.models.cblof import CBLOF

from pyod.models.feature_bagging import FeatureBagging

from pyod.models.hbos import HBOS

from pyod.models.iforest import IForest

from pyod.models.knn import KNN

from pyod.models.lof import LOF

from pyod.utils.data import generate_data, get_outliers_inliers

from sklearn.preprocessing import MinMaxScaler
#List Files

os.listdir("../input/home-credit")
df = pd.read_csv("../input/home-credit/home_Credit.csv")
df.head()
#test_data = df[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']]

test_data = df[['AMT_ANNUITY','AMT_CREDIT']]

test_data.hist()
#Normal

for col in ['AMT_ANNUITY']:

    X = test_data[col]

    mean = np.mean(X)

    std = np.std(X)

    n = len(X)

    c = np.random.normal(mean,std,n)

    w,x,z = plt.hist(c,100,normed=True)

    plt.plot(x, 1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std**2)), linewidth=2,color='g')

    print(col,'Total Entries',n,' | Mean',mean,' | Sigma',std)

    plt.savefig(col+'_normal.png')

    plt.show()
#2-Sigma

for col in ['AMT_ANNUITY']:

    X = test_data[col]

    mean = np.mean(X)

    std = np.std(X)

    std2 = 2 * np.std(X)

    n = len(X)

    c = np.random.normal(mean,std,n)

    w,x,z = plt.hist(c,100,normed=True)

    plt.plot(x, 1/(std2*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std2**2)), linewidth=2,color='y')

    print(col,'Total Entries',n,' | Mean',mean,' | 2-Sigma',std2)

    plt.savefig(col+'_2sig.png')

    plt.show()
#3-Sigma

for col in ['AMT_ANNUITY']:

    X = test_data[col]

    mean = np.mean(X)

    std = np.std(X)

    std3 = 3 * np.std(X)

    n = len(X)

    c = np.random.normal(mean,std,n)

    w,x,z = plt.hist(c,100,normed=True)

    plt.plot(x, 1/(std3*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std3**2)), linewidth=2,color='r')

    print(col,'Total Entries',n,' | Mean',mean,' | 3-Sigma',std3)

    plt.savefig(col+'_3sig.png')

    plt.show()
#All

for col in ['AMT_ANNUITY']:

    X = test_data[col]

    mean = np.mean(X)

    std = np.std(X)

    std2 = 2 * np.std(X)

    std3 = 3 * np.std(X)

    n = len(X)

    c = np.random.normal(mean,std,n)

    w,x,z = plt.hist(c,100,normed=True)

    plt.plot(x, 1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std**2)), linewidth=2,color='g')

    plt.plot(x, 1/(std2*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std2**2)), linewidth=2,color='y')

    plt.plot(x, 1/(std3*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std3**2)), linewidth=2,color='r')

    print(col,'Total Entries',n,' | Mean',mean,' | Sigma',std,' | 2-Sigma',std2,' | 3-Sigma',std3)

    plt.savefig(col+'_all.png')

    plt.show()
#Normal

for col in ['AMT_CREDIT']:

    X = test_data[col]

    mean = np.mean(X)

    std = np.std(X)

    #std2 = 2 * np.std(X)

    #std3 = 3 * np.std(X)

    n = len(X)

    c = np.random.normal(mean,std,n)

    w,x,z = plt.hist(c,100,normed=True)

    plt.plot(x, 1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std**2)), linewidth=2,color='g')

    #plt.plot(x, 1/(std2*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std2**2)), linewidth=2,color='y')

    #plt.plot(x, 1/(std3*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std3**2)), linewidth=2,color='r')

    print(col,'Total Entries',n,' | Mean',mean,' | Sigma',std)

    plt.savefig(col+'_normal.png')

    plt.show()
#2-Sigma

for col in ['AMT_CREDIT']:

    X = test_data[col]

    mean = np.mean(X)

    std = np.std(X)

    std2 = 2 * np.std(X)

    #std3 = 3 * np.std(X)

    n = len(X)

    c = np.random.normal(mean,std,n)

    w,x,z = plt.hist(c,100,normed=True)

    #plt.plot(x, 1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std**2)), linewidth=2,color='g')

    plt.plot(x, 1/(std2*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std2**2)), linewidth=2,color='y')

    #plt.plot(x, 1/(std3*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std3**2)), linewidth=2,color='r')

    print(col,'Total Entries',n,' | Mean',mean,' | 2-Sigma',std2)

    plt.savefig(col+'_2sig.png')

    plt.show()
#3-Sigma

for col in ['AMT_CREDIT']:

    X = test_data[col]

    mean = np.mean(X)

    std = np.std(X)

    #std2 = 2 * np.std(X)

    std3 = 3 * np.std(X)

    n = len(X)

    c = np.random.normal(mean,std,n)

    w,x,z = plt.hist(c,100,normed=True)

    #plt.plot(x, 1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std**2)), linewidth=2,color='g')

    #plt.plot(x, 1/(std2*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std2**2)), linewidth=2,color='y')

    plt.plot(x, 1/(std3*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std3**2)), linewidth=2,color='r')

    print(col,'Total Entries',n,' | Mean',mean,' | 3-Sigma',std3)

    plt.savefig(col+'_3sig.png')

    plt.show()
#All

for col in ['AMT_CREDIT']:

    X = test_data[col]

    mean = np.mean(X)

    std = np.std(X)

    std2 = 2 * np.std(X)

    std3 = 3 * np.std(X)

    n = len(X)

    c = np.random.normal(mean,std,n)

    w,x,z = plt.hist(c,100,normed=True)

    plt.plot(x, 1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std**2)), linewidth=2,color='g')

    plt.plot(x, 1/(std2*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std2**2)), linewidth=2,color='y')

    plt.plot(x, 1/(std3*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std3**2)), linewidth=2,color='r')

    print(col,'Total Entries',n,' | Mean',mean,' | Sigma',std,' | 2-Sigma',std2,' | 3-Sigma',std3)

    plt.savefig(col+'_all.png')

    plt.show()
test_data.corr()
#columns = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']

columns = ['AMT_ANNUITY','AMT_CREDIT']
#for k,v in df_dict.items():

#    columns = var_dict[k]

for col in columns:

    mean = test_data[col].mean()

    std = test_data[col].std()

    sigma2 = std*2

    print (col,': mean : ',mean,': std : ', std,': 2-sigma : ', sigma2)
list(test_data.columns)
random_state = np.random.RandomState(42)

outliers_fraction = 0.05

# Define seven outlier detection tools to be compared

classifiers = {

        'ABOD : Angle-based Outlier Detector': ABOD(contamination=outliers_fraction),

        'CBLOF : Cluster-based Local Outlier Factor':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),

        'FB : Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),

        'HNOD : Histogram-base Outlier Detection': HBOS(contamination=outliers_fraction),

        'IF : Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),

        'KNN : K Nearest Neighbors': KNN(contamination=outliers_fraction),

        'AvgKNN : Average KNN': KNN(method='mean',contamination=outliers_fraction)

}
test_data = test_data.fillna(0)
#for df_name, frame in df_dict.items():

df_dict = {'test_data':test_data}

dfx = test_data.copy()

for df_name, frame in df_dict.items():

    columns = list(frame.columns)

    X_df = frame.loc[:,columns]

    scaler = MinMaxScaler()

    X_df.loc[:,columns] = scaler.fit_transform(X_df[columns])

    X = X_df.values

    print("\n\n[+] Detecting outliers for frame", df_name,"columns :",columns,"size :",X_df.shape[0])

    #xx , yy = np.meshgrid(np.linspace(0,1 , X_df.shape[0]), np.linspace(0, 1, X_df.shape[0]))

    #Graph

    print('Initial Graph')

    if X.shape[1] == 3:

        fig = plt.figure(figsize=(8,8))

        ax = fig.add_subplot(111, projection='3d')

        xs = frame[columns[0]].values

        ys = frame[columns[1]].values

        zs = frame[columns[2]].values

        ax.scatter(xs, ys, zs, marker='o')



        ax.set_xlabel(columns[0])

        ax.set_ylabel(columns[1])

        ax.set_zlabel(columns[2])

        plt.title(df_name)

        plt.savefig(df_name+'.png',)

        plt.show()

    elif X.shape[1] == 2:

        fig = plt.figure(figsize=(8,8))

        ax = fig.add_subplot(111)

        xs = frame[columns[0]].values

        ys = frame[columns[1]].values

        ax.scatter(xs, ys, marker='o')



        ax.set_xlabel(columns[0])

        ax.set_ylabel(columns[1])

        plt.title(df_name)

        plt.savefig(df_name+'.png',)

        plt.show()

    else:

        frame[columns].hist(bins=50)

    for i, (clf_name, clf) in enumerate(classifiers.items()):

        try:        

            clf.fit(X)

            # predict raw anomaly score

            #scores_pred = clf.decision_function(X) * -1

        

            # prediction of a datapoint category outlier or inlier

            y_pred = clf.predict(X)

            n_inliers = len(y_pred) - np.count_nonzero(y_pred)

            n_outliers = np.count_nonzero(y_pred == 1)

        

            # copy of dataframe

            

            out_col_name = 'outlier '+str(clf_name.split(' ')[0])

            dfx[out_col_name] = y_pred.tolist()

        

            # IXn inliner features

            IX_list = []

            for col in columns:

                IX =  np.array(dfx[col][dfx[out_col_name] == 0]).reshape(-1,1)

                IX_list.append(IX)

            

            # OXn - outlier features

            OX_list = []

            for col in columns:

                OX =  dfx[col][dfx[out_col_name] == 1].values.reshape(-1,1)

            

            print('OUTLIERS : ',n_outliers,' | INLIERS : ',n_inliers, " | ",clf_name)

            

        except Exception as e:

            print("[-] Algoithm not supported :",clf_name,e)

        

dfx.to_csv(df_name+'.csv', index=False)      
dfx.head()
param_dict = {'ABOD':'outlier ABOD','CBLOF':'outlier CBLOF','FB':'outlier FB','HNOD':'outlier HNOD','IF':'outlier IF','KNN':'outlier KNN','AvgKnn':'outlier AvgKNN'}

param_dict
columns = list(test_data.columns)

print(columns)

for k,v in param_dict.items():

    print("Modelling  graph for",k)

    if len(columns) == 3:

        fig = plt.figure(figsize=(8,8))

        ax = fig.add_subplot(111, projection='3d')

        frame = dfx[dfx[v] == 0]

        xs = frame[columns[0]].values

        ys = frame[columns[1]].values

        zs = frame[columns[2]].values

        ax.scatter(xs, ys, zs, marker='o')

    

        frame1 = dfx[dfx[v] == 1]

        xs = frame1[columns[0]].values

        ys = frame1[columns[1]].values

        zs = frame1[columns[2]].values

        ax.scatter(xs, ys, zs, marker='^')



        ax.set_xlabel(columns[0])

        ax.set_ylabel(columns[1])

        ax.set_zlabel(columns[2])

        plt.title(k)

        plt.savefig(k+'.png',)

        plt.show()

    elif len(columns) == 2:

        fig = plt.figure(figsize=(8,8))

        ax = fig.add_subplot(111)

        frame = dfx[dfx[v] == 0]

        xs = frame[columns[0]].values

        ys = frame[columns[1]].values

        ax.scatter(xs, ys, marker='o')

    

        frame1 = dfx[dfx[v] == 1]

        xs = frame1[columns[0]].values

        ys = frame1[columns[1]].values

        ax.scatter(xs, ys, marker='^')



        ax.set_xlabel(columns[0])

        ax.set_ylabel(columns[1])

        plt.title(k)

        plt.savefig(k+'.png',)

        plt.show()