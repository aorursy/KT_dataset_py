import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

pd.set_option("display.max_columns",None) 

pd.set_option("display.max_rows",None) 



import warnings

warnings.filterwarnings("ignore")



from IPython.display import Image

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN

from sklearn.neighbors import LocalOutlierFactor

sns.set(style="darkgrid", palette="pastel", color_codes=True)

sns.set_context('talk')



from pathlib import Path

data_dir = Path('../input/images')
Image(filename=data_dir/'O2.jpeg')
Image(filename=data_dir/'O5.png')
Image(filename=data_dir/'O6.png')
df_1 = pd.read_csv("../input/heart-disease-uci/heart.csv")

df_1.head()
df_1.isnull().sum()
df_1.describe()
plt.figure(figsize = (4,8))

sns.boxplot(y = df_1.chol)
Image(filename=data_dir/'O3.png')
def out_iqr(df , column):

    global lower,upper

    q25, q75 = np.quantile(df[column], 0.25), np.quantile(df[column], 0.75)

    # calculate the IQR

    iqr = q75 - q25

    # calculate the outlier cutoff

    cut_off = iqr * 1.5

    # calculate the lower and upper bound value

    lower, upper = q25 - cut_off, q75 + cut_off

    print('The IQR is',iqr)

    print('The lower bound value is', lower)

    print('The upper bound value is', upper)

    # Calculate the number of records below and above lower and above bound value respectively

    df1 = df[df[column] > upper]

    df2 = df[df[column] < lower]

    return print('Total number of outliers are', df1.shape[0]+ df2.shape[0])
out_iqr(df_1,'chol')

#Input the dataset and the required column
plt.figure(figsize = (10,6))

sns.distplot(df_1.chol, kde=False)

plt.axvspan(xmin = lower,xmax= df_1.chol.min(),alpha=0.2, color='red')

plt.axvspan(xmin = upper,xmax= df_1.chol.max(),alpha=0.2, color='red')
#Data Frame without outliers

df_new = df_1[(df_1['chol'] < upper) | (df_1['chol'] > lower)]
Image(filename=data_dir/'O4.png')
df_2 = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")

df_2.head()
plt.figure(figsize = (10,5))

sns.distplot(df_2['writing score'])
def out_std(df, column):

    global lower,upper

    # calculate the mean and standard deviation of the data frame

    data_mean, data_std = df[column].mean(), df[column].std()

    # calculate the cutoff value

    cut_off = data_std * 3

    # calculate the lower and upper bound value

    lower, upper = data_mean - cut_off, data_mean + cut_off

    print('The lower bound value is', lower)

    print('The upper bound value is', upper)

    # Calculate the number of records below and above lower and above bound value respectively

    df1 = df[df[column] > upper]

    df2 = df[df[column] < lower]

    return print('Total number of outliers are', df1.shape[0]+ df2.shape[0])
out_std(df_2,'writing score')
plt.figure(figsize = (10,5))

sns.distplot(df_2['writing score'], kde=False)

plt.axvspan(xmin = lower,xmax= df_2['writing score'].min(),alpha=0.2, color='red')

plt.axvspan(xmin = upper,xmax= df_2['writing score'].max(),alpha=0.2, color='red')
#Data Frame without outliers

df_new = df_2[(df_2['writing score'] < upper) | (df_2['writing score'] > lower)]
df_3 = pd.read_csv("../input/insurance/insurance.csv")

df_3.head()
df_3.describe()
df_3.isnull().sum()
plt.figure(figsize = (10,5))

sns.distplot(df_3['charges'])
def out_zscore(data):

    global outliers,zscore

    outliers = []

    zscore = []

    threshold = 3

    mean = np.mean(data)

    std = np.std(data)

    for i in data:

        z_score= (i - mean)/std 

        zscore.append(z_score)

        if np.abs(z_score) > threshold:

            outliers.append(i)

    return print("Total number of outliers are",len(outliers))
out_zscore(df_3.charges)
plt.figure(figsize = (10,5))

sns.distplot(zscore)

plt.axvspan(xmin = 3 ,xmax= max(zscore),alpha=0.2, color='red')
#Data Frame without outliers

df_new = df_3[(df_3['charges'] < 3) | (df_3['charges'] > -3)]
Image(filename=data_dir/'O8.png')
#Import necessary libraries

from sklearn.ensemble import IsolationForest

#The required columns

cols = ['writing score','reading score','math score']

#Plotting the sub plot

fig, axs = plt.subplots(1, 3, figsize=(20, 5), facecolor='w', edgecolor='k')

axs = axs.ravel()



for i, column in enumerate(cols):

    isolation_forest = IsolationForest(contamination='auto')

    isolation_forest.fit(df_2[column].values.reshape(-1,1))



    xx = np.linspace(df_2[column].min(), df_2[column].max(), len(df_2)).reshape(-1,1)

    anomaly_score = isolation_forest.decision_function(xx)

    outlier = isolation_forest.predict(xx)

    

    axs[i].plot(xx, anomaly_score, label='anomaly score')

    axs[i].fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 

                     where=outlier==-1, color='r', 

                     alpha=.4, label='outlier region')

    axs[i].legend()

    axs[i].set_title(column)
Image(filename=data_dir/'O9.png')
X = df_3[['age','bmi']].values



db = DBSCAN(eps=3.0, min_samples=10).fit(X)

labels = db.labels_
pd.Series(labels).value_counts()
plt.figure(figsize=(12,12))



unique_labels = set(labels)

colors = ['blue', 'red']



for color,label in zip(colors, unique_labels):

    sample_mask = [True if l == label else False for l in labels]

    plt.plot(X[:,0][sample_mask], X[:, 1][sample_mask], 'o', color=color);

plt.xlabel('Age');

plt.ylabel('BMI');
Image(filename=data_dir/'O10.png')
clf = LocalOutlierFactor(n_neighbors=50, contamination='auto')

X = df_1[['age','chol']].values

y_pred = clf.fit_predict(X)
plt.figure(figsize=(12,12))

# plot the level sets of the decision function



in_mask = [True if l == 1 else False for l in y_pred]

out_mask = [True if l == -1 else False for l in y_pred]



plt.title("Local Outlier Factor (LOF)")

# inliers

a = plt.scatter(X[in_mask, 0], X[in_mask, 1], c = 'blue',

                edgecolor = 'k', s = 30)

# outliers

b = plt.scatter(X[out_mask, 0], X[out_mask, 1], c = 'red',

                edgecolor = 'k', s = 30)

plt.axis('tight')

plt.xlabel('Age');

plt.ylabel('Cholestrol');

plt.show()