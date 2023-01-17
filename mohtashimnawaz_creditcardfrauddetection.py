import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

print(data.shape)

data.head()
data.dtypes
# Class count

print(data['Class'].value_counts())

sns.countplot(data=data, x='Class')
# Distribution of Amount

plt.figure(figsize=(20,6))

ax1 = plt.subplot(1,3,1)

ax2 = plt.subplot(1,3,2)

ax3 = plt.subplot(1,3,3)

sns.distplot(data['Amount'], ax=ax1)

sns.distplot((data[data['Class']==0])['Amount'], ax=ax2)

sns.distplot((data[data['Class']==1])['Amount'], ax=ax3)
# Investigating time

print(data['Time'].value_counts())

plt.figure(figsize=(20,6))

ax1 = plt.subplot(1,3,1)

ax2 = plt.subplot(1,3,2)

ax3 = plt.subplot(1,3,3)

sns.distplot(data['Time'],ax=ax1)

sns.distplot(data[data['Class']==1]['Time'],ax=ax2)

sns.distplot(data[data['Class']==0]['Time'],ax=ax3)
# Other variables are normalized so let's plot correlation matrix

plt.figure(figsize=(10,10))

sns.heatmap(data.corr())
def choseRandom(d, c):

    """d is the original dataframe and c is the column name denoting variable to be predicted(class)"""

    values = d[c].unique()

    lst =[]

    for i in values:

        lst.append(d[d[c]==i][c].count())

    min_count = min(lst)

    df = pd.DataFrame()

    for i in values:

        df = df.append(d[d[c]==i].sample(n=min_count))

        

    return df
sampled_random = choseRandom(data,'Class')

print(sampled_random.shape)

sampled_random.head()
plt.figure(figsize=(6,4))

sns.set(style='darkgrid')

sns.countplot(sampled_random['Class'])

plt.xlabel("Class Distribution")

plt.title('Class Distribution of resampled data')

plt.show()
# Plotting distributions of 'Amount'

plt.figure(figsize=(20,6))

ax1 = plt.subplot(1,3,1)

ax2 = plt.subplot(1,3,2)

ax3 = plt.subplot(1,3,3)

sns.set(style='darkgrid')

sns.distplot(sampled_random['Amount'], ax=ax1)

sns.distplot((sampled_random[sampled_random['Class']==0])['Amount'], ax=ax2)

sns.distplot((sampled_random[sampled_random['Class']==1])['Amount'], ax=ax3)
# Heatmap on new sampled data

plt.figure(figsize=(8,6))

sns.heatmap(sampled_random.corr())

plt.show()
from sklearn.model_selection import train_test_split

def splitData(d, test_size=0.2):

    train, test = train_test_split(d, stratify = d['Class'], test_size=test_size)

    return train,test



train, test = splitData(data)
# Let us plot some box plot for Class-vs-Amount

sns.set(style='darkgrid')

sns.boxplot(x = 'Class', y = 'Amount', data = sampled_random, hue='Class')
# boxplots for variables V1-V28

plt.figure(figsize=(20,10))

ax1 = plt.subplot(2,4,1)

ax2 = plt.subplot(2,4,2)

ax3 = plt.subplot(2,4,3)

ax4 = plt.subplot(2,4,4)

ax5 = plt.subplot(2,4,5)

ax6 = plt.subplot(2,4,6)

ax7 = plt.subplot(2,4,7)

ax8 = plt.subplot(2,4,8)

sns.set(style='darkgrid')

sns.boxplot(x = 'Class', y = 'V1', data = sampled_random, hue='Class', ax=ax1)

sns.boxplot(x = 'Class', y = 'V2', data = sampled_random, hue='Class', ax=ax2)

sns.boxplot(x = 'Class', y = 'V28', data = sampled_random, hue='Class', ax=ax3)

sns.boxplot(x = 'Class', y = 'V26', data = sampled_random, hue='Class', ax=ax4)

sns.boxplot(x = 'Class', y = 'V20', data = sampled_random, hue='Class', ax=ax5)

sns.boxplot(x = 'Class', y = 'V18', data = sampled_random, hue='Class', ax=ax6)

sns.boxplot(x = 'Class', y = 'V12', data = sampled_random, hue='Class', ax=ax7)

sns.boxplot(x = 'Class', y = 'V8', data = sampled_random, hue='Class', ax=ax8)
# Cleaning up the outliers 

def cleanOutliers(d, q1, q3, r):

    """d is the data frame, q1 and q3 are quantiles and r is the range above of below quantiles, 

    Generally, q1=0.25, q3=0.75, r=1.5"""

    cols = d.columns

    for i in cols:

        Q1 = d[i].quantile(q1)

        Q3 = d[i].quantile(q3)

        IQR = Q3-Q1

        low = Q1 - r*IQR

        high = Q3 + r*IQR

        d = d[(d[i]>=low) & (d[i]<=high)]

    return d
temp = cleanOutliers(train, 0.25, 0.75, 1.5)

print("Class count for 0.25-0.75 and 1.5 IQR fields",temp['Class'].value_counts())

temp = cleanOutliers(train, 0.2, 0.8, 2.0)

print("Class count for 0.2-0.8 and 2.0 IQR fields",temp['Class'].value_counts())

temp = cleanOutliers(train, 0.1, 0.9, 2.5)

print("Class count for 0.1-0.9 and 2.5 IQR fields",temp['Class'].value_counts())

temp = cleanOutliers(train, 0, 1, 2.5)

print("Class count including all samples",temp['Class'].value_counts())
def logTransform(d,col):

    d[col] = np.log1p(d[col])

    return d

def invLog(d, col):

    d[col] = np.expm1(d[col])

    return d[col]
train = logTransform(train, 'Amount')

test = logTransform(test, 'Amount')

plt.figure(figsize=(20,8))

ax1 = plt.subplot(1,2,1)

ax2 = plt.subplot(1,2,2)

sns.distplot(train['Amount'], ax=ax1)

sns.distplot(test['Amount'],ax=ax2)

plt.show()
from sklearn.preprocessing import StandardScaler

def getScaleDataParams(d, col):

    trans = StandardScaler().fit(d[[col]])

    return trans

def scaleData(d, col, param):

    d[[col]] = param.transform(d[[col]])

    return d

def scaleDF(d):

    """Pass a dataFrame to scale all columns, provided that all the columns are numeric"""

    cols = d.columns

    for i in cols:

        params = getScaleDataParams(d,i)

        d = scaleData(d, i, params)

    return d
train.iloc[:,:-1] = scaleDF(train.iloc[:,:-1])

test.iloc[:,:-1] = scaleDF(test.iloc[:,:-1])

train.head()
# Undersampling on training data

random_train = choseRandom(train,'Class')

print(random_train.shape)

random_train.head()
# PCA

from sklearn.decomposition import PCA

def getFullPCAData(d):

    pca = PCA()

    pca_data = pca.fit_transform(d)

    return pca_data, pca.explained_variance_ratio_*100



def getPCAData(d, n_components=2):

    pca = PCA(n_components=n_components)

    pca_data = pca.fit_transform(d)

    return pca_data, pca.explained_variance_ratio_*100
def plotPCAVariations(pca_data,var):

    per_var = np.round(var, decimals=1)

    labels = ["PC"+str(x) for x in range(1,len(per_var)+1)]

    plt.figure(figsize=(20,6))

    plt.bar(x = range(1,len(per_var)+1), height = per_var, tick_label = labels)

    plt.show()
pca_data, var = getFullPCAData(random_train.iloc[:,:-1])

plotPCAVariations(pca_data,var)
def plotPCAData(d, mask):

    """Plots PCA Data considering n_components = 2"""

    pca_2c, var = getPCAData(d,2)

    pca_df = pd.DataFrame(pca_2c, columns=['PC1','PC2'])

    pca_df['Class'] = mask.values

    plt.figure(figsize=(8,6))

    sns.scatterplot(x='PC1', y='PC2',hue='Class',data = pca_df)

    plt.show()

    return pca_df

    

mask = random_train.iloc[:,-1]

pca_df = plotPCAData(random_train.iloc[:,:-1], mask)
from sklearn.manifold import TSNE

def getTSNE(d):

    tsne = TSNE()

    d_emb = tsne.fit_transform(d)

    return d_emb
d_emb = getTSNE(random_train.iloc[:,:-1])
def plotTSNEData(d, mask):

    """Plots PCA Data considering n_components = 2"""

    d_emb = getTSNE(d)

    tsne_df = pd.DataFrame(d_emb, columns=['C1','C2'])

    tsne_df['Class'] = mask.values

    plt.figure(figsize=(8,6))

    sns.scatterplot(x='C1', y='C2',hue='Class',data = tsne_df)

    plt.show()

    return tsne_df

    

mask = random_train.iloc[:,-1]

tsne_df = plotTSNEData(random_train.iloc[:,:-1], mask)