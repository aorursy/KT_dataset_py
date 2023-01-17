import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn import preprocessing

from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns

from pandas import DataFrame,Series

from sklearn.linear_model import LinearRegression



from sklearn import metrics, svm

from sklearn.linear_model           import LinearRegression

from sklearn.linear_model           import LogisticRegression

from sklearn.tree                   import DecisionTreeClassifier

from sklearn.neighbors              import KNeighborsClassifier

from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis

from sklearn.naive_bayes            import GaussianNB

from sklearn.svm                    import SVC

from sklearn import preprocessing

from sklearn import utils



data = pd.read_csv('../input/TestMissingData.csv')

data_train = pd.read_csv('../input/CompletedDataset.csv')

data.head(10)
# Check the current values

data = data.drop(['City'], axis = 1)

data.describe()
total = data.isnull().sum().sort_values(ascending=False)

percent =(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
dataFillMean = data.fillna(data.mean())

dataFillMean.describe()
def CovSimilarity(data):

    cov = data.corr()

    return cov

    

CovSimilarity(dataFillMean)
from matplotlib.collections import EllipseCollection

def plot_corr_ellipses(data, ax=None, **kwargs):

    

    M = np.array(data)

    if not M.ndim == 2:

        raise ValueError('data must be a 2D array')

    if ax is None:

        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect':'equal'})

        ax.set_xlim(-0.5, M.shape[1] - 0.5)

        ax.set_ylim(-0.5, M.shape[0] - 0.5)

    # xy locations of each ellipse center

    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    

    # set the relative sizes of the major/minor axes according to the strength

    # the positive/negative correlation

    w = np.ones_like(M).ravel()

    h = 1 - np.abs(M).ravel()

    a = 45 * np.sign(M).ravel()

    

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,

                           transOffset=ax.transData, array=M.ravel(), **kwargs)

    ax.add_collection(ec)

    

    if isinstance(data, pd.DataFrame):

        ax.set_xticks(np.arange(M.shape[1]))

        ax.set_xticklabels(data.columns, rotation=90)

        ax.set_yticks(np.arange(M.shape[0]))

        ax.set_yticklabels(data.index)

    

    return ec

 

fig, ax = plt.subplots(1, 1)

m = plot_corr_ellipses(data.corr(), ax=ax, cmap='Greens')

cb = fig.colorbar(m)

cb.set_label('Correlation coefficient')

ax.margins(0.1)

current_fig = plt.gcf()  

current_fig.savefig('my_0.pdf', bbox_inches='tight')  

 

import seaborn as sns

sns.clustermap(data=data.corr(), annot=True, cmap='Greens').savefig('my_1.pdf', bbox_inches='tight')

def CosSimilarity(data):

    col = data.shape[1]

    row = data.shape[0]

    # Standardize data sets and convert them to multidimensional arrays

    dataPre = preprocessing.scale(data)



    dataRel = np.array(dataPre)

    dataRel = dataRel.reshape(col,row)

    

    # Cosine similarity matrix

    cos = pd.DataFrame(cosine_similarity(dataRel))

    return cos



CosSimilarity(dataFillMean)
# Clustering via Geography Attributes

from mpl_toolkits.mplot3d import Axes3D



# Loading Geography Features subdataset

dataGeo = data[['AQI', 'Longititute', 'Latitude', 'Altitude']]

# print(dataGeo.head())



# Creat a #d Plot Project

x, y, z = dataGeo['Longititute'], dataGeo['Latitude'], dataGeo['Altitude']

ax = plt.subplot(111, projection='3d')



# Use K-Means Clustering

from sklearn.cluster import KMeans

y_pred = KMeans(n_clusters=5, random_state=6).fit_predict(dataGeo)

ax.scatter(x, y, z, c = y_pred)  



# Plot Axis

ax.set_zlabel('Altitude') 

ax.set_ylabel('Latitude')

ax.set_xlabel('Longititute')

plt.title('K-Means Clustering 3D Graph')

plt.show()



#聚类二维平面图

plt.scatter(x,y,c = y_pred)

plt.xlabel('Longititude')

plt.ylabel('Latitude')

plt.title('K-Means Clustering Planar Graph')

plt.show()#显示模块中的所有绘图对象



# Merging the Dataset and Labels from Cluster

label = pd.DataFrame(y_pred)

label.columns = ['label']

label.head()

dataLab = pd.merge(data, label, how='right', left_index=True, right_index=True, sort=False)





# Function to Fix the Dataset

def FillMean(S):

    cluNum = S['label'].value_counts()

    for i in range(len(cluNum)):

        cluName = cluNum.index[i]

        sub = S[S['label'] == cluName]

        S[S['label'] == cluName] = sub.fillna(sub.mean())

    return S



dataFixed = FillMean(dataLab)

dataFixed.describe()
# Evaluate the K-Means clustering model effects of the currently selected parameters

from sklearn import metrics

metrics.calinski_harabaz_score(dataGeo, y_pred)  
# The similarity calculation of filled in the missing data after k-means clustering

data_corr1 = CovSimilarity(dataFixed)

data_corr1
# Visualization of similarity calculation results

import seaborn as sns

sns.clustermap(data=data_corr1, annot=True, cmap='Blues').savefig('my_1.pdf', bbox_inches='tight')

data_train_corr = CovSimilarity(data)
# Create a 3D drawing project

x, y, z = dataGeo['Longititute'], dataGeo['Latitude'], dataGeo['Altitude']

ax = plt.subplot(111, projection='3d')



# Use DBSCAN clustering

from sklearn.cluster import DBSCAN

y_pred2 = DBSCAN(eps =15, min_samples = 7).fit_predict(dataGeo)

ax.scatter(x, y, z, c = y_pred2) 



# print(y_pred[:20])



# Draw axes

ax.set_zlabel('Altitude') 

ax.set_ylabel('Latitude')

ax.set_xlabel('Longititute')

plt.title('DBSCAN Clustering 3D Graph')

plt.show()



# Clustering 2D plan

plt.scatter(x,y,c = y_pred2)

plt.xlabel('Longititude')

plt.ylabel('Latitude')

plt.title('DBSCAN Clustering Planar Graph')

plt.show()#显示模块中的所有绘图对象



# Merging the Dataset and Labels from Cluster

label = pd.DataFrame(y_pred2)

label.columns = ['label']

label.head()

dataLab = pd.merge(data, label, how='right', left_index=True, right_index=True, sort=False)



dataLab = pd.merge(data, label, how='right', left_index=True, right_index=True, sort=False)

dataFixed2 = FillMean(dataLab)

dataFixed2.describe()
# Evaluate the DBSCAN clustering model effects of the currently selected parameters

from sklearn import metrics

metrics.calinski_harabaz_score(dataGeo, y_pred2)
data_corr2 = CovSimilarity(dataFixed2)

data_corr2
# Visualization of similarity calculation results

import seaborn as sns

sns.clustermap(data=data_corr2, annot=True, cmap='Purples').savefig('my_1.pdf', bbox_inches='tight')

sns.clustermap(data=CovSimilarity(data_train), annot=True, cmap='Reds').savefig('my_1.pdf', bbox_inches='tight')
# Build up a dataset

missing_precipitation = dataFixed



# The complete data part of Precipitation as training sets and missing data part of Precipitation as test sets

missing_precipitation_train = missing_precipitation[data['Precipitation'].notnull()]

missing_precipitation_test = missing_precipitation[data['Precipitation'].isnull()]



# Constructing the X and Y values of the training set and prediction set respectively

X_train = missing_precipitation_train.drop(['Precipitation'], axis=1)

Y_train = missing_precipitation_train['Precipitation']

X_test = missing_precipitation_test.drop(['Precipitation'], axis=1)

Y_test = missing_precipitation_test['Precipitation']





# Standardize data

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()



# Trainning the test sets and standardize it

ss.fit(X_train)

X_train = ss.transform(X_train)

X_test = ss.transform(X_test)



# Bayesian

from sklearn import linear_model

lin = linear_model.BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,

        fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,

        normalize=False, tol=0.001, verbose=False)

lin.fit(X_train,Y_train)





dataFixed3 = dataFixed

dataFixed3.loc[(data['Precipitation'].isnull()), 'Precipitation'] = lin.predict(X_test)



dataFixed3.describe()
data_corr3 = CovSimilarity(dataFixed3)

data_corr3
b = plt.subplots(figsize=(15,9))

b = sns.heatmap(data_corr3, vmin=-1, vmax=1 , annot=True , square=True)
from sklearn.model_selection import train_test_split # Split data module

from sklearn.neighbors import KNeighborsClassifier # kNN，k-NearestNeighbor



# k-NN modelling

knn = KNeighborsClassifier()



# Training model

knn.fit(X_train, Y_train.astype('int'))



dataFixed4 = dataFixed

dataFixed4.loc[(data['Precipitation'].isnull()), 'Precipitation'] = knn.predict(X_test)

dataFixed4.describe()

data_corr4 = CovSimilarity(dataFixed4)

data_corr4
import seaborn as sns

sns.clustermap(data=data_corr4, annot=True, cmap='Reds').savefig('my_1.pdf', bbox_inches='tight')
# SVR

svr = svm.SVR()

svr.fit(X_train, Y_train)



dataFixed5 = dataFixed

dataFixed5.loc[(data['Precipitation'].isnull()), 'Precipitation'] = svr.predict(X_test)

dataFixed5.describe()
data_corr5 = CovSimilarity(dataFixed5)

data_corr5
import seaborn as sns

sns.clustermap(data=data_corr5, annot=True, cmap='Oranges').savefig('my_1.pdf', bbox_inches='tight')
lr = LogisticRegression()

lr.fit(X_train, Y_train.astype('int'))



dataFixed6 = dataFixed

dataFixed6.loc[(data['Precipitation'].isnull()), 'Precipitation'] = lr.predict(X_test)

dataFixed6.describe()
data_corr6 = CovSimilarity(dataFixed6)

data_corr6
import seaborn as sns

sns.clustermap(data=data_corr6, annot=True, cmap='Greys').savefig('my_1.pdf', bbox_inches='tight')
dtc = DecisionTreeClassifier()

dtc.fit(X_train, Y_train.astype('int'))



dataFixed7 = dataFixed

dataFixed7.loc[(data['Precipitation'].isnull()), 'Precipitation'] = dtc.predict(X_test)

dataFixed7.describe()
data_corr7 = CovSimilarity(dataFixed7)

data_corr7
import seaborn as sns

sns.clustermap(data=data_corr7, annot=True, cmap='Blues').savefig('my_1.pdf', bbox_inches='tight')
lda = LinearDiscriminantAnalysis()

lda.fit(X_train, Y_train.astype('int'))



dataFixed8 = dataFixed

dataFixed8.loc[(data['Precipitation'].isnull()), 'Precipitation'] = lda.predict(X_test)

dataFixed8.describe()
data_corr8 = CovSimilarity(dataFixed8)

data_corr8
import seaborn as sns

sns.clustermap(data=data_corr8, annot=True, cmap='Reds').savefig('my_1.pdf', bbox_inches='tight')
svc = SVC()

svc.fit(X_train, Y_train.astype('int'))



dataFixed9 = dataFixed

dataFixed9.loc[(data['Precipitation'].isnull()), 'Precipitation'] = svc.predict(X_test)

dataFixed9.describe()
data_corr9 = CovSimilarity(dataFixed9)

data_corr9
import seaborn as sns

sns.clustermap(data=data_corr9, annot=True, cmap='Purples').savefig('my_1.pdf', bbox_inches='tight')
gnb = GaussianNB()

gnb.fit(X_train, Y_train.astype('int'))



dataFixed10 = dataFixed

dataFixed10.loc[(data['Precipitation'].isnull()), 'Precipitation'] = gnb.predict(X_test)

dataFixed10.describe()
data_corr10 = CovSimilarity(dataFixed10)

data_corr10
import seaborn as sns

sns.clustermap(data=data_corr10, annot=True, cmap='Oranges').savefig('my_1.pdf', bbox_inches='tight')
# Draw the global scatter diagram of the results of k-means method filling

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='whitegrid', context='notebook')

cols = ['AQI','Precipitation','GDP','Temperature','Longititute','Latitude','Altitude','PopulationDensity','Coastal','GreenCoverageRate','Incineration(10,000ton)']

sns.pairplot(dataFixed[cols], size=2.5)

plt.tight_layout()

# plt.savefig('./figures/scatter.png', dpi=300)

plt.show()
# Draw the global scatter diagram of the results of Decision Tree Classify method filling

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='whitegrid', context='notebook')

cols = ['AQI','Precipitation','GDP','Temperature','Longititute','Latitude','Altitude','PopulationDensity','Coastal','GreenCoverageRate','Incineration(10,000ton)']

sns.pairplot(dataFixed7, size=2.5)

plt.tight_layout()

# plt.savefig('./figures/scatter.png', dpi=300)

plt.show()
# Draw the global scatter diagram of the results of Gaussian Naive Bayes (GNB) method filling

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='whitegrid', context='notebook')

cols = ['AQI','Precipitation','GDP','Temperature','Longititute','Latitude','Altitude','PopulationDensity','Coastal','GreenCoverageRate','Incineration(10,000ton)']

sns.pairplot(dataFixed10, size=2.5)

plt.tight_layout()

# plt.savefig('./figures/scatter.png', dpi=300)

plt.show()
data1 = dataFixed7[['AQI', 'Precipitation']]

data1 = data1.sort_values(by = 'Precipitation', ascending = 'False')

plt.subplots(figsize=(15,9))

x = data1.Precipitation

y = data1.AQI

plt.plot(x, y, marker='o', mec='y', mfc='y')

plt.xlabel('Precipitation') 

plt.ylabel("AQI") 

plt.title("Relations between Precipitation and AQI") 

plt.show()
data2 = dataFixed7[['AQI', 'GDP']]

data2 = data.sort_values(by = 'GDP', ascending = 'False')

plt.subplots(figsize=(30,9))

x = data2.GDP

#x = Latitude_normed

#normalized_x = preprocessing.normalize(x).fit(x).values.reshape(-1,1)

y = data2.AQI

#y = AQI_normed

#normalized_y = preprocessing.normalize(y).fit(y).values.reshape(-1,1)

plt.plot(x, y, marker='o', mec='g', mfc='y')

plt.xlabel('GDP') 

plt.ylabel("AQI") 

plt.title("Relations between GDP and AQI") 

plt.show()
data3 = dataFixed7[['AQI', 'Temperature']]

data3 = data3.sort_values(by = 'Temperature', ascending = 'False')

plt.subplots(figsize=(30,10))

x = data3.Temperature

y = data3.AQI

plt.plot(x, y, marker='o', mec='r', mfc='y')

plt.xlabel('Temperature') 

plt.ylabel("AQI") 

plt.title("Relations between Temperature and AQI") 

plt.show()
# 3D diagram of longitude, latitude and AQI

from mpl_toolkits.mplot3d import Axes3D

data4 = dataFixed7[['AQI', 'Longititute', 'Latitude']]

plt.subplots(figsize=(20,10))

x, y, z = data4['Longititute'], data4['Latitude'], dataGeo['AQI']

ax = plt.subplot(111, projection='3d')

ax.scatter(x, y, z, linewidths = 4)  

ax.set_zlabel('AQI') 

ax.set_ylabel('Latitude')

ax.set_xlabel('Longititute')

plt.title('Relations between Longititute, Latitude and AQI')

plt.show()
# AQI and Latitude

data5 = dataFixed7[['AQI', 'Latitude']]

data5 = data5.sort_values(by = 'Latitude', ascending = 'False')

plt.subplots(figsize=(20,10))

x = data5.Latitude

y = data5.AQI

plt.plot(x, y, marker='o', mec='r', mfc='y')

plt.xlabel('Latitude') 

plt.ylabel("AQI") 

plt.title("Relations between Latitude and AQI") 

plt.show()
# AQI and Longititute

data6 = dataFixed7[['AQI', 'Longititute']]

data6 = data6.sort_values(by = 'Longititute', ascending = 'False')

plt.subplots(figsize=(30,10))

x = data6.Longititute

y = data6.AQI

plt.plot(x, y, marker='o', mec='c', mfc='y')

plt.xlabel('Longititute') 

plt.ylabel("AQI") 

plt.title("Relations between Longititute and AQI") 

plt.show()
# AQI and Altitude

data7 = dataFixed7[['AQI', 'Altitude']]

data7 = data7.sort_values(by = 'Altitude', ascending = 'False')

plt.subplots(figsize=(30,10))

x = data7.Altitude

y = data7.AQI

plt.plot(x, y, marker='o', mec='r', mfc='y')

plt.xlabel('Altitude') 

plt.ylabel("AQI") 

plt.title("Relations between Altitude and AQI") 

plt.show()
# AQI and PopulationDensity

data8 = dataFixed7[['AQI', 'PopulationDensity']]

data8 = data8.sort_values(by = 'PopulationDensity', ascending = 'False')

plt.subplots(figsize=(30,10))

x = data8.PopulationDensity

y = data8.AQI

plt.plot(x, y, marker='o', mec='g', mfc='y')

plt.xlabel('PopulationDensity') 

plt.ylabel("AQI") 

plt.title("Relations between PopulationDensity and AQI") 

plt.show()
aqi_Coastal = 0

aqi_no_Coastal = 0

c_Coastal = 0

c_no_Coastal = 0

for i in range(len(dataFixed7)):

    if dataFixed7.Coastal.loc[i] == 0:

        aqi_no_Coastal += dataFixed7.AQI.loc[i]

        c_no_Coastal += 1

    else:

        aqi_Coastal += dataFixed7.AQI.loc[i]

        c_Coastal += 1



mean_aqi_Coastal = aqi_Coastal/ (c_Coastal * 1.0 )

mean_aqi_no_Coastal = aqi_no_Coastal/ (c_no_Coastal * 1.0 )

name_list2 = ['Coastal', 'Non Coastal']

num_list2 = [mean_aqi_Coastal, mean_aqi_no_Coastal]

plt.figure(figsize = (8,8))

rects=plt.bar(range(len(num_list2)), num_list2, color='y')

index=[0,1]

index=[float(c) for c in index]

plt.xticks(index, name_list2)

plt.ylabel("Mean AQI Value")

plt.title('Relations between Coastal and AQI')

#plt.ylim((0, 0.5))

for i, rect in enumerate(rects):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height,  '%.4f' %num_list2[i], ha='center', va='bottom')

plt.show()
data9 = dataFixed7[['AQI', 'GreenCoverageRate']]

data9 = data9.sort_values(by = 'GreenCoverageRate', ascending = 'False')

plt.subplots(figsize=(30,10))

x = data9.GreenCoverageRate

y = data9.AQI

plt.plot(x, y, marker='o', mec='b', mfc='y')

plt.xlabel('GreenCoverageRate') 

plt.ylabel("AQI") 

plt.title("Relations between GreenCoverageRate and AQI") 

plt.show()
import numpy as np 

import matplotlib.pyplot as plt

x = dataFixed7[['Precipitation']].values

X = x.reshape(-1, 1)

y = dataFixed7['AQI'].values

plt.scatter(x, y)

plt.show()
# To compile the least square method classes

class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):

        self.eta = eta

        self.n_iter = n_iter

    def fit(self, X, y):   # X is a column vector,y is a row vector

        self.w_ = np.zeros(1 + X.shape[1])   #Initialize (1,2) row vectors which are all 0 and store the two coefficients of the line fitting by the iterative process

        self.cost_ = []

        for i in range(self.n_iter):

            output = self.net_input(X)

            errors = (y - output)   # The errors are the error entries for row vectors that have the same dimension as y

            self.w_[1:] += self.eta * X.T.dot(errors)   # Fitting the primary coefficient of the line

            self.w_[0] += self.eta * errors.sum()   # Fitting constant term of line

            cost = (errors**2).sum() / 2.0   # The sum of the squares of the residuals and half the objective function

            self.cost_.append(cost)

        return self

    def net_input(self, X):

        return np.dot(X, self.w_[1:]) + self.w_[0]   

    def predict(self, X):

        return self.net_input(X)

# Cost_ is a statistical list of the squares and halves of residuals for each iteration,

# w_ contains two parameters of the line for each iteration, and errors are residuals for each iteration



X = dataFixed7[['Temperature']].values   #X is (*,1) dimensional column vector

y = dataFixed7['AQI'].values   #y is (*, ) row vector

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)   

# The fit_transform method can be divided into fit and transform steps in StanderdScalar, which will be merged in order to differentiate the LinearRegressionGD class

y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()   

#y[:, np.newaxis] equal to y[np.newaxis].T,that is, df[['MEDV']].values；flatten method is used to change back to 1*n vectors

#fit_transform method is in order to regularize the “column vector” directly

lr = LinearRegressionGD()

lr.fit(X_std, y_std)   # This fit is the class of LinearRegressionGD, Note the difference between the different fit methods used in sklearn and their environments

#Output:<__main__.LinearRegressionGD at 0x16add278>

plt.plot(range(1, lr.n_iter+1), lr.cost_)

plt.ylabel('Temperature')

plt.xlabel('AQI')

plt.tight_layout()

# plt.savefig('./figures/cost.png', dpi=300)

plt.show()
def lin_regplot(X, y, model):

    plt.scatter(X, y, c='lightblue')

    plt.plot(X, model.predict(X), color='red', linewidth=2)    

    return 

lin_regplot(X_std, y_std, lr)

plt.xlabel('Temperature')

plt.ylabel('AQI')

plt.tight_layout()

# plt.savefig('./figures/gradient_fit.png', dpi=300)

plt.show()
import numpy

from sklearn.tree import DecisionTreeRegressor

#http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor

X = data[['Altitude']].values

y = data['AQI'].values

 

tree = DecisionTreeRegressor(max_depth=5)   #max_depth Setting the depth of tree

tree.fit(X, y)   # Various attributes obtained after modeling: tree, features used and importance of features

 

sort_idx = X.flatten().argsort()   #The vectors constructed by the index of the smallest element to the largest element in X

 

lin_regplot(X[sort_idx], y[sort_idx], tree)

plt.xlabel('Altitude')

plt.ylabel('AQI')

# plt.savefig('./figures/tree_regression.png', dpi=300)

plt.show()

#The horizontal red line represents the c value, and the vertical red line represents the shard point selected by the feature column
X = data[['PopulationDensity']].values

y = data['AQI'].values

 

tree = DecisionTreeRegressor(max_depth=8)   #max_depth Setting the depth of tree

tree.fit(X, y)   # Various attributes obtained after modeling: tree, features used and importance of features

 

sort_idx = X.flatten().argsort()   #The vectors constructed by the index of the smallest element to the largest element in X

 

lin_regplot(X[sort_idx], y[sort_idx], tree)

plt.xlabel('PopulationDensity')

plt.ylabel('AQI')

# plt.savefig('./figures/tree_regression.png', dpi=300)

plt.show()

#The horizontal red line represents the c value, and the vertical red line represents the shard point selected by the feature column
X = data[['Incineration(10,000ton)']].values

y = data['AQI'].values

 

tree = DecisionTreeRegressor(max_depth=8)   #max_depth Setting the depth of tree

tree.fit(X, y)   # Various attributes obtained after modeling: tree, features used and importance of features

 

sort_idx = X.flatten().argsort()   #The vectors constructed by the index of the smallest element to the largest element in X

 

lin_regplot(X[sort_idx], y[sort_idx], tree)

plt.xlabel('Incineration(10,000ton)')

plt.ylabel('AQI')

# plt.savefig('./figures/tree_regression.png', dpi=300)

plt.show()

#The horizontal red line represents the c value, and the vertical red line represents the shard point selected by the feature column
X = data[['GreenCoverageRate']].values

y = data['AQI'].values

 

tree = DecisionTreeRegressor(max_depth=8)   #max_depth Setting the depth of tree

tree.fit(X, y)   # Various attributes obtained after modeling: tree, features used and importance of features

 

sort_idx = X.flatten().argsort()   #The vectors constructed by the index of the smallest element to the largest element in X

 

lin_regplot(X[sort_idx], y[sort_idx], tree)

plt.xlabel('GreenCoverageRate')

plt.ylabel('AQI')

# plt.savefig('./figures/tree_regression.png', dpi=300)

plt.show()

#The horizontal red line represents the c value, and the vertical red line represents the shard point selected by the feature column