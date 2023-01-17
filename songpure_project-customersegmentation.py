import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns 

import plotly as py

import plotly.graph_objs as go

from sklearn.cluster import KMeans

import warnings

import os

warnings.filterwarnings("ignore")

py.offline.init_notebook_mode(connected = True)

%config InlineBacked.figure_format = 'svg'
df = pd.read_csv(r'../input/CutomerInfoDataset.csv')
df.head()
df.shape
df.describe()
df.dtypes
df.isnull().sum()
plt.style.use('fivethirtyeight')
plt.figure(1 , figsize = (10 , 6))

sns.countplot(y = 'Aging' , data = df)

plt.show()
plt.figure(1 , figsize = (10 , 6))

for aging in [0, 31, 61, 91, 121]:

    plt.scatter(x = 'Aging' , y = 'RegisCap' , data = df[df['Aging'] == aging] ,

                s = 200 , alpha = 0.5 , label = aging)

plt.xlabel('Aging'), plt.ylabel('RegisCap') 

plt.title('Aging vs RegisCap')

plt.legend()

plt.show()
plt.figure(1 , figsize = (13 , 7))

n = 0 

for x in ['RegisCap', 'NOOrder/M', 'PriceOrder/M']:

    for y in ['RegisCap', 'NOOrder/M', 'PriceOrder/M']:

        n += 1

        plt.subplot(6 , 3 , n)

        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

        sns.regplot(x = x , y = y , data = df)

        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y)

plt.show()
plt.figure(1 , figsize = (10 , 6))

for aging in [0, 31, 61, 91, 121]:

    plt.scatter(x = 'NOOrder/M' , y = 'RegisCap' , data = df[df['Aging'] == aging] ,

                s = 200 , alpha = 0.5 , label = aging)

plt.xlabel('NOOrder/M'), plt.ylabel('RegisCap') 

plt.title('NOOrder/M vs RegisCap')

plt.legend()

plt.show()
plt.figure(1 , figsize = (10, 6))

for aging in [0, 31, 61, 91, 121]:

    plt.scatter(x = 'PriceOrder/M' , y = 'RegisCap' , data = df[df['Aging'] == aging] ,

                s = 200 , alpha = 0.5 , label = aging)

plt.xlabel('PriceOrder/M'), plt.ylabel('RegisCap') 

plt.title('PriceOrder/M vs RegisCap')

plt.legend()

plt.show()
plt.figure(1 , figsize = (10, 6))

for aging in [0, 31, 61, 91, 121]:

    plt.scatter(x = 'NOOrder/M' , y = 'Employee' , data = df[df['Aging'] == aging] ,

                s = 200 , alpha = 0.5 , label = aging)

plt.xlabel('NOOrder/M'), plt.ylabel('Employee') 

plt.title('NOOrder/M vs Employee')

plt.legend()

plt.show()
plt.figure(1 , figsize = (10, 6))

for focusType in [1, 2, 3]:

    plt.scatter(x = 'PriceOrder/M' , y = 'Employee' , data = df[df['FocusType'] == focusType] ,

                s = 200 , alpha = 0.5 , label = focusType)

plt.xlabel('PriceOrder/M'), plt.ylabel('Employee') 

plt.title('PriceOrder/M vs Employee')

plt.legend()

plt.show()
plt.figure(1 , figsize = (10, 6))

for focusType in [1, 2, 3]:

    plt.scatter(x = 'NOOrder/M' , y = 'PriceOrder/M' , data = df[df['FocusType'] == focusType] ,

                s = 200 , alpha = 0.5 , label = focusType)

plt.xlabel('NOOrder/M'), plt.ylabel('PriceOrder/M') 

plt.title('NOOrder/M vs PriceOrder/M')

plt.legend()

plt.show()
from sklearn.preprocessing import StandardScaler

# standardize the features

sc = StandardScaler()

dfs = sc.fit_transform(df)
col = ["NOOrder/M", "PriceOrder/M", "RegisCap", "Employee", "Aging", "CreditLimit"]

dfk = df.copy()

dfk[col] = sc.fit_transform(dfk[col])
'''PriceOrder/M and RegisCap'''

# X1 = df[['AVGNOOrderPerMonth' , 'RegisteredCapital']].iloc[: , :].values

X1 = dfs[:, [2, 3]]

inertia = []

for n in range(1 , 11):

    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

    algorithm.fit(X1)

    inertia.append(algorithm.inertia_)

    

algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(X1)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_



h = 0.02

x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1

y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 



plt.figure(1 , figsize = (10 , 6) )

plt.clf()

Z = Z.reshape(xx.shape)

plt.imshow(Z , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')



plt.scatter( x = 'PriceOrder/M' ,y = 'RegisCap' , data = dfk , c = labels1 , 

            s = 200 )

plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.ylabel('RegisCap') , plt.xlabel('PriceOrder/M')

plt.show()
'''NOOrder/M and RegisCap'''

# X1 = df[['AVGNOOrderPerMonth' , 'RegisteredCapital']].iloc[: , :].values

X1 = dfs[:, [1, 3]]

inertia = []

for n in range(1 , 11):

    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

    algorithm.fit(X1)

    inertia.append(algorithm.inertia_)

    

algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(X1)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_



h = 0.02

x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1

y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 



plt.figure(1 , figsize = (10 , 6) )

plt.clf()

Z = Z.reshape(xx.shape)

plt.imshow(Z , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')



plt.scatter( x = 'NOOrder/M' ,y = 'RegisCap' , data = dfk , c = labels1 , 

            s = 200 )

plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.ylabel('RegisCap') , plt.xlabel('NOOrder/M')

plt.show()
'''PriceOrder/M and Employee'''

X1 = dfs[:, [2, 4]]

inertia = []

for n in range(1 , 11):

    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

    algorithm.fit(X1)

    inertia.append(algorithm.inertia_)

    

algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(X1)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_



h = 0.02

x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1

y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 



plt.figure(1 , figsize = (10 , 6) )

plt.clf()

Z = Z.reshape(xx.shape)

plt.imshow(Z , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')



plt.scatter( x = 'PriceOrder/M' ,y = 'Employee' , data = dfk , c = labels1 , 

            s = 200 )

plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.ylabel('Employee') , plt.xlabel('PriceOrder/M')

plt.show()
'''NOOrder/M and PriceOrder/M'''

X1 = dfs[:, [1, 2]]

inertia = []

for n in range(1 , 11):

    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

    algorithm.fit(X1)

    inertia.append(algorithm.inertia_)

    

algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(X1)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_



h = 0.02

x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1

y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 



plt.figure(1 , figsize = (10 , 6) )

plt.clf()

Z = Z.reshape(xx.shape)

plt.imshow(Z , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')



plt.scatter( x = 'NOOrder/M' ,y = 'PriceOrder/M' , data = dfk , c = labels1 , 

            s = 200 )

plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.ylabel('PriceOrder/M') , plt.xlabel('NOOrder/M')

plt.show()
X3 = df[['NOOrder/M' , 'PriceOrder/M' ,'Employee']].iloc[: , :].values

inertia = []

for n in range(1 , 11):

    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

    algorithm.fit(X3)

    inertia.append(algorithm.inertia_)
algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(X3)

labels3 = algorithm.labels_

centroids3 = algorithm.cluster_centers_
df3d = df.copy()
df3d['label3'] =  labels3

trace1 = go.Scatter3d(

    x= df3d['NOOrder/M'],

    y= df3d['PriceOrder/M'],

    z= df3d['Employee'],

    mode='markers',

     marker=dict(

        color = df3d['label3'], 

        size= 20,

        line=dict(

            color= df3d['label3'],

            width= 12

        ),

        opacity=0.8

     )

)

data = [trace1]

layout = go.Layout(

    title= 'Clusters',

    scene = dict(

            xaxis = dict(title  = 'NOOrder/M'),

            yaxis = dict(title  = 'PriceOrder/M'),

            zaxis = dict(title  = 'Employee')

        )

)

fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)
df['FocusType'].value_counts()
df.isnull().any()
numeric_cols = list(df.columns) 
plt.figure(figsize=(10,5))

sns.heatmap(df.corr(), annot = True, cmap='coolwarm',linewidths=.1)

plt.show()
#Draw boxplots for all numeric columns

fig, axes = plt.subplots(figsize=(18, 10), nrows=3, ncols=3, squeeze=0)

i=0

for ax, col in zip(axes.reshape(-1), numeric_cols):

  ax.boxplot(df[col], labels=[col], sym='k.')



plt.tight_layout()
outliers=[]

def detect_outlier(data_1):

    

    threshold=3

    mean_1 = np.mean(data_1)

    std_1 =np.std(data_1)

    

    index = 0

    

    for y in data_1:

        z_score= (y - mean_1)/std_1 

        if np.abs(z_score) > threshold:

            outliers.append(index)

        

        index = index + 1

    return outliers
outlier_z_score = detect_outlier(df['NOOrder/M'])

df.drop(outlier_z_score, inplace=True)

print(outlier_z_score)
# df.reset_index(inplace=True)

# outlier_z_score = detect_outlier(df['CreditLimit'])

# df.drop(outlier_z_score, inplace=True)

# print(outlier_z_score)
# df.reset_index(inplace=True)

# outlier_z_score = detect_outlier(df['RegisCap'])

# df.drop(outlier_z_score, inplace=True)

# print(outlier_z_score)
# df.reset_index(inplace=True)

# outlier_z_score = detect_outlier(df['Employee'])

# df.drop(outlier_z_score, inplace=True)

# print(outlier_z_score)
df.shape
df.head()
col = ["NOOrder/M", "PriceOrder/M", "RegisCap", "Employee", "Aging", "CreditLimit"]

sc = StandardScaler()

df[col] = sc.fit_transform(df[col])
df.head()
X = df.iloc[:,1:-1]

y = df.iloc[:,-1]
X.head()
y.head()
#Label encoding on the 'target' column

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings("ignore")

le = LabelEncoder()

y = le.fit_transform(y)
#Convert array to Series

y = pd.Series(y)

y.value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape
y_train.shape
X_test.shape
y_test.shape
# from sklearn.feature_selection import SelectKBest

# from sklearn.feature_selection import mutual_info_classif

# from sklearn.feature_selection import chi2
# #Select 4 features

# selector = SelectKBest(score_func=mutual_info_classif, k=4)



# #Fit the selector to the training data set.

# selector_model = selector.fit(X_train, y_train)
# #Show selected features

# select_column = X_train.columns[selector_model.get_support(indices=True)]

# select_column
# select_column_fix = ['NOOrder/M', 'PriceOrder/M', 'Aging','CreditLimit']
# X_train = X_train[select_column_fix]
# X_test = X_test[select_column_fix]
# X_train.shape
# X_test.shape
from sklearn.linear_model import LogisticRegression



mymodel_Logistic = LogisticRegression(multi_class='auto', solver='liblinear')

mymodel_Logistic.fit(X_train, y_train)



y_pred = mymodel_Logistic.predict(X_test)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average='weighted')

recall = recall_score(y_test, y_pred, average='weighted')

f1 = f1_score(y_test, y_pred, average='weighted')



print("Performance:")

print(" >accuracy = " + str(accuracy))

print(" >precision = " + str(precision))

print(" >recall = " + str(recall))

print(" >f1 = " + str(f1))
from sklearn.neighbors import KNeighborsClassifier



mymodel_KNeighbors = KNeighborsClassifier(n_neighbors=3)

mymodel_KNeighbors.fit(X_train, y_train)



y_pred = mymodel_KNeighbors.predict(X_test)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average='weighted')

recall = recall_score(y_test, y_pred, average='weighted')

f1 = f1_score(y_test, y_pred, average='weighted')



print("Performance:")

print(" >accuracy = " + str(accuracy))

print(" >precision = " + str(precision))

print(" >recall = " + str(recall))

print(" >f1 = " + str(f1))
from sklearn import svm



mymodel_SVM = svm.SVC(kernel='linear')

mymodel_SVM.fit(X_train, y_train)



y_pred = mymodel_SVM.predict(X_test)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average='weighted')

recall = recall_score(y_test, y_pred, average='weighted')

f1 = f1_score(y_test, y_pred, average='weighted')



print("Performance:")

print(" >accuracy = " + str(accuracy))

print(" >precision = " + str(precision))

print(" >recall = " + str(recall))

print(" >f1 = " + str(f1))