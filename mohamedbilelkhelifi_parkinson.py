import numpy as np  #numpy

import pandas as pd  #pandas

import matplotlib.pyplot as plt # plotting

import os # accessing directory structure

from IPython.display import Image

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

!ls ../input

dataframe = pd.read_csv('../input/parkinsons.csv')

dataframe.head()
dataframe.columns
dataframe.describe()
df=dataframe.reindex(columns =(['name', 'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',

       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',

       'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',

       'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',

       'spread1', 'spread2', 'D2', 'PPE', 'status']))
df.isnull().sum()
df.isna().sum()
df.nunique()
df.status.unique()
df['MDVP:Jitter(Abs)'].unique()
print ("Nombre des colonnes = " , df.columns.size)


df.drop(['name'], axis=1, inplace=True)
print ("Nombre des colonnes = " , df.columns.size)
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 20]] # For displaying purposes, pick columns that have between 1 and 20 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
plotPerColumnDistribution(df, 25, 5)
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    #filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title('Matrice de Correlation ', fontsize=15)

    plt.show()
plotCorrelationMatrix(df, 8)
# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Diagramme de dispersion et de densité')

    plt.show()
plotScatterMatrix(df, 20, 10)
import matplotlib.pyplot as plt # side-stepping mpl backend

df['status'].value_counts().plot(kind='bar')

print((df['status'].value_counts()))

plt.suptitle('Status')

plt.show()

from sklearn.cluster import KMeans

from sklearn.metrics.cluster import silhouette_score

from sklearn.preprocessing import scale, robust_scale

x = df.iloc[:,:23].values



print ("data,feature",x.shape)

wcss = []

for i in range(1, 15):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)

    print (i , kmeans.inertia_)

    

#Plotting the results onto a line graph, allowing us to observe 'The elbow'

plt.plot(range(1, 15), wcss,'bx-')

plt.title('Methode elbow avec inertia')

plt.xlabel('Nombre des groupes')

plt.ylabel('WCSS') #within cluster sum of squares

plt.show()
df[:1]
x[0]

#kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
pred_y = kmeans.fit_predict(x)

plt.scatter(x[:,0], x[:,1])

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')

plt.show()
df['Cluster'] = KMeans(n_clusters=7).fit_predict(x) + 1
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.3,random_state=1)
print(train.shape)

print(test.shape)
#making independent variables for training

train_X = train.iloc[:, 1:25]

#making responsible variables for training

train_y=train.status

#making independent variables for testing

test_X= test.iloc[:, 1:25]

#making responsible variables for testing

test_y =test.status
print(train_X.shape)

print(train_y.shape)

print(test_X.shape)

print(test_y.shape)
print("train x \n",train_X)

print("train y \n ",train_y)

print("test x\n",test_X)

print("test y \n",test_y)
#Without Hyper Parameters Tuning

#1-1,DesicionTree

#importing module

from sklearn.tree import DecisionTreeClassifier

#making the instance

model= DecisionTreeClassifier(random_state=1)

#learning

model.fit(train_X,train_y)

#Prediction

prediction=model.predict(test_X)

#importing the metrics module

from sklearn import metrics

#evaluation(Accuracy)

print("Accuracy:",metrics.accuracy_score(prediction,test_y))

#evaluation(Confusion Metrix)

print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,test_y))