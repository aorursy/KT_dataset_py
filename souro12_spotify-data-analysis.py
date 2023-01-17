# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

import squarify as sq

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

import seaborn as sns

import sklearn

import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
filename = '/kaggle/input/top50spotify2019/top50.csv'

df = pd.read_csv(filename, encoding='ISO-8859-1')

df.head()
print(df.shape)
df.rename(columns={'Track.Name':'track_name','Artist.Name':'artist_name','Beats.Per.Minute':'beats_per_minute','Loudness..dB..':'Loudness(dB)','Valence.':'Valence','Length.':'Length', 'Acousticness..':'Acousticness','Speechiness.':'Speechiness'},inplace=True)

df.head()
df.isnull().sum()

df.fillna(0)
print(df.dtypes)
print(type(df['Genre']))

popular_genre = df.groupby('Genre').size()

print(popular_genre)

genre_list = df['Genre'].values.tolist()
print(df.groupby('artist_name').size())

popular_artist=df.groupby('artist_name').size()

print(popular_artist)

artist_list=df['artist_name'].values.tolist()
df.isnull().sum()

df.fillna(0)
pd.set_option('precision', 3 )

df.describe()
skew = df.skew()

print(skew)

transform = np.asarray(df[['Liveness']].values)

df.transform = stats.boxcox(transform)[0]



plt.hist(df['Liveness'], bins = 10)

plt.show()



plt.hist(df.transform, bins = 10)

plt.show()
transform1=np.asarray(df[['Popularity']].values)

df_transform1 = stats.boxcox(transform1)[0]

sns.distplot(df['Popularity'],bins=10,kde=True,kde_kws={"color": "k", "lw": 2, "label": "KDE"},color='yellow')

plt.show()

sns.distplot(df_transform1,bins=10,kde=True,kde_kws={"color": "k", "lw": 2, "label": "KDE"},color='black') #corrected skew data

plt.show()
pd.set_option('display.width', 100)

pd.set_option('precision', 3)

correlation=df.corr(method='spearman')

print(correlation)
# Bar graph to see the number of songs of each genre

fig, ax=plt.subplots(figsize=(30,12))

length=np.arange(len(popular_genre))

plt.bar(length,popular_genre,color='green',edgecolor='black',alpha=0.7)

plt.xticks(length,genre_list)

plt.title('Most popular genre',fontsize=18)

plt.xlabel('Genre',fontsize=16)

plt.ylabel('Number of songs',fontsize=16)

plt.show()
# heatmap of the correlation 

plt.figure(figsize=(10,10))

plt.title('Correlation heatmap')

sns.heatmap(correlation,annot=True,vmin=-1,vmax=1,cmap="GnBu_r",center=1)
fig, ax=plt.subplots(figsize=(12,12))

length=np.arange(len(popular_artist))

plt.barh(length,popular_artist,color='red',edgecolor='black',alpha=0.7)

plt.yticks(length,artist_list)

plt.title('Most popular artists',fontsize=18)

plt.ylabel('Artists',fontsize=16)

plt.xlabel('Number of songs',fontsize=16)

plt.show()
#Linear regression, first create test and train dataset

x=df.loc[:,['Energy','Danceability','Length','Loudness(dB)','Acousticness']].values

y=df.loc[:,'Popularity'].values
# Creating a test and training dataset

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
# Linear regression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

print(regressor.intercept_)

print(regressor.coef_)
#Displaying the difference between the actual and the predicted

y_pred = regressor.predict(X_test)

df_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print(df_output)
#Checking the accuracy of Linear Regression

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
plt.figure(figsize=(10,10))

plt.plot(y_pred,y_test,color='green',linestyle='dashed',marker='^',markerfacecolor='purple',markersize=5)

plt.title('Error analysis')

plt.xlabel('Predicted values')

plt.ylabel('Test values')
# Cross validation score

x = df.loc[:,['Energy','Danceability']].values

y = df.loc[:,'Popularity'].values

regressor = LinearRegression()

mse = cross_val_score(regressor,X_train,y_train,scoring='neg_mean_squared_error',cv=5)

mse_mean = np.mean(mse)

print (mse_mean)

diff = metrics.mean_squared_error(y_test, y_pred)-abs(mse_mean)

print (diff)
x=df.loc[:,['artist_name']].values

y=df.loc[:,'Genre'].values
x.shape

encoder=LabelEncoder()

x = encoder.fit_transform(x)

x=pd.DataFrame(x)

x
# Label Encoding of target

Encoder_y=LabelEncoder()

Y = Encoder_y.fit_transform(y)

Y=pd.DataFrame(Y)

Y
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)



#Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(x_train)

x_train=sc.transform(x_train)

x_test=sc.transform(x_test)
# KNN Classification

# sorted(sklearn.neighbors.VALID_METRICS['brute'])

knn = KNeighborsClassifier(n_neighbors = 17)

knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)
error=[]

for i in range(1,30):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i=knn.predict(X_test)

    error.append(np.mean(pred_i!=y_test))
plt.figure(figsize=(10,10))

plt.plot(range(1,30),error,color='yellow',marker='o',markerfacecolor='red',markersize=10)

plt.title('Error Rate K value')

plt.xlabel('K Value')

plt.ylabel('Mean error')
x=df.loc[:,['Energy','Length','Danceability','beats_per_minute', 'Acousticness']].values

y=df.loc[:,'Popularity'].values



# Creating a test and training dataset

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred=gnb.predict(X_test)

df_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print(df_output)
# Testing the accuracy of Naive Bayes 

scores=cross_val_score(gnb,X_train,y_train,scoring='accuracy',cv=3).mean()*100

print(scores)
sns.jointplot(x=y_test, y=y_pred, kind="kde", color="r");