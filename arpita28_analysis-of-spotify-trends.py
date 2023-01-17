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

from sklearn.preprocessing import MinMaxScaler,LabelEncoder

from sklearn.model_selection import train_test_split,cross_val_score, KFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB

from sklearn.svm import LinearSVC, SVC

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
filename='/kaggle/input/top50spotify2019/top50.csv'

df=pd.read_csv(filename,encoding='ISO-8859-1')

df.head()
#Calculates the number of rows and columns

print(df.shape)
#Renaming the columns

df.rename(columns={'Track.Name':'track_name','Artist.Name':'artist_name','Beats.Per.Minute':'beats_per_minute','Loudness..dB..':'Loudness(dB)','Valence.':'Valence','Length.':'Length', 'Acousticness..':'Acousticness','Speechiness.':'Speechiness'},inplace=True)

df.head()
df.isnull().sum()

df.fillna(0)
# The datatypes of the different attributes of the dataset

print(df.dtypes)
#Calculating the number of songs of each genre

print(type(df['Genre']))

popular_genre=df.groupby('Genre').size().unique

print(popular_genre)

genre_list=df['Genre'].values.tolist()
#Calculating the number of songs by each of the artists

print(df.groupby('artist_name').size())

popular_artist=df.groupby('artist_name').size()

print(popular_artist)

artist_list=df['artist_name'].values.tolist()
df.isnull().sum()

df.fillna(0)
pd.set_option('precision', 3)

df.describe()
#Finding out the skew for each attribute

skew=df.skew()

print(skew)

# Removing the skew by using the boxcox transformations

transform=np.asarray(df[['Liveness']].values)

df_transform = stats.boxcox(transform)[0]

# Plotting a histogram to show the difference 

plt.hist(df['Liveness'],bins=10) #original data

plt.show()

plt.hist(df_transform,bins=10) #corrected skew data

plt.show()
transform1=np.asarray(df[['Popularity']].values)

df_transform1 = stats.boxcox(transform1)[0]

# Plotting a histogram to show the difference 

# plt.hist(df['Popularity'],bins=10) original data

# plt.show()

# plt.hist(df_transform1,bins=10) #corrected skew data

# plt.show()

sns.distplot(df['Popularity'],bins=10,kde=True,kde_kws={"color": "k", "lw": 2, "label": "KDE"},color='yellow')

plt.show()

sns.distplot(df_transform1,bins=10,kde=True,kde_kws={"color": "k", "lw": 2, "label": "KDE"},color='black') #corrected skew data

plt.show()
pd.set_option('display.width', 100)

pd.set_option('precision', 3)

correlation=df.corr(method='spearman')

print(correlation)


xtick = ['dance pop', 'pop', 'latin', 'edm', 'canadian hip hop',

'panamanian pop', 'electropop', 'reggaeton flow', 'canadian pop',

'reggaeton', 'dfw rap', 'brostep', 'country rap', 'escape room',

'trap music', 'big room', 'boy band', 'pop house', 'australian pop',

'r&b en espanol', 'atl hip hop']

length = np.arange(len(xtick))

genre_groupby = df.groupby('Genre')['track_name'].agg(len)

plt.figure(figsize = (15,7))

plt.bar(length, genre_groupby)

plt.xticks(length,xtick)

plt.xticks(rotation = 90)

plt.xlabel('Genre', fontsize = 20)

plt.ylabel('Count of the tracks', fontsize = 20)

plt.title('Genre vs Count of the tracks', fontsize = 25)

# heatmap of the correlation 

plt.figure(figsize=(10,10))

plt.title('Correlation heatmap')

sns.heatmap(correlation,annot=True,vmin=-1,vmax=1,cmap="GnBu_r",center=1)
fig = plt.figure(figsize = (15,7))

df.groupby('artist_name')['track_name'].agg(len).sort_values(ascending = False).plot(kind = 'bar')

plt.xlabel('Artist Name', fontsize = 20)

plt.ylabel('Count of songs', fontsize = 20)

plt.title('Artist Name vs Count of songs', fontsize = 30)
# Analysing the relationship between energy and loudness

fig=plt.subplots(figsize=(10,10))

sns.regplot(x='Energy',y='Loudness(dB)',data=df,color='black')
fig=plt.subplots(figsize=(10,10))

plt.title('Dependence between energy and popularity')

sns.regplot(x='Energy', y='Popularity',

            ci=None, data=df)

sns.kdeplot(df.Energy,df.Popularity)
scatter_matrix(df)

plt.gcf().set_size_inches(30, 30)

plt.show()
df.plot(kind='box', subplots=True)

plt.gcf().set_size_inches(30,30)

plt.show()
plt.figure(figsize=(14,8))

sq.plot(sizes=df.Genre.value_counts(), label=df["Genre"].unique(), alpha=.8 )

plt.axis('off')

plt.show()
#Pie charts 

labels = df.artist_name.value_counts().index

sizes = df.artist_name.value_counts().values

colors = ['red', 'yellowgreen', 'lightcoral', 'lightskyblue','cyan', 'green', 'black','yellow']

plt.figure(figsize = (10,10))

plt.pie(sizes, labels=labels, colors=colors)

autopct=('%1.1f%%')

plt.axis('equal')

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

plt.plot(y_pred,y_test,color='black',linestyle='dashed',marker='*',markerfacecolor='red',markersize=10)

plt.title('Error analysis')

plt.xlabel('Predicted values')

plt.ylabel('Test values')
# Cross validation score

x=df.loc[:,['Energy','Danceability']].values

y=df.loc[:,'Popularity'].values

regressor=LinearRegression()

mse=cross_val_score(regressor,X_train,y_train,scoring='neg_mean_squared_error',cv=5)

mse_mean=np.mean(mse)

print(mse_mean)

diff=metrics.mean_squared_error(y_test, y_pred)-abs(mse_mean)

print(diff)
x=df.loc[:,['artist_name']].values

y=df.loc[:,'Genre'].values
# Label encoding of features

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

plt.plot(range(1,30),error,color='black',marker='o',markerfacecolor='cyan',markersize=10)

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
sns.jointplot(x=y_test, y=y_pred, kind="kde", color="r")
x=df.loc[:,['Energy','Length','Danceability','beats_per_minute', 'Acousticness']].values

y=df.loc[:,'Popularity'].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
# Linear SVM model 

LinSVC = LinearSVC(penalty='l2', loss='squared_hinge', dual=True)

LinSVC.fit(X_train, y_train)

y_pred=gnb.predict(X_test)

df_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print(df_output)
# Testing the accuracy

scores=cross_val_score(LinSVC,X_train,y_train,scoring='accuracy',cv=3).mean()*100

print(scores)
sns.jointplot(x=y_test, y=y_pred, kind="reg", color="b");