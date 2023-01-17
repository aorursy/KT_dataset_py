# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
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
spotify='/kaggle/input/top50spotify2019/top50.csv'

mtb=pd.read_csv(spotify,encoding='ISO-8859-1')

mtb.head()
mtb.isnull().sum()

mtb.fillna(0)
print(type(mtb['Artist.Name']))

popular_genre=mtb.groupby('Artist.Name').size().unique

print(popular_genre)

genre_list=mtb['Artist.Name'].values.tolist()
print(type(mtb['Track.Name']))

popular_genre=mtb.groupby('Track.Name').size().unique

print(popular_genre) 

genre_list=mtb['Track.Name'].values.tolist()
print(type(mtb['Genre']))

popular_genre=mtb.groupby('Genre').size().unique

print(popular_genre)

genre_list=mtb['Genre'].values.tolist()
#Calculating the number of songs by each of the artists

print(mtb.groupby('Artist.Name').size())

popular_Artist=mtb.groupby('Artist.Name').size()

print(popular_Artist)

Artist_list=mtb['Artist.Name'].values.tolist()
skew=mtb.skew()

print(skew)

# Removing the skew by using the boxcox transformations

transform=np.asarray(mtb[['Liveness']].values)

# Plotting a histogram to show the difference 

plt.hist(mtb['Liveness'],bins=10) #original data

plt.show()

plt.show()
scatter_matrix(mtb)

plt.gcf().set_size_inches(20, 30)

plt.show()
fig = plt.figure(figsize = (15,7))

mtb.groupby('Artist.Name')['Track.Name'].agg(len).sort_values(ascending = False).plot(kind = 'bar')

plt.xlabel('Artist Name', fontsize = 20)

plt.ylabel('Count of songs', fontsize = 20)

plt.title('Artist Name vs Count of songs', fontsize = 30)
xtick = ['dance pop', 'pop', 'latin', 'edm', 'canadian hip hop',

'panamanian pop', 'electropop', 'reggaeton flow', 'canadian pop',

'reggaeton', 'dfw rap', 'brostep', 'country rap', 'escape room',

'trap music', 'big room', 'boy band', 'pop house', 'australian pop',

'r&b en espanol', 'atl hip hop']

length = np.arange(len(xtick))

genre_groupby = mtb.groupby('Genre')['Track.Name'].agg(len)

plt.figure(figsize = (15,7))

plt.bar(length, genre_groupby)

plt.xticks(length,xtick)

plt.xticks(rotation = 90)

plt.xlabel('Genre', fontsize = 20)

plt.ylabel('Count of the tracks', fontsize = 20)

plt.title('Genre vs Count of the tracks', fontsize = 25)
mtb.plot(kind='box', subplots=True)

plt.gcf().set_size_inches(15,15)

plt.show()
fig=plt.subplots(figsize=(10,10))

plt.title('Dependence between energy and popularity')

sns.regplot(x='Energy', y='Popularity',

            ci=None, data=mtb)

sns.kdeplot(mtb.Energy,mtb.Popularity)
x=mtb.loc[:,['Artist.Name']].values

y=mtb.loc[:,'Genre'].values
x.shape

encoder=LabelEncoder()

x = encoder.fit_transform(x)

x=pd.DataFrame(x)

x
Encoder_y=LabelEncoder()

Y = Encoder_y.fit_transform(y)

Y=pd.DataFrame(Y)

Y