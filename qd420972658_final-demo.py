# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install regressors

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import statsmodels.formula.api as sm

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
d = pd.read_csv("../input/tmdb_5000_movies.csv")

# d.head()
#Change data to only shows the month value for each release date

months = []

for row in d["release_date"]:

    months.append(pd.to_datetime(row).month)



d["release_month"] = months
#Change data to only shows the business quarter value for each release date

quarters = []

for row in d["release_month"]:

    if row < 4:

        quarters.append(1)

    elif row > 9:

        quarters.append(4)

    elif 3 < row < 7:

        quarters.append(2)

    else:

        quarters.append(3)



d["release_quarter"] = quarters

# d.head(5)
d['genres'] = d['genres'].str.replace('id', '')

d['genres'] = d['genres'].str.replace('name', '')

d['genres'] = d['genres'].str.replace(':', '')

d['genres'] = d['genres'].str.replace('{', '')

d['genres'] = d['genres'].str.replace('}', '')

d['genres'] = d['genres'].str.replace('"', '')

d['genres'] = d['genres'].str.replace(',', '')

d['genres'] = d['genres'].str.replace('0', '')

d['genres'] = d['genres'].str.replace('1', '')

d['genres'] = d['genres'].str.replace('2', '')

d['genres'] = d['genres'].str.replace('3', '')

d['genres'] = d['genres'].str.replace('4', '')

d['genres'] = d['genres'].str.replace('5', '')

d['genres'] = d['genres'].str.replace('6', '')

d['genres'] = d['genres'].str.replace('7', '')

d['genres'] = d['genres'].str.replace('8', '')

d['genres'] = d['genres'].str.replace('9', '')

# print(d['genres'])
#Change data to split up the genre values

scifi = []

for row in d["genres"]:

    if "Science Fiction" in row:

        scifi.append(1)

    else:

        scifi.append(0)



d["scifi"] = scifi
#Change data to split up the genre values

fantasy = []

for row in d["genres"]:

    if "Fantasy" in row:

        fantasy.append(1)

    else:

        fantasy.append(0)



d["fantasy"] = fantasy
#Change data to split up the genre values

romance = []

for row in d["genres"]:

    if "Romance" in row:

        romance.append(1)

    else:

        romance.append(0)



d["romance"] = romance
#Change data to split up the genre values

comedy = []

for row in d["genres"]:

    if "Comedy" in row:

        comedy.append(1)

    else:

        comedy.append(0)



d["comedy"] = comedy
#Change data to split up the genre values

drama = []

for row in d["genres"]:

    if "Drama" in row:

        drama.append(1)

    else:

        drama.append(0)



d["drama"] = drama
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

from sklearn.linear_model import LinearRegression

d = d.dropna()

print(d.isnull().sum())
!pip install pygam
from pygam import LogisticGAM,LinearGAM, s, f, l
# X, y = revenue(return_X_y=True)

dfInput_Linear = pd.DataFrame({'budget':d['budget'],'release_quarter':d["release_quarter"],'fantasy':d["fantasy"],'comedy':d['comedy'],'drama':d['drama'],'scifi':d['scifi']})

dfInput_Linear.head()

X = dfInput_Linear.values
dfOutput_Linear = pd.DataFrame({'revenue': d['revenue']})

dfOutput_Linear.head()

y = dfOutput_Linear.values
gam = LinearGAM(s(0) + s(1) + f(2) + f(3) + f(4) + f(5))

gam.gridsearch(X, y)
gam.summary()
plt.figure();

fig, axs = plt.subplots(1,6,figsize=(20,5));



titles = ['budget', 'release_quarter', 'fantasy', 'comedy', 'drama', 'scifi']

for i, ax in enumerate(axs):

    XX = gam.generate_X_grid(term=i)

    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))

    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')

    if i == 0:

        ax.set_ylim(-30,30)

    ax.set_title(titles[i]);
yy = gam.predict(dfInput_Linear)

yy
# Sample = {'budget':200000000, 'release_quarter':2, 'fantasy':1, 'comedy':1, 'drama':1, 'scifi':1}

Sample = np.array([[200000000.0 ,       2.0 ,           1.0 ,          1.0 ,     1.0 ,      1.0]])

Predict_Revenue = gam.predict(Sample)

print("When the budget is:",Sample[0,0])

print("release in quarter",Sample[0,1])

if Sample[0,2]==1.0:

    print('with fantasy genre')

else:

    print('')

if Sample[0,3]==1.0:

    print('with fantasy comedy')

else:

    print('')

if Sample[0,4]==1.0:

    print('with fantasy drama')

else:

    print('')

if Sample[0,4]==1.0:

    print('with fantasy science friction')

else:

    print('')

print("The revenue prediction is",str(Predict_Revenue))
dfInput_Linear1 = pd.DataFrame({'budget':d['budget'],'vote_average':d["vote_average"],'runtime':d["runtime"],'vote_count':d['vote_count']})

dfInput_Linear1.head()

X1 = dfInput_Linear1.values
dfOutput_Linear1 = pd.DataFrame({'revenue': d['revenue']})

dfOutput_Linear1.head()

y1 = dfOutput_Linear1.values
gam1 = LinearGAM(s(0) + s(1) + s(2) + s(3))

gam1.gridsearch(X1, y1)
gam1.summary()
plt.figure();

fig, axs = plt.subplots(1,4,figsize=(20,5));



titles = ["budget","vote_average","runtime","vote_count"]

for i, ax in enumerate(axs):

    XX = gam1.generate_X_grid(term=i)

    ax.plot(XX[:, i], gam1.partial_dependence(term=i, X=XX))

    ax.plot(XX[:, i], gam1.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')

    if i == 0:

        ax.set_ylim(-30,30)

    ax.set_title(titles[i]);
yy1 = gam1.predict(dfInput_Linear1)

yy1
# Sample1 = {"budget"=200000000.0 , "vote_average"=6.5 , "runtime"=150 , "vote_count"=10000 }

Sample1 = np.array([[200000000.0 ,        6.5 ,               150.0 ,          1000.0 ]])

Predict_Revenue1 = gam1.predict(Sample1)

print("When the budget is:",Sample1[0,0])

print("Vote average is:",Sample1[0,1])

print("Run time is:",Sample1[0,2])

print("Vote count is:",Sample1[0,3])

print("The revenue prediction is",str(Predict_Revenue1))