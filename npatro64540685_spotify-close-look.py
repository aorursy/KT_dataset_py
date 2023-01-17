# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/../input/top50spotify2019'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import pandas as pd

import numpy as np

import warnings 

from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

%matplotlib inline
top50 = pd.read_csv('../input/top50spotify2019/top50.csv', encoding='latin1')



top50['Track.Name'] = top50['Track.Name'].astype('category')

top50['Artist.Name'] = top50['Artist.Name'].astype('category')

top50['Genre'] = top50['Genre'].astype('category')



top50.rename(columns={"Loudness..dB..": "Loudness", "Acousticness..": "Acousticness"}, inplace=True)

top50.Genre.unique()
top50.head()
sns.distplot(top50['Beats.Per.Minute'])
sns.distplot(top50['Energy'])
list1 = list()

mylabels = list()

for genre in top50.Genre.cat.categories:

    list1.append(top50[top50.Genre == genre].Danceability)

    mylabels.append(genre)

sns.set_style("whitegrid")

fig, ax = plt.subplots()

fig.set_size_inches(11.7,8.27)

h = plt.hist(list1, bins=30, stacked=True, rwidth=1, label=mylabels)

plt.title("Danceability By Genre",fontsize=35, color="DarkBlue", fontname="Console")

plt.ylabel("Number of Tracks", fontsize=35, color="Red")

plt.xlabel("Danceability", fontsize=35, color="Green")

plt.yticks(fontsize=20)

plt.xticks(fontsize=20)

plt.legend(frameon=True,fancybox=True,shadow=True,framealpha=1,prop={'size':5})

plt.show()

list1 = list()

mylabels = list()

for genre in top50.Genre.cat.categories:

    list1.append(top50[top50.Genre == genre].Energy)

    mylabels.append(genre)

sns.set_style("whitegrid")

fig, ax = plt.subplots()

fig.set_size_inches(11.7,8.27)

h = plt.hist(list1, bins=30, stacked=True, rwidth=1, label=mylabels)

plt.title("Energy By Genre",fontsize=35, color="DarkBlue", fontname="Console")

plt.ylabel("Number of Tracks", fontsize=35, color="Red")

plt.xlabel("Energy", fontsize=35, color="Green")

plt.yticks(fontsize=20)

plt.xticks(fontsize=20)

plt.legend(frameon=True,fancybox=True,shadow=True,framealpha=1,prop={'size':5})

plt.show()
list1 = list()

mylabels = list()

for genre in top50.Genre.cat.categories:

    list1.append(top50[top50.Genre == genre]['Length.'])

    mylabels.append(genre)

sns.set_style("whitegrid")

fig, ax = plt.subplots()

fig.set_size_inches(11.7,8.27)

h = plt.hist(list1, bins=30, stacked=True, rwidth=1, label=mylabels)

plt.title("Length By Genre",fontsize=35, color="DarkBlue", fontname="Console")

plt.ylabel("Number of Tracks", fontsize=35, color="Red")

plt.xlabel("Length", fontsize=35, color="Green")

plt.yticks(fontsize=20)

plt.xticks(fontsize=20)

plt.legend(frameon=True,fancybox=True,shadow=True,framealpha=1,prop={'size':5})

plt.show()
list1 = list()

mylabels = list()

for genre in top50.Genre.cat.categories:

    list1.append(top50[top50.Genre == genre].Danceability)

    mylabels.append(genre)

sns.set_style("whitegrid")

fig, ax = plt.subplots()

fig.set_size_inches(11.7,8.27)

h = plt.hist(list1, bins=30, stacked=True, rwidth=1, label=mylabels)

plt.title("Danceability By Genre",fontsize=35, color="DarkBlue", fontname="Console")

plt.ylabel("Number of Tracks", fontsize=35, color="Red")

plt.xlabel("Danceability", fontsize=35, color="Green")

plt.yticks(fontsize=20)

plt.xticks(fontsize=20)

plt.legend(frameon=True,fancybox=True,shadow=True,framealpha=1,prop={'size':5})

plt.show()
sns.jointplot(x='Energy', y='Popularity',data=top50,kind='hex')
fig, ax = plt.subplots()

fig.set_size_inches(15,20)



sns.heatmap(top50.corr(), annot = True)
sns.lmplot(x='Loudness', y='Energy', data=top50)
X = top50[['Beats.Per.Minute', 'Energy', 'Danceability',

               'Loudness', 'Liveness', 'Valence.', 'Length.', 'Acousticness', 'Speechiness.']]

y = top50['Popularity']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
from statsmodels.api import OLS



OLS(y_train,X_train).fit().summary()
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
from statsmodels.formula.api import ols

results = ols('Loudness ~ Energy', data=top50).fit()

results.summary()
X = top50[['Loudness']]

y = top50['Energy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
from statsmodels.api import OLS



OLS(y_train,X_train).fit().summary()