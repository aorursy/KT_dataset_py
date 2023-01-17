# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the required libraries

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
# Let's take a look at our columns

df.columns
print(df.isnull().sum()) 

# drop them if there are any 

df = df.dropna()
# if we wanted the counts instead, we could just use Count. Count returns the number of instances,

# not the sums of the values like above

x = df.groupby(['Year']).count()

x = x['Global_Sales']

y = x.index.astype(int)



plt.figure(figsize=(12,8))

colors = sns.color_palette("muted")

ax = sns.barplot(y = y, x = x, orient='h', palette=colors)

ax.set_xlabel(xlabel='Number of releases', fontsize=16)

ax.set_ylabel(ylabel='Year', fontsize=16)

ax.set_title(label='Game Releases Per Year', fontsize=20)



plt.show()
vg_data = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
print(vg_data.info())

print(vg_data.describe())
# let's choose a cutoff and drop any publishers that have published less than X games



for i in vg_data['Publisher'].unique():

    if vg_data['Publisher'][vg_data['Publisher'] == i].count() < 60:

        vg_data['Publisher'][vg_data['Publisher'] == i] = 'Other'



for i in vg_data['Platform'].unique():

    if vg_data['Platform'][vg_data['Platform'] == i].count() < 100:

        vg_data['Platform'][vg_data['Platform'] == i] = 'Other'



# try plotting the new publisher and platform data

sns.countplot(x='Publisher', data=vg_data)

plt.title("# Games Published By Publisher")

plt.xticks(rotation=-90)

plt.show()



plat_data = vg_data['Platform'].value_counts(sort=False)

sns.countplot(y='Platform', data=vg_data)

plt.title("# Games Published Per Console")

plt.xticks(rotation=-90)

plt.show()
sns.barplot(x='Genre', y='Global_Sales', data=vg_data)

plt.title("Total Sales Per Genre")

plt.xticks(rotation=-45)

plt.show()
# try visualizing the number of games in a specific genre

for i in vg_data['Platform'].unique():

    vg_data['Genre'][vg_data['Platform'] == i].value_counts().plot(kind='line', label=i, figsize=(20, 10), grid=True)



# set the legend and ticks

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=20, borderaxespad=0.)

plt.xticks(np.arange(12), tuple(vg_data['Genre'].unique()))

plt.tight_layout()

plt.show()
# going to attempt to carry out linear regression and predict the global sales of games

# based off of the sales in North America Sales



X = vg_data.iloc[:, 6].values

y = vg_data.iloc[:, 10].values



# train test split and split the dataframe



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
# reshape the data into long 2D arrays with 1 column and as many rows as necessary

X_train = X_train.reshape(-1, 1)

X_test = X_test.reshape(-1, 1)

y_train = y_train.reshape(-1, 1)

y_test = y_test.reshape(-1, 1)



lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
# Let's create a function for plotting

def plot_regression(classifier):



    plt.scatter(X_train, y_train,color='blue')

    plt.plot(X_train, classifier.predict(X_train), color='red')

    plt.title('(Training set)')

    plt.xlabel('North America Sales')

    plt.ylabel('Global Sales')

    plt.show()



    plt.scatter(X_test, y_test,color='blue')

    plt.plot(X_train, classifier.predict(X_train), color='red')

    plt.title('(Testing set)')

    plt.xlabel('North America Sales')

    plt.ylabel('Global Sales')

    plt.show()

    

plot_regression(lin_reg)

print("Training set score: {:.2f}".format(lin_reg.score(X_train, y_train)))

print("Test set score: {:.2f}".format(lin_reg.score(X_test, y_test)))
# Decision tree regressor

DTree_regressor = DecisionTreeRegressor(random_state=5)

DTree_regressor.fit(X_train, y_train)

plot_regression(DTree_regressor)



print("Training set score: {:.2f}".format(DTree_regressor.score(X_train, y_train)))

print("Test set score: {:.2f}".format(DTree_regressor.score(X_test, y_test)))
# Now let’s try a Random Forest regressor algorithm.

RF_regressor = RandomForestRegressor(n_estimators=300, random_state=5)

RF_regressor.fit(X_train, y_train)

plot_regression(RF_regressor)



print("Training set score: {:.2f}".format(RF_regressor.score(X_train, y_train)))

print("Test set score: {:.2f}".format(RF_regressor.score(X_test, y_test)))
components = [

    ('scaling', StandardScaler()),

    ('PCA', PCA()),

    ('regression', LinearRegression())

]



pca = Pipeline(components)

pca.fit(X_train, y_train)

plot_regression(pca)

print("Training set score: {:.2f}".format(pca.score(X_train, y_train)))

print("Test set score: {:.2f}".format(pca.score(X_test, y_test)))
# We are now going to try different regression algorithms to see what results we get. 

# Let's try Elastic Net regressor

elastic = ElasticNet()

elastic.fit(X_train, y_train)

plot_regression(elastic)

print("Training set score: {:.2f}".format(elastic.score(X_train, y_train)))

print("Test set score: {:.2f}".format(elastic.score(X_test, y_test)))
# Now let’s try Ridge regression

ridge_reg = Ridge()

ridge_reg.fit(X_train, y_train)

plot_regression(ridge_reg)

print("Training set score: {:.2f}".format(ridge_reg.score(X_train, y_train)))

print("Test set score: {:.2f}".format(ridge_reg.score(X_test, y_test)))
# Here’s a Lasso regression implementation

lasso_reg = Lasso()

lasso_reg.fit(X_train, y_train)

plot_regression(lasso_reg)

print("Training set score: {:.2f}".format(lasso_reg.score(X_train, y_train)))

print("Test set score: {:.2f}".format(lasso_reg.score(X_test, y_test)))
# ADA Boost regressor

ada_reg = AdaBoostRegressor()

ada_reg.fit(X_train, y_train)

plot_regression(ada_reg)



print("Training set score: {:.2f}".format(ada_reg.score(X_train, y_train)))

print("Test set score: {:.2f}".format(ada_reg.score(X_test, y_test)))