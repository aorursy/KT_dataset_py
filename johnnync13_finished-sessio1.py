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
# some imports



from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))



# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)

 

# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

plt.rc('font', size=12) 

plt.rc('figure', figsize = (12, 5))



# Settings for the visualizations

import seaborn as sns

sns.set_style("whitegrid")

sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2,'font.family': [u'times']})



import pandas as pd

pd.set_option('display.max_rows', 25)

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 50)



# Ignore useless warnings (see SciPy issue #5998)

import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")



# create output folder

if not os.path.exists('output'):

    os.makedirs('output')

if not os.path.exists('output/session1'):

    os.makedirs('output/session1')
## load data

train_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/train_set.csv',index_col=0) 

test_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/test_set.csv',index_col=0) 

train_set.describe()
# print the dataset size

print("There is", train_set.shape[0], "samples")

print("Each sample has", train_set.shape[1], "features")
# print the top elements from the dataset

train_set.head()
train_set.tail()
# As it can be seen the database contains several features, some of them numerical and some of them are categorical.

# It is important to check each of the to understand it.
# we can see the type of each features as follows

train_set.dtypes
# print those categorical features

train_set.select_dtypes(include=['object']).head()
# We can check how many different type there is in the dataset using the folliwing line

train_set["Type"].value_counts()
sns.countplot(y="Type", data=train_set, color="c")
sns.distplot(train_set["Price"])

plt.show()
fig = plt.figure(figsize = (20,10))

ax = fig.gca()

train_set.hist(ax = ax)

plt.show()
sns.distplot(train_set["YearBuilt"])

plt.show()
sns.distplot(train_set["Distance"])

plt.show()
sns.distplot(train_set["Longtitude"])

plt.show()
sns.distplot(train_set["Rooms"])
train_set = pd.get_dummies(train_set,columns= ['Regionname','Type','Method'],drop_first=True)
train_set.head()
test_set = pd.get_dummies(test_set,columns= ['Regionname','Type','Method'],drop_first=True)
test_set.head()
train_set = train_set.drop(['Suburb', 'Address', 'SellerG','Date','CouncilArea'],axis=1)
train_set = train_set[(train_set["Price"]>0) & (train_set["Price"] <= 6500000.0)]
test_set = test_set.drop(['Suburb', 'Address', 'SellerG','Date','CouncilArea'],axis=1)
train_set
from sklearn.preprocessing import StandardScaler



y = train_set[['Price']]



train_set = train_set.drop(['Price'],axis=1)



train_set = train_set.fillna(train_set.mean())

test_set = test_set.fillna(test_set.mean())



sc_X = StandardScaler()

X = sc_X.fit_transform(train_set.values)

X_test = sc_X.transform(test_set.values)



X = pd.DataFrame(X)

X.columns = train_set.columns.tolist()

train_set = X



X_test = pd.DataFrame(X_test)

X_test.columns = test_set.columns.tolist()



test_set = X_test
## the features





"""

features = train_set.columns.tolist()

features.remove('Bedroom2')

features.remove('Car')

features.remove('Bathroom')

features.remove('Postcode')

"""

features = ['Longtitude','Rooms','Distance','Lattitude','Landsize']



## DEFINE YOUR FEATURES

X = train_set[features]

#y = train_set[['Price']]



## the model

# KNeighborsRegressor

from sklearn import neighbors

#amb 6 349281.15504

#amb 7 345742.31580 hagués quedat primer

n_neighbors = 6 # you can modify this paramenter (ONLY THIS ONE!!!) 



model = neighbors.KNeighborsRegressor(n_neighbors)



## fit the model

model.fit(X, y)



## predict training set

y_pred = model.predict(X)



## Evaluate the model and plot it

from sklearn.metrics import mean_squared_error, r2_score

print("----- EVALUATION ON TRAIN SET ------")

print("RMSE",np.sqrt(mean_squared_error(y, y_pred)))

print("R^2: ",r2_score(y, y_pred))





plt.scatter(y, y_pred)

plt.xlabel('Price')

plt.ylabel('Predicted price');

plt.show()

ax = sns.regplot(x=y, y=y_pred,line_kws={"color": "red"} )







## predict the test set and generate the submission file

X_test = test_set[features]

y_pred = model.predict(X_test)



df_output = pd.DataFrame(y_pred)

df_output = df_output.reset_index()

df_output.columns = ['index','Price']



df_output.to_csv('output/session1/neighbours.csv',index=False)
if __name__ == '__main__':

    # Mostrem únicament 100 dades en el gràfic

    size = 100

    sample = np.random.choice(X.shape[0], size=size, replace=False)

    y = y.to_numpy()

    y_pred = y_pred[:X.shape[0]]

    # Mínima i màxima Y a mostrar en el gràfic

    miny = np.minimum(y_pred[sample].min(), y[sample].min()) + 1e2

    maxy = np.maximum(y_pred[sample].max(), y[sample].max()) + 1e6



    # Visualització de les dades originals i les prediccions

    plt.figure(figsize=(20, 4))

    plt.scatter(range(size), y[sample], color='green', label = 'price')

    plt.scatter(range(size), y_pred[sample], color='red', label = 'pred_price')

    plt.xlabel('100 random samples')

    plt.ylabel('Price')

    plt.yscale('symlog')

    plt.ylim([miny, maxy])

    plt.grid(axis='y', which='minor', alpha=0.2)

    plt.grid(axis='y', which='major', alpha=0.3)

    plt.legend()



    for x in range(size):

        plt.plot((x, x), (miny, maxy), '-.', color='gray', alpha=0.2)



    print('Mean error: {:.2f}'.format(np.mean(np.abs(y - y_pred))))

    print("RMSE",np.sqrt(mean_squared_error(y, y_pred)))
## the features



#features = ['Longtitude','Rooms','Bathroom','Bedroom2','Postcode','Distance','BuildingArea','Lattitude','YearBuilt','Landsize']

features = train_set.columns.tolist()

features.remove('Postcode')



## DEFINE YOUR FEATURES

X = train_set[features]

#y = train_set[['Price']]



## the model

# LinearRegressor

from sklearn.linear_model import LinearRegression

lm = LinearRegression()



## fit the model

lm.fit(X, y)



## predict training set

y_pred = lm.predict(X)



## Evaluate the model and plot it

from sklearn.metrics import mean_squared_error, r2_score

print("----- EVALUATION ON TRAIN SET ------")

print("RMSE",np.sqrt(mean_squared_error(y, y_pred)))

print("R^2: ",r2_score(y, y_pred))









plt.scatter(y, y_pred)

plt.xlabel('Price')

plt.ylabel('Predicted price');

plt.show()

ax = sns.regplot(x=y, y=y_pred,line_kws={"color": "red"} )





## predict the test set and generate the submission file

X_test = test_set[features]

y_pred = lm.predict(X_test)



df_output = pd.DataFrame(y_pred)

df_output = df_output.reset_index()

df_output.columns = ['index','Price']



df_output.to_csv('output/session1/linear_regression.csv',index=False)
if __name__ == '__main__':

    # Mostrem únicament 100 dades en el gràfic

    size = 100

    sample = np.random.choice(X.shape[0], size=size, replace=False)

    y_pred = y_pred[:X.shape[0]]

    # Mínima i màxima Y a mostrar en el gràfic

    miny = np.minimum(y_pred[sample].min(), y[sample].min()) + 1e5

    maxy = np.maximum(y_pred[sample].max(), y[sample].max()) + 1e6



    # Visualització de les dades originals i les prediccions

    plt.figure(figsize=(20, 4))

    plt.scatter(range(size), y[sample], color='green', label = 'price')

    plt.scatter(range(size), y_pred[sample], color='red', label = 'pred_price')

    plt.xlabel('100 random samples')

    plt.ylabel('Price')

    plt.yscale('symlog')

    plt.ylim([miny, maxy])

    plt.grid(axis='y', which='minor', alpha=0.2)

    plt.grid(axis='y', which='major', alpha=0.3)

    plt.legend()



    for x in range(size):

        plt.plot((x, x), (miny, maxy), '-.', color='gray', alpha=0.2)

    print('Mean error: {:.2f}'.format(np.mean(np.abs(y - y_pred))))

    print("RMSE",np.sqrt(mean_squared_error(y, y_pred)))

display(features)
train_set.corr()
plt.figure(figsize=(20,10))

plt.title('Correlacion',fontsize=15)

sns.heatmap(train_set.corr(),annot=True,fmt='.2f',cmap='RdYlGn')