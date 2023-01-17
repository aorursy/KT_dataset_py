import numpy as np 

import pandas as pd 

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()





# Input data files are available in the "../input/" directory.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#Some additional imports

%matplotlib inline

from scipy import stats, integrate

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression #will be used later for our linear regression

import seaborn as sns

sns.set(color_codes=True)
#Import the data into train and test



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# let's start by reviewing the data 

train.head()

# we have 81 columns of data!
# I want to start by reviewing the correlation between all of the variables



#Correlation map to see how features are correlated with SalePrice

#correlationmatrix = train.corr()

#plt.subplots(figsize=(25,15))  # we need to make it bigger

#sns.heatmap(correlationmatrix, vmax=0.9, square=True)
train['TotalBsmtSF'].mean()
train['GarageArea'].mean()
train['GarageCars'].mean()
sns.distplot(train["SalePrice"])
# we'll want to use the log of Sale Price

(np.log(train["SalePrice"])).hist()

import numpy as np

import pandas as pd

import seaborn as sns

from scipy import stats

import matplotlib as mpl

import matplotlib.pyplot as plt



#with sns.axes_style("white"):

 #   g = sns.FacetGrid(train, row="YrSold", col="Street", margin_titles=True, size=2.5)

#g.map(plt.scatter, "SalePrice", "GarageArea", color="#334488", edgecolor="white", lw=.5);

#g.set_axis_labels("Sale Price (US Dollars)", "Garage Area");

#g.set(xticks=[10, 30, 50], yticks=[2, 6, 10]);

#g.fig.subplots_adjust(wspace=.02, hspace=.02);
sns.boxplot(train["SalePrice"])
fig, splivarea = plt.subplots()

splivarea.scatter(x = train['GrLivArea'], y = train['SalePrice'])
trainstats1=train.describe()

trainstats1
train = train[train['GrLivArea'] <= 4000]
fig, splivarea2 = plt.subplots()

splivarea2.scatter(x = train['GrLivArea'], y = train['SalePrice'])
sns.regplot(x="SalePrice", y="GrLivArea", data=train);
train.shape
train.fillna(train.mean())

test.fillna(test.mean())
train.shape
#train = train.dropna(axis=1)

#test = test.dropna(axis=1)
train = pd.get_dummies(train)

train = train.fillna(train.mean())

test = pd.get_dummies(test)
#test["EveryoneIsAverage"] = train['SalePrice'].mean()
#Let's run a linear regression model between saleprice and GrLivArea.  That is, let us determine if we can

#predict the SalePrice (on avergae) using the GrLivArea variable.
#X = train.iloc[:, 1:-1] #include all columns except for saleprice

y = train.iloc[:, -1]

X = train.drop(['SalePrice', 'Id'], axis=1)
test.shape
#Import Module

from sklearn.model_selection import train_test_split





train_X, test_X, train_y, test_y = train_test_split(X, y, 

                                                    train_size=0.66,

                                                    test_size=0.34,

                                                    random_state=123)
from sklearn.linear_model import LinearRegression

classifier = LinearRegression()

classifier.fit(train_X, train_y)
print ("R^2 is: \n", classifier.score(test_X, test_y))
classifier.score(train_X, train_y)
#rm = LinearRegression.Ridge(alpha=1)

#ridge_model = rm.fit(train_X, train_y)

#preds_ridge = ridge_model.predict(test_X)
from sklearn.neighbors import KNeighborsClassifier

#This creates a model object.

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(train_X, train_y)

pred_y = knn.predict(test_X)
test.fillna(0, inplace=True)
pred_y3 = knn.predict(test)

knn.score(test_X, test_y)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



forest_model = RandomForestRegressor()

forest_model.fit(train_X, train_y)

pred_y1 = forest_model.predict(test_X)
forest_model.score(train_X, train_y)
forest_model.score(test_X, test_y)
test_X.shape
submission = pd.DataFrame()

submission['Id'] = test.Id

feats = test.select_dtypes(

        include=[np.number]).drop(['Id'], axis=1).interpolate()

predictions = pred_y1

final_predictions = np.exp(pred_y1)
#submission.to_csv('submission1.csv', index=False)
#test['SalePrice'] = final_predictions

#submission.head()
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice":pred_y1

    })

submission.to_csv('EveryoneIsAverage.csv', index=False)