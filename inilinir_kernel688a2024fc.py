# Bu proje için gerekli kütüphaneleri içe aktar

import numpy as np

import pandas as pd

from sklearn.model_selection import ShuffleSplit



%matplotlib inline



# Veriseti yukle

data = pd.read_csv('../input/train.csv')

data = data.select_dtypes(include=[np.number])

data = data.fillna(data.mean())



prices = data['SalePrice']

features = data.drop('SalePrice', axis = 1)

    

# Basarili

print("Dataset has {} data points with {} variables each.".format(*data.shape))
# Min fiyat

minimum_price = np.amin(prices)



# Maximum fiyat

maximum_price = np.amax(prices)



# Ortalama fiyat

mean_price = np.mean(prices)



# Median fiyat

median_price = np.median(prices)



# Fiyatlarin standart sapmasi

std_price = np.std(prices)





print("Minimum price: ${}".format(minimum_price)) 

print("Maximum price: ${}".format(maximum_price))

print("Mean price: ${}".format(mean_price))

print("Median price ${}".format(median_price))

print("Standard deviation of prices: ${}".format(std_price))
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





# sns.pairplot(data, height=2.5)

# plt.tight_layout()



# cm = np.corrcoef(data.values.T)

# sns.set(font_scale=1.5)

# hm = sns.heatmap(cm,

#                 cbar=True,

#                 annot=True,

#                 square=True,

#                 fmt='.2f',

#                 annot_kws={'size': 15},

                

#                 )



corrmat = data.corr()

f, ax = plt.subplots(figsize=(20, 9))

sns.heatmap(corrmat, vmax=.8, annot=True);
from sklearn.metrics import r2_score



def performance_metric(y_true, y_predict):

    """ Calculates and returns the performance score between 

        true (y_true) and predicted (y_predict) values based on the metric chosen. """

    

    score = r2_score(y_true, y_predict)

    

    # Return the score

    return score
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state = 42)



# Basarili

print("Egitim ve test ayrilmasi basarili")
# Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV



def fit_model(X, y):

    """ Performs grid search over the 'max_depth' parameter for a 

        decision tree regressor trained on the input data [X, y]. """

    

    # Create cross-validation sets from the training data

    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)



    # Create a decision tree regressor object

    regressor = DecisionTreeRegressor()



    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10

    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}



    # Transform 'performance_metric' into a scoring function using 'make_scorer' 

    scoring_fnc = make_scorer(performance_metric)



    # Create the grid search cv object --> GridSearchCV()

    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)



    # Fit the grid search object to the data to compute the optimal model

    grid = grid.fit(X, y)



    # Return the optimal model after fitting the data

    return grid.best_estimator_
# Fit the training data to the model using grid search

# NaN degerlerin ortalama degerle degistirilmesi

X_train = X_train.fillna(X_train.mean())

y_train = y_train.fillna(y_train.mean())

X_test = X_test.fillna(X_test.mean())

y_test = y_test.fillna(y_test.mean())



reg = fit_model(X_train, y_train)



# Produce the value for 'max_depth'

print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
from sklearn.metrics import make_scorer

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

lr = LinearRegression()

lr.fit(X_train,y_train)

test_pre = lr.predict(X_test)

train_pre = lr.predict(X_train)



print(r2_score(y_test,test_pre))



plt.scatter(train_pre, train_pre - y_train, c = "blue",  label = "Training data")

plt.scatter(test_pre,test_pre - y_test, c = "black",  label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()
test = pd.read_csv("../input/test.csv")

print(test.head())
test = test.select_dtypes(include=[np.number])

test = test.fillna(test.mean())



predicted_prices = lr.predict(test)

print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

print(my_submission)

my_submission.to_csv('submission.csv', index=False)