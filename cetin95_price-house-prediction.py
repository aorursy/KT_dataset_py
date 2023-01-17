import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import cross_val_score

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.metrics.scorer import make_scorer

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_classification

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn import tree

import pydot

from sklearn.tree import DecisionTreeRegressor



from IPython.display import Image  



%matplotlib inline







# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train_y = train.SalePrice

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']



# Create training predictors data

train_X = train[predictor_cols]



my_model = RandomForestRegressor()

my_model.fit(train_X, train_y)

test_X = test[predictor_cols]

# Use the model to make predictions

predicted_prices = my_model.predict(test_X)

# We will look at the predicted prices to ensure we have something sensible.

print(predicted_prices)
plt.scatter(train.loc[:,"GrLivArea"], train.loc[:,"SalePrice"])
train2 = train[(train.GrLivArea < 4000) & (train.SalePrice < 400000)]

plt.scatter(train2.loc[:,"GrLivArea"], train2.loc[:,"SalePrice"])
train.head()
# Take a small sample of the data in order to work with nicer diagrams.

train2 = train2.sample(frac=0.2)

X = train2[['GrLivArea']]

y = train2['SalePrice']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)

plt.scatter(X_train, y_train)
# fit a linear model

model = linear_model.LinearRegression()

model = model.fit(X_train, y_train)

train_predictions = model.predict(X_train)

test_predictions = model.predict(X_test)
#Plot regression line on training data

plt.scatter(X_train, y_train,  color='black')

plt.plot(X_train, train_predictions, color='blue', linewidth=3)
# Show the intercept and the coefficient of the model (which is a line)

print (model.intercept_)

print (model.coef_)
#Plot regression line on test data

plt.scatter(X_test, y_test,  color='black')

plt.plot(X_test, test_predictions, color='blue', linewidth=3)
# training and test set mean absolute error (MAE)

print("Training MAE:", mean_absolute_error(y_train, train_predictions))

# or

print(np.mean(np.abs(y_train - train_predictions)))



print("Test MAE:", mean_absolute_error(y_test, test_predictions))
def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("Test MAPE:", mean_absolute_percentage_error(y_test, test_predictions))
print("r-squared:", model.score(X_test, y_test))
 #baseline - predict always the mean SalePrice value



mean_sales_price = y_train.mean()

mean_array = np.ones(y_test.size)*mean_sales_price

print("Baseline (mean) MAE:", mean_absolute_error(y_test, mean_array))

print("Baseline (mean) MAPE:", mean_absolute_percentage_error(y_test, mean_array))
from sklearn.metrics import mean_squared_log_error





def kaggle_score(y_true,y_pred):

    return np.sqrt(mean_squared_log_error(y_true, y_pred));



print("Test Kaggle-score:", kaggle_score(y_test, test_predictions))

print("Baseline (mean) Kaggle-score:", kaggle_score(y_test,mean_array))
# Prediction with two input variables



X = train.loc[:,['GrLivArea','BedroomAbvGr']]

y = train.loc[:,'SalePrice']



X.head(5)
#train = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

#train.to_csv('submission.csv', index=False)
X, y = make_classification(n_samples = 10,n_features=2, n_redundant=0, n_informative=1, class_sep=0.2,

                             n_clusters_per_class=1, random_state=2)

plt.figure(figsize=(5,5))

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,

            s=40, edgecolor='k');
X, y = make_classification(n_samples = 15, n_features=2, n_redundant=0, n_informative=1, class_sep=0.5,

                             n_clusters_per_class=1, random_state=1)



fig, ax = plt.subplots(1,2,figsize=(10,4));



ax[0].scatter(X[:, 0], X[:, 1], marker='o', c=y,

            s=40, edgecolor='black');



clf = DecisionTreeClassifier().fit(X, y)

clf.fit(X, y)

plot_step = 0.02

n_classes = 2

plot_colors = "rgb"



x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),

                         np.arange(y_min, y_max, plot_step))

#plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)



Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)



ax[1].set_ylim(ax[0].get_ylim())

ax[1].set_xlim(ax[0].get_xlim())

for i, color in zip(range(n_classes), plot_colors):

    idx = np.where(y == i)

    ax[1].scatter(X[idx, 0], X[idx, 1], c=color, 

                cmap=plt.cm.RdYlBu, edgecolor='black', s=40)
dot_data = tree.export_graphviz(clf, out_file=None) 

# Draw graph

graph = pydot.graph_from_dot_data(dot_data)  



# Show graph

Image(graph[0].create_png()) 
plt.figure(figsize=(5,5))

X, y = make_classification(n_samples = 30, n_features=2, n_redundant=0, n_informative=2, class_sep=0.9,

                             n_clusters_per_class=1, random_state=1, n_classes=3)

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,

            s=40, edgecolor='black');



clf = DecisionTreeClassifier()

# You can play with n_neighbors

clf.fit(X, y)

plot_step = 0.02

n_classes = 3

plot_colors = "rgb"



x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),

                         np.arange(y_min, y_max, plot_step))

plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)



Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)



for i, color in zip(range(n_classes), plot_colors):

    idx = np.where(y == i)

    plt.scatter(X[idx, 0], X[idx, 1], c=color, 

                cmap=plt.cm.RdYlBu, edgecolor='black', s=40)

dot_data = tree.export_graphviz(clf, out_file=None) 

# Draw graph

graph = pydot.graph_from_dot_data(dot_data)  



# Show graph

Image(graph[0].create_png())
plt.figure(figsize=(5,5))

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,

            s=40, edgecolor='black');



clf = DecisionTreeClassifier(min_samples_split=15)

clf.fit(X, y)

plot_step = 0.02

n_classes = 3

plot_colors = "rgb"



x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),

                         np.arange(y_min, y_max, plot_step))

plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)



Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)



for i, color in zip(range(n_classes), plot_colors):

    idx = np.where(y == i)

    plt.scatter(X[idx, 0], X[idx, 1], c=color, 

                cmap=plt.cm.RdYlBu, edgecolor='black', s=40)
dot_data = tree.export_graphviz(clf, out_file=None) 

# Draw graph

graph = pydot.graph_from_dot_data(dot_data)  



# Show graph

Image(graph[0].create_png())


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)

plt.scatter(X_train, y_train)
# Fit regression model

regr_1 = DecisionTreeRegressor(max_depth=1)

#play with max_depth



regr_1.fit(X_train, y_train)



# Predict

X_test = np.arange(0.0, 3000.0, 1)[:, np.newaxis]

y_1 = regr_1.predict(X_test)





fig, ax = plt.subplots(1,2,figsize=(10,4));



# Plot the results

ax[0].scatter(X_train, y_train)



ax[1].set_ylim(ax[0].get_ylim())

ax[1].set_xlim(ax[0].get_xlim())

ax[1].scatter(X_train, y_train, s=20, edgecolor="black",

            c="darkorange", label="data")

ax[1].plot(X_test, y_1, color="cornflowerblue",

         label="max_depth=2", linewidth=2)

#plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)

plt.xlabel("data")

plt.ylabel("target")

plt.title("Decision Tree Regression")

plt.legend()

plt.show()
dot_data = tree.export_graphviz(regr_1, out_file=None) 

# Draw graph

graph = pydot.graph_from_dot_data(dot_data)  



# Show graph

Image(graph[0].create_png())
train = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

train.to_csv('submission.csv', index=False)