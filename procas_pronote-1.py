import matplotlib.pyplot as plt
import pandas as pd

plt.show()
games=pd.read_csv('../input/game_data.csv')
print(games.columns)


print(games.shape)
import matplotlib.pyplot as plt
plt.hist(games["average_rate"])
plt.show()


# print the first row of all games with average rating 4
#iloc[0] gives Game 3
#iloc[1] gives Game 4
#iloc[2] gives Game 5
#iloc[3] gives Game 6
#iloc[4] gives Game 8

print(games[games["average_rate"]==4].iloc[4])

games=games.dropna(axis=0)

#clustering : grouping similar types of data together
from sklearn.cluster import KMeans
## initialise with number of clusters and random state
kmeans_model=KMeans(n_clusters=5, random_state=1)
## get only the numeric columns from the dataset
num_columns=games._get_numeric_data()
## fit the model using numeric columns
kmeans_model.fit(num_columns)
## get the cluster assignments
labels=kmeans_model.labels_
#plotting the clusters
## import the PCA model : Pca is a dimentionality reduction technique rather than
# a machine learning model

from sklearn.decomposition import PCA

#Create a PCA model

pca_2=PCA(2)
## fit the PCA model to the numeric columns
plot_columns=pca_2.fit_transform(num_columns)
## make a scatter plot of each game, shaded according to cluster assignment labels
plt.scatter (x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
## show the plot
plt.show()
games.corr()["average_rate"]
# # we notice that users_rate, total_weight and average_weight have the highest
# correlations to average_rate
#splitting data into training and test sets
## import convenience function to split the sets

from sklearn.cross_validation import train_test_split

## generate training and test sets
train=games.sample(frac=0.8, random_state=1)
## select anything not in the training set ans put it in the testing set
test=games.loc[~games.index.isin(train.index)]
## print shapes of both the sets

print(train.shape)
print(test.shape)
# Fitting linear regression
## import linear regression model
from sklearn.linear_model import LinearRegression
## initialise the model class
model=LinearRegression()
## define target for prediction and predicting columns
columns=games.columns.tolist()
columns=[c for c in columns if c not in ["average_rate", "name"]]
target="average_rate"
## fit the model into training data
model.fit(train[columns], train[target])
#predicting the error

## import scikit-learn function to compute error
from sklearn.metrics import mean_squared_error

## generate our predictions for the test set
predictions=model.predict(test[columns])

##compute error between test predictions and actual values
mean_squared_error(predictions, test[target])
# Fitting RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

## initialise model with some parameters
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
## fit model to the data
model.fit(train[columns], train[target])
## compute the error
mean_squared_error(predictions, test[target])
