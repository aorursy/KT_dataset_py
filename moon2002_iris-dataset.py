import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn import datasets

iris = pd.read_csv('../input/iris/Iris.csv')

iris.head()
import seaborn as sns; sns.set()

sns.pairplot(iris, hue='Species', height=1.5);
X_iris = iris.drop('Species', axis=1)

X_iris.shape
y_iris = iris['Species']

y_iris.shape
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,

                                                random_state=1)
from sklearn.naive_bayes import GaussianNB # 1. choose model class

model = GaussianNB()                       # 2. instantiate model

model.fit(Xtrain, ytrain)                  # 3. fit model to data

y_model = model.predict(Xtest)             # 4. predict on new data
from sklearn.metrics import accuracy_score

accuracy_score(ytest, y_model)
from sklearn.decomposition import PCA  # 1. Choose the model class

model = PCA(n_components=2)            # 2. Instantiate the model with hyperparameters

model.fit(X_iris)                      # 3. Fit to data. Notice y is not specified!

X_2D = model.transform(X_iris)         # 4. Transform the data to two dimensio
iris['PCA1'] = X_2D[:, 0]

iris['PCA2'] = X_2D[:, 1]

sns.lmplot("PCA1", "PCA2", hue='Species', data=iris, fit_reg=False);
from sklearn.mixture import GaussianMixture      # 1. Choose the model class

model = GaussianMixture(n_components=3,

            covariance_type='full')  # 2. Instantiate the model with hyperparameters

model.fit(X_iris)                    # 3. Fit to data. Notice y is not specified!

y_gmm = model.predict(X_iris)        # 4. Determine cluster labels
iris['cluster'] = y_gmm

sns.lmplot("PCA1", "PCA2", data=iris, hue='Species',

           col='cluster', fit_reg=False);