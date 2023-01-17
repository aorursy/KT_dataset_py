# supress warnings

import warnings

warnings.filterwarnings('ignore')
#importing libraries

import numpy as np

import pandas as pd



# Data Visualizatiom

import matplotlib.pyplot as plt

import seaborn as sns

elec_cons = pd.read_csv("../input/total-electricity-consumption-us.csv",  sep = ',', header= 0 )

elec_cons.head()
#checking duplicates

sum(elec_cons.duplicated(subset = 'Year')) == 0

# No duplicate values
elec_cons.shape
# number of observations: 51
elec_cons.info()
elec_cons.describe()
# Checking Null values

elec_cons.isnull().sum()*100/elec_cons.shape[0]

# There are no NULL values in the dataset, hence it is clean.
# Data is clean and we are good to go.
sns.scatterplot(x = 'Year', y = 'Consumption', data = elec_cons)
# As we can see from the scatterplot there is a non linear relationship between the two variables.
# Let's find out which model suit best to this Data.
# Importing Libraries

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

from sklearn import metrics
# We will manually choose the test-train by picking the data uniformly.
size = len(elec_cons.index)

index = range(0, size, 5)



train = elec_cons[~elec_cons.index.isin(index)]

test = elec_cons[elec_cons.index.isin(index)]

print(len(train))

print(len(test))
# converting X to a two dimensional array, as required by the learning algorithm

X_train = train.Year.values.reshape(-1,1) #Making X two dimensional

y_train = train.Consumption



X_test = test.Year.values.reshape(-1,1) #Making X two dimensional

y_test = test.Consumption
# Pipeline helps you associate two models or objects to be built sequentially with each other, 

# in this case, the objects are PolynomialFeatures() and LinearRegression()



r2_train = []

r2_test = []

degrees = [1, 2, 3]



for degree in degrees:

    pipeline = Pipeline([('poly_features', PolynomialFeatures(degree=degree)),

                     ('model', LinearRegression())])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    r2_test.append(metrics.r2_score(y_test, y_pred))

    

    # training performance

    y_pred_train = pipeline.predict(X_train)

    r2_train.append(metrics.r2_score(y_train, y_pred_train))

    

# plot predictions and actual values against year

    fig, ax = plt.subplots()

    ax.set_xlabel("Year")                                

    ax.set_ylabel("Power consumption")

    ax.set_title("Degree= " + str(degree))

    

    # train data in blue

    ax.scatter(X_train, y_train)

    ax.plot(X_train, y_pred_train)

    

    # test data

    ax.scatter(X_train, y_train)

    ax.plot(X_test, y_pred)

    

    plt.show()
# respective test r-squared scores of predictions

print(degrees)

print(r2_train)

print(r2_test)