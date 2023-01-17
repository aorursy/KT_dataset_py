#importing libraries

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

from sklearn import metrics
#fetching data

elec_cons = pd.read_csv("../input/us-yearly-electricity-consumption/total-electricity-consumption-us.csv",  sep = ',', header= 0 )

elec_cons.head()
# number of observations: 51

elec_cons.shape
# checking NA

# there are no missing values in the dataset

elec_cons.isnull().values.any()
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
# Doing a polynomial regression: Comparing linear, quadratic and cubic fits

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

    

    # test data in orange

    ax.scatter(X_test, y_test)

    ax.plot(X_test, y_pred)

    plt.show()    

    

# plot errors vs y

    fig, ax = plt.subplots()

    ax.set_xlabel("y_test")                                

    ax.set_ylabel("error")

    ax.set_title("Degree= " + str(degree))

    

    ax.scatter(y_test,y_test-y_pred)

    ax.plot(y_test,y_test-y_test)

    plt.show()

    
# respective train and test r-squared scores of predictions

print(degrees)

print(r2_train)

print(r2_test)