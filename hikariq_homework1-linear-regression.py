# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt

from sklearn import linear_model



# create LinearRegression model

regressor = linear_model.LinearRegression()

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# 1. generate training data



def generate_data(noise_std):



    X_train = np.random.rand(2000, 1)

#     print('xtrain ', X_train)



    noise = np.random.rand(2000,1) * noise_std #std : 0.6



    Y_train = 24 + 4 * X_train + noise

#     print('Y_train', Y_train)



    #test data

    X_test = np.random.rand(50, 1)

    test_noise = np.random.rand(50, 1) * noise_std

    Y_test = 24 + 4 * X_test + test_noise

    

    return X_train, Y_train, X_test, Y_test



X_train, Y_train, X_test, Y_test = generate_data(0.6)


# 2. plot all data in a plot

def plot_data():

    

    plt.plot(X_train[:, 0], Y_train[:, 0], "o", alpha=0.1, label="train data")

    # plot test samples as points ("o")

    plt.plot(X_test[:, 0], Y_test[:, 0], "o", label="test data")

    # set plot title 

    plt.title("Plot generated data: ")

    # set axis name

    plt.xlabel("X")

    plt.ylabel("Y")

    # create legend on top right

    plt.legend()

    # show plot

    plt.show()



plot_data()




# 3. Call linear regression

def train():

# train the regression model using fit()

    regressor.fit(X_train, Y_train)



# 6. plot all dataset and regression line

def plot_regressor_result(Y_pred):

    print(f"Learned parameters- W: {regressor.coef_} - b:{regressor.intercept_}")

    # plot training samples as points ("o"), index 0 to get scalar values, transparency 0.2

    plt.plot(X_train[:, 0], Y_train[:, 0], "o", alpha=0.2, label="train")



    # plot predict data

    plt.plot(X_test[:, 0], Y_pred[:, 0], "o", label="predict")

    # plot test data

    plt.plot(X_test[:, 0], Y_test[:, 0], "o", label="test")

    # plot regression model as a line (X as x axis and projection on regressor line as y axis)

    plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, label="regression line")

    # plt.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, label="regression test")

    # set plot title

    plt.title("Plot calculated data:")

    # set axis name

    plt.xlabel("X")

    plt.ylabel("Y")

    # create legend on top left

    plt.legend()

    # show plot

    plt.show()

    

train()



# 4. predict on test data    

Y_test_predict = regressor.predict(X_test)



# 5. print loss

score_1 = regressor.score(X_test, Y_test)

print(f"Test score: {score_1}")

plot_regressor_result(Y_test_predict)
def redo(std):

    

    plot_data()

    train()

    Y_test_predict2 = regressor.predict(X_test)

    plot_regressor_result(Y_test_predict2)

    score = regressor.score(X_test, Y_test)

    return score


std1 = 0.6

X_train, Y_train, X_test, Y_test = generate_data(std1)

score_1 = redo(std1)


#7: redo with std = 0.9

std = 0.9

X_train, Y_train, X_test, Y_test = generate_data(std)

score_2 = redo(std)

# print(X_train,Y_train)



print(f'compare loss: {score_1} vs {score_2}')
# 8. change std and functions:

# y = 5 + 2x + 3 * x**2 + 4 * x**3 + 2 * x**4 + x**5

def generate_data_2(std):

    



    X_train = np.random.rand(2000, 1)



    noise = np.random.rand(2000,1) * std



    Y_train = 5 + 2 * X_train + 3 * X_train**2 + 4 * X_train**3 + 2 * X_train**4 + X_train**5 + noise



    #test data

    X_test = np.random.rand(50, 1)

    test_noise = np.random.rand(50, 1) * std

    Y_test = 5 + 2 * X_test + 3 * X_test**2 + 4 * X_test**3 + 2 * X_test**4 + X_test**5 + test_noise

    

    return X_train, Y_train, X_test, Y_test
# redo with std : 0.2

std = 0.2

X_train, Y_train, X_test, Y_test = generate_data_2(std)

score_3 = redo(std)



print(f'compare loss 2 vs 3: {score_2} vs {score_3}')