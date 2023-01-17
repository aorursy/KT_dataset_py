import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.datasets import make_blobs



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
instances = 1000

groups = 3

features = 2



# Randomly generates multicategory data for classification

# random state ensures same data each time so we can compare runs

X, y = make_blobs(n_samples=instances, centers=groups, n_features=features, random_state=99)



# Extractin the input information and putting it into a dataframe for manipulations

df_X = pd.DataFrame()

df_X['bias'] = [1]*instances

df_X['feat1'] = X[:,0]

df_X['feat2'] = X[:,1]

print(df_X.head())



# bringing the y values into a dataframe for manipulations

df_y = pd.DataFrame()

df_y['label'] = y

print(df_y.head())



# PLotting the three groups according to the two features

plt.scatter(df_X.loc[df_y['label'] == 0, 'feat1'], df_X.loc[df_y['label'] == 0, 'feat2'],

            c='r', label = 'Cat. 0')

plt.scatter(df_X.loc[df_y['label'] == 1, 'feat1'], df_X.loc[df_y['label'] == 1, 'feat2'],

            c='m', label = 'Cat. 1')

plt.scatter(df_X.loc[df_y['label'] == 2, 'feat1'], df_X.loc[df_y['label'] == 2, 'feat2'],

            c='b', label = 'Cat. 2')

plt.title('The three categories')

plt.xlabel('feat 1')

plt.ylabel('feat 2')

plt.legend()

plt.show()
# Splitting the y-values into test and train

df_X_test = df_X[int(instances * 0.8):]

df_X = df_X[:int(instances * 0.8)]



# Creating the dummy variables. Ie chaging the three category label to three binary categories

dummies = pd.get_dummies(df_y['label'])

df_y = pd.concat((df_y, dummies), axis = 1)



# Splitting the y-values into test and train

df_y_test = df_y[int(instances * 0.8):]

df_y = df_y[:int(instances * 0.8)]



print(df_y.head())
# The fucntion we get our probability of yes or no (0 or 1)

def log_func(X, theta):

    return 1/(1+np.exp(-np.matmul(X, theta)))



# The quantification of how good our parameters are

def cost_func(X, y, theta):

    mm = len(X)

    return (-1/mm) * np.sum((y * np.log(log_func(X, theta))) + ((1-y) * (np.log(1-log_func(X, theta)))))



# Adjusts the weights according to the partial derivatives of the cost function

def gradient_descent(X, y, theta, alpha):

    mm = len(y)

    return theta - ((alpha*mm)*np.matmul(np.transpose(X), (log_func(X, theta))-y))
# Feature scaling to aid convergence

df_X['feat1'] = df_X['feat1'] / (df_X['feat1'].max() - df_X['feat1'].min())

df_X['feat2'] = df_X['feat2'] / (df_X['feat2'].max() - df_X['feat2'].min())



# Adding polynomial terms as there are groups which are not so easily split by linear

df_X['feat1^2'] = df_X['feat1'] ** 2

df_X['feat2^2'] = df_X['feat2'] ** 2



print(df_X.head())



# Obtaining our feature inputs (same as before)

X = df_X.values 

# Our optimization parameter

alpha = 1e-6

# The number of loops we are using to converge

loops = 100000
# Loops the parameter update over and over to get the best possible parameters

def get_params(target, alpha, loops):

    y = df_y[target].values

    # Our inital guess of the parameters, this time in vector form

    theta = np.random.randint(0, 2, X.shape[1])



    print(f'Initial parameters are {theta}')



    # This will be used to track the cost over the iterations to check it is decreasing

    cost_tracker = []

    for ii in range(loops):

        cost_tracker.append(cost_func(X, y, theta))

        theta = gradient_descent(X, y, theta, alpha)



    print(f'Final parameters are {theta}')

    

    #PLotting the cost as a function of its the iteration number

    plt.plot(cost_tracker)

    plt.xlabel('iteration number')

    plt.ylabel('Cost')

    plt.title(f'Category {target}')

    plt.show()

    

    return theta



print('Category 0\n')

theta_0 = get_params(0, alpha, loops)

print('Category 1\n')

theta_1 = get_params(1, alpha, loops)

print('Category 2\n')

theta_2 = get_params(2, alpha, loops)
# Finding the value of the logistic function for a single pair of values of the features

def find_probability(weights, x1, x2):

    return weights[0] + weights[1]*x1 + weights[2]*x2 + weights[3]*x1*x1 + weights[4]*x2*x2



# Finds the probability of being a certain category for many points

def find_grid(grid, weights):

    prob = []

    for row in grid:

        prob.append(1/(1+np.exp(-find_probability(weights, row[0], row[1]))))

    return np.array(prob)



# Creating the 2d area we will plot our decision line in

x1 = np.linspace(0, 1.1, 10)

x2 = np.linspace(-1.0, 0.5, 10)

ax1, ax2 = np.meshgrid(x1,x2)

grid = np.c_[ax1.ravel(), ax2.ravel()]
# PLots out probabilities in 2d space in order to visialise them

def plot_prob_grid(theta, name='Category'):

    # gets the probability array

    prob_grid = find_grid(grid, theta).reshape(10,10)



    fig, ax = plt.subplots(dpi = 100)



    # Plots the probabilities as a contour plot

    contour = ax.contourf(ax1, ax2, prob_grid)

    ax_c = fig.colorbar(contour)

    ax_c.set_label(f"$P(y = Cat. {name})$")

    ax_c.set_ticks([0, .25, .5, .75, 1])



    ax = plt.scatter(df_X.loc[df_y['label'] == 0, 'feat1'], df_X.loc[df_y['label'] == 0, 'feat2'],

                c='r', label = 'Cat. 0', s=1)

    ax = plt.scatter(df_X.loc[df_y['label'] == 1, 'feat1'], df_X.loc[df_y['label'] == 1, 'feat2'],

                c='m', label = 'Cat. 1', s=1)

    ax = plt.scatter(df_X.loc[df_y['label'] == 2, 'feat1'], df_X.loc[df_y['label'] == 2, 'feat2'],

                c='b', label = 'Cat. 2', s=1)

    plt.title(f'Decision boundary for category {name}')

    plt.xlabel('feat 1')

    plt.ylabel('feat 2')

    plt.legend()



    plt.show()

    

plot_prob_grid(theta_0, name = '0')

plot_prob_grid(theta_1, name = '1')

plot_prob_grid(theta_2, name = '2')
# Calculating the probability of each category for all the points in our train

df_y['prob_0'] = log_func(X, theta_0)

df_y['prob_1'] = log_func(X, theta_1)

df_y['prob_2'] = log_func(X, theta_2)



# Fidning the highest value of our probabilites

df_y['Pred_label'] = df_y[['prob_0','prob_1', 'prob_2']].idxmax(axis=1)

df_y['Pred_label'] = df_y['Pred_label'].replace({"prob_0": 0,

                                                 "prob_1": 1,

                                                 "prob_2": 2})



print(df_y.head())

print('\nPearson R correlation - ', df_y['label'].corr(df_y['Pred_label']))
print(df_X_test.head())

# Feature scaling for the test values

df_X_test['feat1'] = df_X_test['feat1'] / (df_X_test['feat1'].max() - df_X_test['feat1'].min())

df_X_test['feat2'] = df_X_test['feat2'] / (df_X_test['feat2'].max() - df_X_test['feat2'].min())



# Getting the polynomial terms for the test

df_X_test['feat1^2'] = df_X_test['feat1'] ** 2

df_X_test['feat2^2'] = df_X_test['feat2'] ** 2



# Extracting the input data to calculate probabilities from

X_test = df_X_test.values



# Calculating probabilities of each category using the parameters we calculated earlier

df_y_test['prob_0'] = log_func(X_test, theta_0)

df_y_test['prob_1'] = log_func(X_test, theta_1)

df_y_test['prob_2'] = log_func(X_test, theta_2)



# Fidning the highest value of our probabilites

df_y_test['Pred_label'] = df_y_test[['prob_0','prob_1', 'prob_2']].idxmax(axis=1)

df_y_test['Pred_label'] = df_y_test['Pred_label'].replace({"prob_0": 0,

                                                 "prob_1": 1,

                                                 "prob_2": 2})



print(df_y_test.head())

print('\nPearson R correlation - ', df_y_test['label'].corr(df_y_test['Pred_label']))