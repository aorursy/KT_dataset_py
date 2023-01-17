#Importing the dataset

import pandas as pd

housing = pd.read_csv('../input/housing/Housing.csv')

housing.head()
# Converting Yes to 1 and No to 0

housing['mainroad'] = housing['mainroad'].map({'yes': 1, 'no': 0})

housing['guestroom'] = housing['guestroom'].map({'yes': 1, 'no': 0})

housing['basement'] = housing['basement'].map({'yes': 1, 'no': 0})

housing['hotwaterheating'] = housing['hotwaterheating'].map({'yes': 1, 'no': 0})

housing['airconditioning'] = housing['airconditioning'].map({'yes': 1, 'no': 0})

housing['prefarea'] = housing['prefarea'].map({'yes': 1, 'no': 0})
#Converting furnishingstatus column to binary column using get_dummies

status = pd.get_dummies(housing['furnishingstatus'],drop_first=True)

housing = pd.concat([housing,status],axis=1)

housing.drop(['furnishingstatus'],axis=1,inplace=True)
housing.head()
# Normalisisng the data

housing = (housing - housing.mean())/housing.std()

housing.head()
# Simple linear regression

# Assign feature variable X

X = housing['area']



# Assign response variable to y

y = housing['price']
# Conventional way to import seaborn

import seaborn as sns



# To visualise in the notebook

%matplotlib inline
# Visualise the relationship between the features and the response using scatterplots

sns.pairplot(housing, x_vars='area', y_vars='price',size=7, aspect=0.7, kind='scatter')
import numpy as np

X = np.array(X)

y = np.array(y)
# Implement gradient descent function

# Takes in X, y, current m and c (both initialised to 0), num_iterations, learning rate

# returns gradient at current m and c for each pair of m and c



def gradient(X, y, m_current=0, c_current=0, iters=1000, learning_rate=0.01):

    N = float(len(y))

    gd_df = pd.DataFrame( columns = ['m_current', 'c_current','cost'])

    for i in range(iters):

        y_current = (m_current * X) + c_current

        cost = sum([data**2 for data in (y-y_current)]) / N

        m_gradient = -(2/N) * sum(X * (y - y_current))

        c_gradient = -(2/N) * sum(y - y_current)

        m_current = m_current - (learning_rate * m_gradient)

        c_current = c_current - (learning_rate * c_gradient)

        gd_df.loc[i] = [m_current,c_current,cost]

    return(gd_df)

# print gradients at multiple (m, c) pairs

# notice that gradient decreased gradually towards 0

# we have used 1000 iterations, can use more if needed

gradients = gradient(X,y)

gradients
# plotting cost against num_iterations

gradients.reset_index().plot.line(x='index', y=['cost'])
# Assigning feature variable X

X = housing[['area','bedrooms']]



# Assigning response variable y

y = housing['price']
# Add a columns of 1s as an intercept to X.

# The intercept column is needed for convenient matrix representation of cost function



X['intercept'] = 1

X = X.reindex_axis(['intercept','area','bedrooms'], axis=1)

X.head()
# Convert X and y to arrays

import numpy as np

X = np.array(X)

y = np.array(y)
# Theta is the vector representing coefficients (intercept, area, bedrooms)

theta = np.matrix(np.array([0,0,0])) 

alpha = 0.01

iterations = 1000
# define cost function

# takes in theta (current values of coefficients b0, b1, b2), X and y

# returns total cost at current b0, b1, b2



def compute_cost(X, y, theta):

    return np.sum(np.square(np.matmul(X, theta) - y)) / (2 * len(y))
# gradient descent

# takes in current X, y, learning rate alpha, num_iters

# returns cost (notice it uses the cost function defined above)



def gradient_descent_multi(X, y, theta, alpha, iterations):

    theta = np.zeros(X.shape[1])

    m = len(X)

    gdm_df = pd.DataFrame( columns = ['Bets','cost'])



    for i in range(iterations):

        gradient = (1/m) * np.matmul(X.T, np.matmul(X, theta) - y)

        theta = theta - alpha * gradient

        cost = compute_cost(X, y, theta)

        gdm_df.loc[i] = [theta,cost]



    return gdm_df
# print costs with various values of coefficients b0, b1, b2

gradient_descent_multi(X, y, theta, alpha, iterations)
# print cost

gradient_descent_multi(X, y, theta, alpha, iterations).reset_index().plot.line(x='index', y=['cost'])