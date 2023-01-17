## Gradient Descent from scratch



#Importing the dataset

import pandas as pd

import seaborn as sns



housing = pd.read_csv('../input/Housing (2).csv')

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
housing.info()
# Normalizisng the data

housing = (housing - housing.mean())/housing.std()

housing.head()
# Simple linear regression

# Assign feature variable X

X = housing['area']



# Assign response variable to y

y = housing['price']
# Visualise the relationship between the features and the response using scatterplots

sns.pairplot(housing, x_vars='area', y_vars='price',size=7, aspect=0.7, kind='scatter')
import numpy as np

X = np.array(X)

y = np.array(y)
def before_gradient(X, y, m_current=0, c_current=0, iters=1000, learning_rate=0.01):

    N = float(len(y))

    #gd_df = pd.DataFrame( columns = ['m_current', 'c_current','cost'])

    #for i in range(iters):

    y_current = (m_current * X) + c_current

    cost = sum([data**2 for data in (y-y_current)]) / N



    return cost

        

cost_gradients=before_gradient(X,y)

cost_gradients
# Implement gradient descent function

# Takes in X, y, current m and c (both initialised to 0), num_iterations, learning rate

# returns gradient at current m and c for each pair of m and c



def gradient(X, y, m_current=0, c_current=0, iters=500, learning_rate=0.01):

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