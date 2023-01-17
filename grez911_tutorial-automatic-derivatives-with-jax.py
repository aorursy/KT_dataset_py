import numpy as np



def predict(X, w, b):

    """Make a prediction.

    

    Args:

        X: a features matrix.

        w: weights (a column vector).

        b: a bias.

      

    Returns:

        vector: a prediction with the same dimensions as a target column vector (n by 1).

    """

    

    # .dot() is a matrix multiplication in Numpy.

    # We can ommit all-ones vector because Numpy can add a scalar to a vector directly.

    return X.dot(w) + b



def J(y_hat, y):

    """Calculate a cost of this solution.

    

    Args:

        y_hat: a prediction vector.

        y: a target vector.

    

    Returns:

        scalar: a cost of this solution.

    """

    # **2 - raise to the power of two.

    # .mean() - calculate a mean value of vector elements.

    return ((y_hat - y)**2).mean()



def dw(X, y_hat, y):

    """Calculate a partial derivative of J with respect to w.

    

    Args:

        X: a features matrix.

        y_hat: a prediction vector.

        y: a target vector.

      

    Returns:

        vector: a partial derivative of J with respect to w.

    """

    # .transpose() - transpose matrix.

    return 2 * X.transpose().dot(y_hat - y) / len(y)



def db(y_hat, y):

    """Calculate a partial derivative of J with respect to b.

    

    Args:

        y_hat: a prediction vector.

        y: a target vector.

    

    Returns:

        vector: a partial derivative of J with respect to b.

    """

    return 2 * (y_hat - y).mean()
# A features matrix.

X = np.array([

                 [4, 7],

                 [1, 8],

                 [-5, -6],

                 [3, -1],

                 [0, 9]

             ])



# A target column vector.

y = np.array([

                 [37],

                 [24],

                 [-34], 

                 [16],

                 [21]

             ])



# Initialize weights and a bias with some random values.

w = np.array([ [2], [-17] ])

b = 6
y_hat = predict(X, w, b)

y_hat
dw(X, y_hat, y)
db(y_hat, y)
!pip install jax jaxlib
%env JAX_PLATFORM_NAME=cpu
import jax.numpy as np

from jax import grad
# A features matrix.

X = np.array([

                 [4., 7.],

                 [1., 8.],

                 [-5., -6.],

                 [3., -1.],

                 [0., 9.]

             ])



# A target column vector.

y = np.array([

                 [37.],

                 [24.],

                 [-34.], 

                 [16.],

                 [21.]

             ])



# Initialize weights and a bias with some random values.

w = np.array([ [2.], [-17.] ])

b = 6.
y_hat = predict(X, w, b)

y_hat
def predict(X, w, b):

    """Make a prediction.

    

    Args:

        X: a features matrix.

        w: weights (a column vector).

        b: a bias.

      

    Returns:

        vector: a prediction with the same dimensions as a target column vector (n by 1).

    """



    # .dot() is a matrix multiplication in Numpy.

    # We can ommit all-ones vector because Numpy can add a scalar to a vector directly.

    return X.dot(w) + b



def J(y_hat, y):

    """Calculate a cost of this solution.

    

    Args:

        y_hat: a prediction vector.

        y: a target vector.

    

    Returns:

        scalar: a cost of this solution.

    """

    # **2 - raise to the power of two.

    # .mean() - calculate a mean value of vector elements.

    return ((y_hat - y)**2).mean()
def J_combined(X, w, b, y):

    """Cost function combining predict() and J() functions.



    Args:

        X: a features matrix.

        w: weights (a column vector).

        b: a bias.

        y: a target vector.



    Returns:

        scalar: a cost of this solution.    

    """

    y_hat = predict(X, w, b)

    return J(y_hat, y)
grad(J_combined, argnums=1)(X, w, b, y)
grad(J_combined, argnums=2)(X, w, b, y)