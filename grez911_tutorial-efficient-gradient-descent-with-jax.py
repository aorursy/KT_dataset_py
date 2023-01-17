!pip -q install jax jaxlib



%env JAX_ENABLE_X64=1

%env JAX_PLATFORM_NAME=cpu



import jax.numpy as np

from jax import grad, jit
def J(X, w, b, y):

    """Cost function for a linear regression. A forward pass of our model.



    Args:

        X: a features matrix.

        w: weights (a column vector).

        b: a bias.

        y: a target vector.



    Returns:

        scalar: a cost of this solution.    

    """

    y_hat = X.dot(w) + b # Predict values.

    return ((y_hat - y)**2).mean() # Return cost.
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



learning_rate = 0.01
w = np.zeros((2, 1))

b = 0.
%timeit grad(J, argnums=1)(X, w, b, y)
%timeit grad(J, argnums=2)(X, w, b, y)
for i in range(100):

    w -= learning_rate * grad(J, argnums=1)(X, w, b, y)

    b -= learning_rate * grad(J, argnums=2)(X, w, b, y)

    

    if i % 10 == 0:

        print(J(X, w, b, y))
w = np.zeros((2, 1))

b = 0.
grad_X = jit(grad(J, argnums=1))

grad_b = jit(grad(J, argnums=2))



# Run once to trigger JIT compilation.

grad_X(X, w, b, y)

grad_b(X, w, b, y)
%timeit grad_X(X, w, b, y)
%timeit grad_b(X, w, b, y)
for i in range(100):

    w -= learning_rate * grad_X(X, w, b, y)

    b -= learning_rate * grad_b(X, w, b, y)

    

    if i % 10 == 0:

        print(J(X, w, b, y))