import numpy as np



def predict(X, w, b):

    """Make a prediction according to the equation number 7.

    

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

    """Calculate a cost of this solution (equation 8).

    

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

    """Calculate a partial derivative of J with respect to w (equation 9).

    

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

    """Calculate a partial derivative of J with respect to b (equation 10).

    

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



# Initialize weights and bias with zeros.

w = np.zeros((X.shape[1], 1))

b = 0



# How much gradient descent steps we will perform.

num_epochs = 50



# A learning rate.

alpha = 0.01



# Here will be stored J for each epoch.

J_array = []



for epoch in range(num_epochs):

    # Equation 7.

    y_hat = predict(X, w, b)



    # Equation 8.

    J_array.append(J(y_hat, y))

    

    # Equation 11.

    w = w - alpha * dw(X, y_hat, y)

    

    # Equation 12.

    # b converges slower than w, so we increased alpha for it by a factor of 10. It's not mandatory though.

    b = b - alpha * db(y_hat, y) * 10
import matplotlib.pyplot as plt



plt.plot(J_array)

plt.xlabel('epoch')

plt.ylabel('J')

plt.show()



# {:.3} - round to three significant figures for f-string.

print(f"w1 = {w[0][0]:.3}")

print(f"w2 = {w[1][0]:.3}")

print(f"b = {b:.3}")

print(f"J = {J_array[-1]:.3}")