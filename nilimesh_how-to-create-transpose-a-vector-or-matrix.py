# ============================================================================

# End-to-End Applied Data Science Recipes @ https://wacamlds.podia.com

# https://wacamlds.podia.com

# ============================================================================



## How to Create & Transpose a Vector or Matrix

def Kickstarter_Example_1():

    print()

    print(format('How to Create/Transpose a Vector and/or Matrix', '*^75'))

    

    # Load library

    import numpy as np

    

    # Create vector

    vector = np.array([1, 2, 3, 4, 5, 6])

    print()

    print("Original Vector: \n", vector)

    # Tranpose vector

    V = vector.T

    print("Transpose Vector: \n", V)

    

    # Create matrix

    matrix = np.array([[1, 2, 3],

                   [4, 5, 6],

                   [7, 8, 9]])

    print()

    print("Original Matrix: \n", matrix)    

    # Transpose matrix

    M = matrix.T    

    print("Transpose Matrix: \n", M)

Kickstarter_Example_1()