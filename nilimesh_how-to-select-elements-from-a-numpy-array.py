# ============================================================================

# End-to-End Applied Data Science Recipes @ https://wacamlds.podia.com

# https://wacamlds.podia.com

# ============================================================================



## How to Select Elements from a Numpy Array

def Kickstarter_Example_3():

    print()

    print(format('How to Select Elements from Numpy Array', '*^52'))    

    

    # Load library

    import numpy as np



    # Create row vector

    vector = np.array([1, 2, 3, 4, 5, 6])

    # Select second element

    print()

    print(vector[1])



    # Create matrix

    matrix = np.array([[1, 2, 3],

                   [4, 5, 6],

                   [7, 8, 9]])

    # Select second row, second column

    print()

    print(matrix[1,1])



    # Create Tensor

    tensor = np.array([

                    [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],

                    [[[3, 3], [3, 3]], [[4, 4], [4, 4]]]

                  ])

    # Select second element of each of the three dimensions

    print()

    print(tensor[1,1,1])

Kickstarter_Example_3()  