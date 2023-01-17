# ============================================================================

# End-to-End Applied Data Science Recipes @ https://wacamlds.podia.com

# https://wacamlds.podia.com

# ============================================================================



## How to Create A Sparse Matrix

def Kickstarter_Example_2():

    print()

    print(format('How to Create A Sparse Matrix', '*^50'))

    

    # Load libraries

    import numpy as np

    from scipy import sparse

    

    # Create a matrix

    matrix = np.array([[1, 2, 3],

                   [4, 5, 6],

                   [7, 8, 9]])

    print()

    print("Original Matrix: \n", matrix) 



    # Create sparse matrices

    print()

    print("Sparse Matrices: ")     

    print()

    print(sparse.csr_matrix(matrix))

    print()

    print(sparse.bsr_matrix(matrix))

    print()

    print(sparse.coo_matrix(matrix))

    print()

    print(sparse.csc_matrix(matrix))

    print()

    print(sparse.dia_matrix(matrix))

    print()

    print(sparse.dok_matrix(matrix))

    print()

    print(sparse.lil_matrix(matrix))

    print()

Kickstarter_Example_2()