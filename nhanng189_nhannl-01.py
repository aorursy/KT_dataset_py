import numpy as np



num = 3



# Tìm định thức của ma trận vuông bậc 3

def det3x3(matrix):

    return matrix[0][0] * matrix[1][1] * matrix[2][2] + matrix[0][1] * matrix[1][2] * matrix[2][0] + matrix[0][2] * matrix[1][0] * matrix[2][1] - matrix[0][2] * matrix[1][1] * matrix[2][0] - matrix[0][0] * matrix[1][2] * matrix[2][1] - matrix[0][1] * matrix[1][0] * matrix[2][2]



# Tìm định thức của ma trận vuông bậc 2

def det2x2(matrix):

    return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]



# Tạo ma trận chuyển vị

def transpose(matrix):

    transposeMatrix = np.ones((num, num))

    for i in range(num):

        for j in range(num):

            transposeMatrix[i][j] = int(matrix[j][i])

    return transposeMatrix



# Tạo ma trận con 2x2 của ma trận 3x3 từ phần tử ở vị trí [row][col]

def getChild2x2(matrix, row, col):    

    numChild = 2

    mapping = [[1,2],[0,2],[0,1]]

    childMatrix = np.ones((numChild, numChild))

    for a in range(numChild):

        for b in range(numChild):

            childMatrix[a][b] = matrix[mapping[row][a]][mapping[col][b]]

    return childMatrix



# Tạo ma trận bổ sung

def adjugate(matrix):

    adjugateMatrix = np.ones((num, num))

    checkerboardMatrix = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])

    for i in range(num):

        for j in range(num):

            childMatrix = getChild2x2(matrix, i, j)

            adjugateMatrix[i][j] = det2x2(childMatrix) * checkerboardMatrix[i][j]

    return adjugateMatrix



# Tạo ma trận nghịch đảo

def inverse(matrix):

    detMatrix = det3x3(matrix)

    print('det(Matrix) =', detMatrix, '\n')

    transposeMatrix = transpose(matrix)

    print('Matrix.T =\n', transposeMatrix, '\n')

    adjugateMatrix = adjugate(transposeMatrix)

    print('Adj(Matrix) =\n', transposeMatrix, '\n')

    

    if detMatrix == 0:

        print('Can not calc Inverse because det(Matrix) = 0\n')

        return

    else:

        inverseMatrix = np.ones((num, num))

        for i in range(num):

            for j in range(num):

                inverseMatrix[i][j] = adjugateMatrix[i][j] / detMatrix

        return inverseMatrix

    

def main():

    matrix = np.random.randint(1, 9, size = (num, num))

    print('Matrix =\n', matrix, '\n')

    inverseMatrix = inverse(matrix)

    print('Matrix^(-1) =\n', inverseMatrix, '\n')

        

main()