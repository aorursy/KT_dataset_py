import numpy as np



a1 = np.random.rand(3,3)

print(a1)

#a1 = [[1,2,3],

#      [4,5,6],

#      [7,8,9]]



def __getSubMatrix(matx, exceptedRow, exceptedCol):

    resultMatx = []

    for j in range(len(matx)):

        newRow = []

        if j != exceptedRow:

            for i in range(len(matx[j])):

                if i != exceptedCol:

                    newRow.append(matx[j][i])

            resultMatx.append(newRow)

    return resultMatx



# Determinant if a matrix: https://en.wikipedia.org/wiki/Determinant

def _det(matx): 

    #base-case

    if len(matx) == 2:

        return matx[0][0]*matx[1][1] - matx[0][1]*matx[1][0]

    calcDet = 0

    for col in range(len(matx)):

        calcDet += ((-1)**col)*matx[0][col]*_det(__getSubMatrix(matx,0,col))

    return calcDet



def __transposeMatx(matx):

    resultMatx = np.zeros((len(matx),len(matx[0])))

    #resultMatx = matx

    for row in range(len(resultMatx)):

        for col in range(len(resultMatx[row])):

            resultMatx[row][col] = matx[col, row]

    return resultMatx



def __cofactorMatx(matx):

    calcMatx = np.zeros((len(matx),len(matx[0])))

    for row in range(len(calcMatx)):

        for col in range(len(calcMatx[row])):

            calcMatx[row][col] = _det(__getSubMatrix(matx, row, col))

    for row in range(len(calcMatx)):                                     # [ + - + ]

        for col in range(len(calcMatx[row])):                            # [ - + - ]

            calcMatx[row][col] = ((-1)**(col + row))*calcMatx[row][col]  # [ + - + ]

    return calcMatx



def _adj(matx):

    calcMatx = __transposeMatx(matx)

    calcMatx = __cofactorMatx(matx)

    return calcMatx



def _inv(matx):

    #inverse Matrix: https://www.wikihow.com/Find-the-Inverse-of-a-3x3-Matrix

    detMatxValue = _det(matx)

    if detMatxValue == 0:

        return "matrix does not have inverse"

    

    calcMatx = _adj(matx)

    for row in range(len(calcMatx)):

        for col in range(len(calcMatx[row])):

            calcMatx[row][col] = calcMatx[row][col]/detMatxValue



    return calcMatx

    

     

print('det = ', _det(a1))

print('inverse = ')

print(_inv(a1))