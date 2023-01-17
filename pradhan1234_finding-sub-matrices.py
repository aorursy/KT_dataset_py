import numpy as np

print("hello")
A = np.array([[1,1,1,1,1,1],

[1,0,0,0,1,1],

[1,0,0,0,1,0],

[1,0,0,0,1,0],

[0,1,1,1,1,1]])
def getBottomRightCoordinates(startrow, startcol, nrow, ncol):

    f1 = False

    f2 = False

    for i in range(startrow, nrow):

        if(A[i][startcol] == 1):

            f1 = True

            break

        for j in range(startcol, ncol):

            if(A[i][j] == 1):

                ncol = j + 1

                f2 = True

                break

            A[i][j] = 2

    

    if(f1 or i == nrow):

        endrow = i-1

    else:

        print("YO in I")

        endrow = i

    if(f2 or j == ncol):

        endcol = j-1

    else:

        print("YO in J")

        endcol = j

                

    return [endrow, endcol]
A.shape
nrow = A.shape[0]

ncol = A.shape[1]



startCoordinates = []

endCoordinates = []
for i in range(nrow):

    for j in range(ncol):

        if(A[i][j] == 0):

            print("Start Coordinates: (",i,",",j,")")

            startCoordinates.append([i,j])

            endCoordinates.append(getBottomRightCoordinates(i, j, nrow, ncol))
startCoordinates
endCoordinates