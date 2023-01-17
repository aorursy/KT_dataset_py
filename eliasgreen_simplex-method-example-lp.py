import numpy as np
# User Defined Input



# Example Input

A = np.array([[-2, 1, 1, 0, 0],

             [-1, 2, 0, 1, 0],

             [1, 0, 0, 0, 1]])



b = np.array([2, 7, 3])



c = np.array([-1, -2, 0, 0, 0])
def Simplex(A, b, c):

    '''Takes input vars, computs corresponding values,

    then uses while loop to iterate until a basic optimal solution is reached.

    RETURNS: cbT, cbIndx, cnT, cnIndx, bHat, cnHat.

    cbT, cbIndex is final basic variable values, and indices

    cnT, cnIndex is final nonbasic variable values and indices

    bHat is final solution values, 

    cnHat is optimality condition'''

    

    #sizes of basic and nonbasic vectors

    basicSize = A.shape[0] # number of constraints, m

    nonbasicSize = A.shape[1] - basicSize #n-m, number of variables

        

    # global index tracker of variables of basic and nonbasic variables (objective)

    # that is, index 0 corresponds with x_0, 1 with x_1 and so on.  So each index corresponds with a variable

    cindx = [i for i in range(0, len(c))]

    

    #basic variable coefficients

    cbT = np.array(c[nonbasicSize:])



    #nonbasic variable coefficients

    cnT = np.array(c[:nonbasicSize])

    

    # run core simplex method until reach the optimal solution

    while True:

        

        # keep track of current indices of basic and non-basic variables

        cbIndx = cindx[nonbasicSize:]

        cnIndx = cindx[:nonbasicSize]

        

        # basis matrix

        B = A[:, cbIndx]

        Binv = np.linalg.inv(B)

        

        # nonbasic variable matrix

        N = A[:, cnIndx]

        

        # bHat, the values of the basic variables

        # recall that at the start the basic variables are the slack variables, and 

        # have values equal the vector b (as primary variables are set to 0 at the start)

        bHat = Binv @ b

        yT = cbT @ Binv

        

        # use to check for optimality, determine variable to enter basis

        cnHat = cnT - (yT @ N)

        

        # find indx of minimum value of cnhat, this is the variable to enter the basis

        cnMinIndx = np.argmin(cnHat)



        # break out of loop, returning values if all values of cnhat are above 0

        if(all(i>=0 for i in cnHat)):

            # use cbIndx to get index values of variables in bHat, and the corresponding index

            # values in bHat are the final solution values for each of the corresponding variables

            # ie value 0 in dbIndx corresponds with first variable, so whatever the index for the 0 is

            # is the index in bHat that has the solution value for that variable.

            return cbT, cbIndx, cnT, cnIndx, bHat, cnHat

        

        # this is the index for the column of coeffs in a for the given variable

        indx = cindx[cnMinIndx]



        Ahat = Binv @ A[:, indx]

        

        # now we want to iterate through Ahat and bHat and pick the minimum ratios

        # only take ratios of variables with Ahat_i values greater than 0

        # pick smallest ratio to get variable that will become nonbasic.

        ratios = []

        for i in range(0, len(bHat)):

            Aval = Ahat[i]

            Bval = bHat[i]



            # don't look at ratios with val less then or eqaul to 0, append to keep index

            if(Aval <= 0):

                ratios.append(10000000)

                continue

            ratios.append(Bval / Aval)



        ratioMinIndx = np.argmin(ratios)



        #switch basic and nonbasic variables using the indices.

        cnT[cnMinIndx], cbT[ratioMinIndx] = cbT[ratioMinIndx], cnT[cnMinIndx]

        # switch global index tracker indices

        cindx[cnMinIndx], cindx[ratioMinIndx + nonbasicSize] = cindx[ratioMinIndx + nonbasicSize], cindx[cnMinIndx]

        # now repeat the loop

        



Simplex(A, b, c)
# example test

A = np.array([[2, 1, 1, 0, 0],

             [2, 3, 0, 1, 0],

             [3, 1, 0, 0, 1]])

c = np.array([-3, -2, 0, 0, 0])

b = np.array([18, 42, 24])



Simplex(A, b, c)
# another example test

A = np.array([[1, 1, 1, 1, 0, 0],

            [-1, 2, -2, 0, 1, 0],

            [2, 1, 0, 0, 0, 1]])



b = np.array([4, 6, 5])

c = np.array([-1, -2, 1, 0, 0, 0])



Simplex(A, b, c)