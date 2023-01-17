from gurobipy import *

import numpy as np





#########Parameters Set-up############



y = np.array([[8, 0, 0, 6, 0, 0, 9, 0, 5],

              [0, 0, 0, 0, 0, 0, 0, 0, 0],

              [0, 0, 0, 0, 2, 0, 3, 1, 0],

              [0, 0, 7, 3, 1, 8, 0, 6, 0],

              [2, 4, 0, 0, 0, 0, 0, 7, 3],

              [0, 0, 0, 0, 0, 0, 0, 0, 0],

              [0, 0, 2, 7, 9, 0, 1, 0, 0],

              [5, 0, 0, 0, 8, 0, 0, 3, 6], 

              [0, 0, 3, 0, 0, 0, 0, 0, 0]])



print(y)



N = y.shape[0]

#########Model Set-up############



m = Model("Sudoku")



# Creat variables

x = m.addVars(N, N, N, vtype=GRB.BINARY, name = "x")



# Set objective

m.setObjective(0, GRB.MINIMIZE)



# Fill-in constraints:

for i in range(N):

    for j in range(N):

        if y[i,j] > 0:

            m.addConstr( x[i,j, y[i,j]-1] == 1 )



# For every column, each digit appears only once:

m.addConstrs(( quicksum(x[i,j,k] for i in range(N)) == 1  for j in range(N) for k in range(N) ))



# For every row, each digit appears only once:

m.addConstrs(( quicksum(x[i,j,k] for j in range(N)) == 1  for i in range(N) for k in range(N) ))



# For every entry (i,j), only one digit is chosen:

m.addConstrs(( quicksum(x[i,j,k] for k in range(N)) == 1  for i in range(N) for j in range(N) ))



# For 3x3 square, each digit appear only once

m.addConstrs(( quicksum(x[i,j,k] for i in range(3*p, 3*(p+1)) for j in range(3*q, 3*(q+1))) == 1

              for k in range(N) for p in range(3) for q in range(3) ))







# Solving the model

m.optimize()





# Print out the solution to the Sudoku problem

for i in range(N):

    for j in range(N):

        for k in range(N):

            if x[i,j,k].x == 1:

                print("%3i" % (k+1), end =" ") 

    print("\n")

    

    