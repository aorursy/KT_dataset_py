from gurobipy import *

import numpy as np



#########Parameters Set-up############



#Objective coefficient: transportation cost from supply node i to demand node j

cost = np.array([[8, 6, 10, 9],

                [9, 12, 13, 7], 

                [14, 9, 16, 5]])





#supply and demand

supply = np.array([35, 50, 40])



demand = np.array([45, 20, 30, 30])



#From the cost matrix, extract the number of supply nodes: M and the number of demand nodes: N

M, N = cost.shape







#########Model Set-up###############



tp = Model("transportation")



# Creat variables

# addVars ( *indices, lb=0.0, ub=GRB.INFINITY, obj=0.0, vtype=GRB.CONTINUOUS, name="" )

x = tp.addVars(M, N)



# Set objective

tp.setObjective( quicksum(cost[i,j]*x[i,j] for i in range(M) for j in range(N)), GRB.MINIMIZE)



# Add supply constraints: 

tp.addConstrs(( quicksum(x[i,j] for j in range(N)) == supply[i] for i in range(M) ), "Supply")



# Add demand constraints: 

tp.addConstrs(( quicksum(x[i,j] for i in range(M)) == demand[j] for j in range(N) ), "Demand")



# Solving the model

tp.optimize()



#  Print optimal solutions and optimal value

for i in range(M):

    for j in range(N):

        print("\n Supply node %g to demand node %g amount: %g" % (i+1, j+1 , x[i,j].x))

    

print('Obj:', tp.objVal)
