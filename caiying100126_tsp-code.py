from gurobipy import *

import numpy as np



#########Parameters Set-up############



#traveling cost from node i to node j

cost = np.array([[1000, 4, 2, 1000, 1000, 1000],

                 [0, 1000, 6, 1000, 1, 7], 

                 [0, 6, 1000, 2, 1, 1000],

                 [0, 1000, 2, 1000, 1, 3],

                 [0, 1, 1, 1, 1000, 1], 

                 [0, 7, 1000, 3, 1, 1000]])



N = cost.shape[0]



#the big M

M = 10000
#########Model Set-up############





tsp = Model("traveling_salesman")



# Creat variables

x = tsp.addVars(N, N, vtype=GRB.BINARY, name = "x")



u = tsp.addVars(N, name = "u")



# Set objective

tsp.setObjective( quicksum(cost[i,j]*x[i,j] for i in range(N) for j in range(N)), GRB.MINIMIZE)



# Assignment constraints:

tsp.addConstrs(( quicksum(x[i,j] for j in range(N)) == 1 for i in range(N) ))

 

tsp.addConstrs(( quicksum(x[i,j] for i in range(N)) == 1 for j in range(N) ))



# Subtour-breaking constraints:

tsp.addConstrs(( u[i] + 1 - u[j] <= M*(1 - x[i,j])  for i in range(N) for j in range(1,N) ))





# Solving the model

tsp.optimize()

#  Print optimal x for x nonzero and optimal value

s_edge = []

for v in x:    

    if x[v].x > 0.001:

        print(x[v].VarName, x[v].x)

        #add both of the indicies by 1

        edge = np.add(v, (1,1))

        #append the edge to the resulting list of edges

        s_edge.append(edge)





print('Obj:', tsp.objVal)

print(s_edge)

for v in u: 

    print(u[v].VarName, u[v].x)
#  Obtain the permutation as a representation of the tour



permu = np.ones(N)

predecessor = 1

for i in range(N):

    for s in s_edge:

        if s[0] == predecessor:

            permu[i] = s[0]

            predecessor = s[1]

            break    

    

print(permu)
# data for the precedent pair



Precedent_Pair = tuplelist([(0,2), (1,3), (4,5)])





tsp.addConstrs( (u[i] <= u[j] for (i,j) in Precedent_Pair)  )





# Solving the new model

tsp.optimize()









#  The list of edges traversed

s_edge = []

for v in x:    

    if x[v].x > 0.001:

        edge = np.add(v, (1,1))

        s_edge.append(edge)

        

#  Obtain the permutation as a representation of the tour

permu = np.ones(N)

predecessor = 1

for i in range(N):

    for s in s_edge:

        if s[0] == predecessor:

            permu[i] = s[0]

            predecessor = s[1]

            break    

    

print(permu)