from gurobipy import *



m = Model("Wyndor")



# Creat variables

# addVar(lb=0.0, ub=GRB.INFINITY, obj=0.0, vtype=GRB.CONTINUOUS, name="", column=None)

# lb: lower bound, ub: upper bound

# vtype: continuous, binary or integer

# name: name for the variable

x1 = m.addVar(name = "x1")

x2 = m.addVar(name = "x2")



# Set objective

# setObjective ( expr, sense=None )

# expr: linear or quadratic expression

# sense: GRB.MINIMIZE or GRB.MAXIMIZE

m.setObjective(3*x1 + 5*x2, GRB.MAXIMIZE)



# Add constraint: 

m.addConstr(x1 <= 4, "Plant1")



m.addConstr(2*x2 <= 12, "Plant2")



m.addConstr(3*x1 + 2*x2 <= 18, "Plant3")



# Solving the model

m.optimize()



#  Print optimal solutions and optimal value



print('-------------------------------------')



for v in m.getVars():

    print(v.VarName, v.x)

    

print('Obj:', m.objVal)



from gurobipy import *

import numpy as np



#########Parameters Set-up############



# Objective coefficient: profit for each product

profit = np.array([3, 5])

print(profit)

print(type(profit))

print(profit.shape)

# Constraint right-hand-side: capacity for each plant

capacity = np.array([4, 12, 18])



# A matrix: the consumption of capacity at plant i of product j

rate = np.array([[1, 0],

                 [0, 2], 

                 [3, 2]])



print(rate)

print(rate.shape)
# From A matrix, extract the number of products: N and the number of plants: M

M, N = rate.shape



print("M = %g,  N = %g" % (M, N) )
#########Model Set-up###############



m = Model("Wyndor")



# Creat variables

# addVars ( *indices, lb=0.0, ub=GRB.INFINITY, obj=0.0, vtype=GRB.CONTINUOUS, name="" )

# indices: can be one or more integer values, e.g.,  x = model.addVars(2), which creates x[0], x[1]; 

#          e.g., x = model.addVars(2, 3), which creates x[0,0],x[0,1],x[0,2],x[1,0],x[1,1],x[1,2];

#          can be a list of tuples, e.g., x = model.addVars([(0,0), (1,1), (2,2)]), which creates x[0,0],x[1,1],x[2,2].



x = m.addVars(N, name = "x")



# Set objective

m.setObjective( quicksum(profit[i]*x[i] for i in range(N)), GRB.MAXIMIZE)



# Add constraints: 

m.addConstrs(( quicksum(rate[i,j]*x[j] for j in range(N)) <= capacity[i] for i in range(M) ), "Plant")



# Solving the model

m.optimize()



# Print optimal solutions and optimal value

print('-------------------------------------')



for v in m.getVars():

    print(v.VarName, v.x)

    

print('Obj:', m.objVal)



from gurobipy import *

import numpy as np

import pandas as pd





#########Parameters Set-up############

# Read the data from the csv file and use the first column as the index of rows

Wyndor_data = pd.read_csv('Wyndor.csv', index_col = 0)

print(Wyndor_data)

print(Wyndor_data.shape)



# Record the number of rows and columns in the data

temp_M, temp_N = Wyndor_data.shape

#One is referred to http://pandas.pydata.org/pandas-docs/stable/indexing.html 

#for indexing and selecting data for the dataframe object in pandas.



# Extracting objective coefficient via selection by position: .iloc

# When slicing the data, the start bound is included, while the end bound is excluded

profit = Wyndor_data.iloc[0, 0:temp_N-1]

print(profit) 

print('Index of the data:')

print(profit.index) 

print('Values of the data:')

print(profit.values)



#Extracting the values by ignoring the index and header of the dataframe

profit = profit.values
#Constraint right-hand-side: capacity for each plant

capacity = Wyndor_data.iloc[1:temp_M, temp_N-1]

capacity = capacity.values



#A matrix: the consumption of capacity at plant i of product j

rate = Wyndor_data.iloc[1:temp_M, 0:temp_N-1]

print(rate)

rate = rate.values

print(rate)



#From A matrix, extract the number of products: N and the number of plants: M

M, N = rate.shape

#########Model Set-up###############



m = Model("Wyndor")



# Creat variables

# addVars ( *indices, lb=0.0, ub=GRB.INFINITY, obj=0.0, vtype=GRB.CONTINUOUS, name="" )

x = m.addVars(N, name = 'Product')



# Set objective

m.setObjective( quicksum(profit[i]*x[i] for i in range(N)), GRB.MAXIMIZE)



# Add constraints: 

m.addConstrs(( quicksum(rate[i,j]*x[j] for j in range(N)) <= capacity[i] for i in range(M) ), name = "Plant")



# Solving the model

m.optimize()



#  Print optimal solutions and optimal value

for v in m.getVars():

    print(v.VarName, v.x)

    

print('Obj:', m.objVal)

#Lecture 4: Print sensitivity information



print("\n Sensitivity information:")

for d in m.getConstrs():

    print(d.ConstrName, d.Pi, d.SARHSUp, d.SARHSLow)

    
