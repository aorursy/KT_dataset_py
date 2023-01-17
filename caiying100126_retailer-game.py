from gurobipy import *

import numpy as np



#########Parameters Set-up############



#the vector of prices

price = np.array([60, 54, 48, 36])

#the vector of demands

demand = np.array([125, 162.5, 217.5, 348.8])

#salvage value

s = 25

#total number of inventory

I = 2000

#Time horizon

T = 15

#full price week

full_price_week = 1



#number of price levels

N = len(price)
#########Model Set-up###############



m = Model("Retail")



# number of weeks to offer price level i

x = m.addVars(N, name = "x")



# set objective

m.setObjective( quicksum(price[i]*demand[i]*x[i] for i in range(N)) 

               + s*(I - quicksum(demand[i]*x[i] for i in range(N))), GRB.MAXIMIZE)



# capcity constraint: 

m.addConstr( quicksum(demand[i]*x[i] for i in range(N)) <= I , "capacity")



# time constraint: 

m.addConstr( quicksum(x[i] for i in range(N)) <= T , "time")



# full price constraint: 

m.addConstr( x[0] >= full_price_week , "full_price")



# Solving the model

m.optimize()



#  Print optimal solutions and optimal value

print("\n Optimal solution:")

for v in m.getVars():

    print(v.VarName, v.x)



print("\n Optimal profit:")

print('Obj:', m.objVal)



#  Print optimal dual solutions

print("\n Dual solutions:")

for d in m.getConstrs():

    print('%s %g %g' % (d.ConstrName, d.Pi, d.SARHSUp))
I = 2001



# reset objective

m.setObjective( quicksum(price[i]*demand[i]*x[i] for i in range(N)) + 

               s*(I - quicksum(demand[i]*x[i] for i in range(N))), GRB.MAXIMIZE)



#extract the inventory constraint

C_capacity = m.getConstrByName("capacity")



#change the initial inventory level by +1

C_capacity.RHS = I

#alternatively one can use

#C_capacity.setAttr(GRB.Attr.RHS, I)



# Solving the model

m.optimize()



#  Print optimal solutions and optimal value

print("\n Optimal solution:")

for v in m.getVars():

    print(v.VarName, v.x)



print("\n Optimal profit:")

print('Obj:', m.objVal)
I = 2000

#change back to the default capacity constraint

C_capacity.RHS = I



#vector of demand shocks

delta = np.array([-20, 0, 20])



#intitalizing the outputs of optimal solutions

profit_v = np.zeros(len(delta))

x_v = np.zeros((len(delta), len(x)))



#Supressing the optimization output

m.setParam('OutputFlag', False )



for k in range(len(delta)):

    print("\n Shock on demand:", delta[k])

    

    # reset objective

    m.setObjective( quicksum(price[i]*(demand[i]+delta[k])*x[i] for i in range(N)) + 

                   s*(I - quicksum((demand[i]+delta[k])*x[i] for i in range(N))), GRB.MAXIMIZE)

    

    #modify the coefficient of x in the capacity constraint

    for i in range(len(x)):

        m.chgCoeff(C_capacity, x[i], demand[i]+delta[k])

        

    m.optimize()

    

    profit_v[k] = m.objVal

    print("\n Profit:", profit_v[k])

    

    for i in range(len(x)):

        x_v[k,i] = x[i].x

    print("\n Pricing decision:", x_v[k,:])

    
#########Model Set-up Using Function###############

def model_setup():

    

    m = Model("Retail")

    

    # number of weeks to offer price level i

    x = m.addVars(N, name = "x")



    # set objective

    m.setObjective( quicksum(price[i]*demand[i]*x[i] for i in range(N)) + 

                   s*(I - quicksum(demand[i]*x[i] for i in range(N))), GRB.MAXIMIZE)



    # capcity constraint: 

    m.addConstr( quicksum(demand[i]*x[i] for i in range(N)) <= I , "capacity")



    # time constraint: 

    m.addConstr( quicksum(x[i] for i in range(N)) <= T , "time")



    # full price constraint: 

    m.addConstr( x[0] >= full_price_week , "full_price")

    

    return m
# setup the model

m = model_setup()



# Solving the model

m.optimize()



#  Print optimal solutions and optimal value

print("\n Optimal solution:")

for v in m.getVars():

    print(v.VarName, v.x)



print("\n Optimal profit:")

print('Obj:', m.objVal)

#change inventory level

I = 2001



# setup the model again

m = model_setup()



# Solving the model

m.optimize()



#  Print optimal solutions and optimal value

print("\n Optimal solution:")

for v in m.getVars():

    print(v.VarName, v.x)



print("\n Optimal profit:")

print('Obj:', m.objVal)