from gurobipy import *

import numpy as np



#########Parameters Set-up for a Dedicated System and a Deterministic Demand############



#supply and demand

supply = np.array([100, 100, 100, 100, 100, 100])



demand = np.array([100, 100, 100, 100, 100, 100])



ARCS = tuplelist([(0,0), (1,1), (2,2), (3,3), (4,4), (5,5)])





N = len(supply)

M = len(demand)



print(ARCS)



#########Model Set-up Using Function###############



def model_setup():

    

    m = Model("Process_Flexi")

    

    # number of weeks to offer price level i

    x = m.addVars(ARCS, name = "x")



    # set objective

    m.setObjective( quicksum(x[i,j] for (i,j) in ARCS), GRB.MAXIMIZE)



    # capcity constraint: 

    m.addConstrs( ( quicksum(x[i,j] for (i,j) in ARCS.select(i,'*')) <= supply[i] for i in range(N)), "capacity")



    # demand constraint: 

    m.addConstrs( ( quicksum(x[i,j] for (i,j) in ARCS.select('*',j)) <= demand[j] for j in range(M) ), "demand")

    

    #Supressing the optimization output

    m.setParam( 'OutputFlag', False )

    

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



#m.write("out.lp")
#########Evaluate the Dedicated System for Random Demand############

ARCS = tuplelist([(0,0), (1,1), (2,2), (3,3), (4,4), (5,5)])





#mean of the demand

mean = np.array([100, 100, 100, 100, 100, 100])



#covariance matrix of the demand (independent with s.d. 30)

cov = np.array([[900, 0, 0, 0, 0, 0], 

                [0, 900, 0, 0, 0, 0],

                [0, 0, 900, 0, 0, 0],

                [0, 0, 0, 900, 0, 0],

                [0, 0, 0, 0, 900, 0],

                [0, 0, 0, 0, 0, 900]])



Sample_Size = 1000



sales_dedicate = np.zeros(Sample_Size)





for i in range(Sample_Size):

    

    # demand is sampled from multivariate normal distribution with mean and cov (and truncated above zero)

    demand = np.maximum(np.random.multivariate_normal(mean, cov), 0)

    

    # setup the model again

    m = model_setup()



    # solving the model

    m.optimize()

    

    # store the maximum sales for the i-th sample

    sales_dedicate[i] = m.objVal



# compute the average of maximum sales

avg_sales_dedicate = np.average(sales_dedicate)    



print('Average maximum sales for dedicated system:', avg_sales_dedicate)

# visiualizing the sales over all samples

import matplotlib.pyplot as plt



plt.hist(sales_dedicate, bins = 60, range = (0, 600))

plt.xlabel('Sales')

plt.ylabel('Frequency')

plt.show()
#########Evaluate the Full Flexible System for Random Demand############

ARCS = tuplelist([(0,0), (0,1), (0,2), (0,3), (0,4), (0,5),

                  (1,0), (1,1), (1,2), (1,3), (1,4), (1,5),

                  (2,0), (2,1), (2,2), (2,3), (2,4), (2,5),

                  (3,0), (3,1), (3,2), (3,3), (3,4), (3,5),

                  (4,0), (4,1), (4,2), (4,3), (4,4), (4,5),

                  (5,0), (5,1), (5,2), (5,3), (5,4), (5,5)])





sales_full = np.zeros(Sample_Size)





for i in range(Sample_Size):

    

    # demand is sampled from multivariate normal distribution with mean and cov (and truncated above zero)

    demand = np.maximum(np.random.multivariate_normal(mean, cov), 0)

    

    # setup the model again

    m = model_setup()



    # solving the model

    m.optimize()

    

    # store the maximum sales for the i-th sample

    sales_full[i] = m.objVal



# compute the average of maximum sales

avg_sales_full = np.average(sales_full)    



print('Average maximum sales for full flexible system:', avg_sales_full)



# visiualizing the sales over all samples



plt.hist(sales_full, bins = 60, range = (0, 600))

plt.xlabel('Sales')

plt.ylabel('Frequency')

plt.show()