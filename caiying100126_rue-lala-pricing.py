from gurobipy import *

import numpy as np





#########Parameters Set-up############

# For convenience demands are put 0 for impossible scenarios, for example, 

# the price for product is 39.99 but the total prices is 3x59.99, i.e., the cell demand[0,0,12] below.





demand = np.array([   [[30, 32, 33, 35, 38, 40, 44, 50, 50,  0,  0,  0,  0],

                       [ 0, 29, 31, 32, 30, 34, 38, 40, 42, 44,  0,  0,  0],

                       [ 0,  0, 25, 29, 28, 28, 31, 33, 35, 36, 38,  0,  0],

                       [ 0,  0,  0, 10, 18, 18, 20, 21, 24, 26, 26, 27,  0],

                       [ 0,  0,  0,  0,  2,  4,  4,  6,  8, 10, 12, 15, 16]],

                   

                      [[60, 65, 68, 70, 73, 76, 78, 82, 83,  0,  0,  0,  0],

                       [ 0, 50, 52, 53, 55, 57, 59, 60, 64, 65,  0,  0,  0],

                       [ 0,  0, 28, 35, 37, 40, 42, 43, 43, 44, 45,  0,  0],

                       [ 0,  0,  0,  7,  9,  9, 10, 12, 12, 14, 14, 14,  0],

                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  2,  2,  2]],

                   

                      [[20, 20, 20, 21, 21, 22, 22, 24, 24,  0,  0,  0,  0],

                       [ 0, 20, 20, 20, 21, 21, 21, 22, 22, 22,  0,  0,  0],

                       [ 0,  0, 19, 19, 20, 20, 21, 22, 22, 22, 23,  0,  0],

                       [ 0,  0,  0, 17, 18, 18, 20, 20, 20, 20, 20, 20,  0],

                       [ 0,  0,  0,  0, 15, 15, 15, 16, 18, 16, 17, 17, 18]]      ])



# number of product, number of price points, number of total price points

N, M, K = demand.shape



# minimum and maximum price point

min_price = 39.99

max_price = 59.99



# vector of prices

price_v = np.linspace(min_price, max_price, num = M)



# vector of all possible total prices

total_price_v = np.linspace(N*min_price, N*max_price, num = K)



# initialize the index of the total price as the mimnimum one

k = 0



print(price_v)

print(total_price_v)
#########Model Set-up############



def model_setup():

    

    m = Model("Ruelala")



    # Creat variables

    x = m.addVars(N, M, vtype=GRB.BINARY, name = "x")

    

    # set objective

    m.setObjective( quicksum(price_v[m]*demand[n,m,k]*x[n,m] for n in range(N) for m in range(M)), GRB.MAXIMIZE)

    

    # for each product, only one price point can be chosen:

    m.addConstrs( ( quicksum(x[n,m] for m in range(M)) == 1 for n in range(N))  )



    # total price constraint:

    m.addConstr( quicksum(price_v[m]*x[n,m] for n in range(N) for m in range(M)) == total_price_v[k]  ) 

    

    #Supressing the optimization output

    m.setParam( 'OutputFlag', False )

    

    return m





#########Solve the Model for Each Possible Total Price############



# initialize the vector of profits

profit_v = np.zeros(K)



# initialize the vector of optimal prices

opt_price_v = np.zeros( (K, N) )



for k in range(K):

    

    # setup the model 

    m_rll = model_setup()



    # solving the model

    m_rll.optimize()

    

    # storing the corresponding information

    profit_v[k] = m_rll.objVal

    

    # extract the variables from the model. NOTE: variables extracted in this way are automatically formatted as a vector

    x = m_rll.getVars()

    

    # reformat the vector as a matrix with dimension NxM

    x = np.reshape(x, (N,M))

    

    for n in range(N):

        for m in range(M):

            if x[n,m].x == 1:

                opt_price_v[k,n] = price_v[m]

    



# find the total price that maximizes the profit



k_max = np.argmax(profit_v)



print("The maximum profit is: %g" % profit_v[k_max])



print("The optimal price is: ", opt_price_v[k_max,:])











print(profit_v)
print(opt_price_v)