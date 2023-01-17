from gurobipy import *

import numpy as np

import pandas as pd

#########Parameters Set-up############



#read dataset



dataset=pd.read_csv('nutrition_1027.csv', index_col=0)

print(dataset)

print(dataset.shape)



# store the current # of rows and columns in the data, M=number of rows, N=number of columns



M, N=dataset.shape

print("the number of rows in dataset is {}".format(M))

print("the number of columns in dataset is {}".format(N))





#slicing the data for objective & constraints



palatability=dataset['palatability'].values

energy=dataset['energy'].values

carbohydrate=dataset['carbohydrate'].values

protein=dataset['protein'].values

fat=dataset['fat'].values

cholesterol=dataset['cholesterol'].values

calcium=dataset['calcium'].values

iron=dataset['iron'].values

vitamin_A=dataset['vitamin_A'].values

vitamin_B1=dataset['vitamin_B1'].values

vitamin_B2=dataset['vitamin_B2'].values

vitamin_C=dataset['vitamin_C'].values

fibre=dataset['fibre'].values



#Big M Method - MM

MM=100000000000

#########Model Set-up###############

m = Model("Meal")

x = m.addVars(M,lb=0.00, name = "x")

y= m.addVars(M, vtype=GRB.BINARY, name = "y")

#print(y)

print(x)
# set objective——maximum palatability

m.setObjective( quicksum(palatability[i]*x[i] for i in range(N)),GRB.MAXIMIZE)





# Nutritional Constraints



# energy constraint: 

m.addConstr(quicksum(energy[i]*x[i] for i in range(M)) ==3000/3 , "energy")



# carbohydrate constraint: 

m.addConstr( quicksum(carbohydrate[i]*x[i] for i in range(M)) ==473/3 , "carbohydrate")



# protein constraint: 

m.addConstr( quicksum(protein[i]*x[i] for i in range(M)) ==95/3 , "protein")



# fat constraint: 

m.addConstr( quicksum(fat[i]*x[i] for i in range(M)) <=81/3 , "fat")



#cholesterol constraint:

m.addConstr( quicksum(cholesterol[i]*x[i] for i in range(M)) <=300/3 , "cholesterol")



#calcium constraint:

m.addConstr( quicksum(calcium[i]*x[i] for i in range(M)) >=800/3 , "calcium")



#iron constraint:

m.addConstr( quicksum(iron[i]*x[i] for i in range(M)) >=6/3 , "iron")



#vitaminA constraint:

m.addConstr( quicksum(vitamin_A[i]*x[i] for i in range(M)) >=750/3 , "vitamin A")



#vitamin B1 constraint:

m.addConstr( quicksum(vitamin_B1[i]*x[i] for i in range(M)) >=1.18/3 , "vitamin B1")



#vitamin B2 constraint:

m.addConstr( quicksum(vitamin_B2[i]*x[i] for i in range(M)) >=1.77/3 , "vitamin B2")



#vitamin C constraint:

m.addConstr( quicksum(vitamin_C[i]*x[i] for i in range(M)) >=30/3 , "vitamin C")



#fibre constraint:

m.addConstr(30/3 >= quicksum(fibre[i]*x[i] for i in range(M)) >=25/3 , "fibre")





#Dish constraints:

m.addConstr( quicksum(y[j] for j in range(0,13)) ==1, "1 staple dish per meal")

m.addConstr( 1<= quicksum(y[j] for j in range(13,28)) <=2, "1 or 2 meat dishes per meal")

m.addConstr( 1<= quicksum(y[j] for j in range(28,49)) <=2, "1 or 2 vegetable dishes per meal")

m.addConstr( quicksum(y[j] for j in range(13,49)) <=3, "max no. of meat and vegetable dishes =3")

m.addConstr( 1<= quicksum(y[j] for j in range(49,64)) <=2, "1 or 2 fruit dishes per meal")

m.addConstr( quicksum(y[j] for j in range(64,74)) <=1, "1 or 2 dairy dishes per meal")





#Minimum Quantity constraints:



#Min quantity of staple (g) for each dish

m.addConstrs( (128*y[i]<= x[i] for i in range(0,13)),"128g_min_quantity")



#Min quantity of meat (g) for each dish

m.addConstrs( (52*y[i]<= x[i] for i in range(13,28)),"52g_min_quantity")



#Min quantity of vegetable (g) for each dish

m.addConstrs( (106*y[i]<= x[i] for i in range(28,49)),"106g_min_quantity")



#Min quantity of fruit (g) for each dish

m.addConstrs( (85*y[i]<= x[i] for i in range(49,64)),"85g_min_quantity")



#Min quantity of dairy (g) for each dish

m.addConstrs( (128*y[i]<= x[i] for i in range(64,74)),"128g_min_quantity")





#Maximum Quantity constraints:



#Max quantity of staple (g) for each dish

m.addConstrs( (x[i] <=256 for i in range(0,13)),"256g_max_quantity")



#Max quantity of meat (g) for each dish

m.addConstrs( (x[i] <=104 for i in range(13,28)),"104g_max_quantity")



#Max quantity of vegetable (g) for each dish

m.addConstrs( (x[i] <=212 for i in range(28,49)),"212g_max_quantity")



#Max quantity of fruit (g) for each dish

m.addConstrs( (x[i] <=170 for i in range(49,64)),"170g_max_quantity")



#Max quantity of dairy (g) for each dish

m.addConstrs( (x[i] <=256 for i in range(64,74)),"256g_max_quantity")





#x is zero when y is zero, Big M Method

m.addConstrs( (x[i] <= MM* y[i] for i in range(M)),"Big M Method")







#trial

m.addConstr(y[2]+y[22]+y[35]+y[46]+y[51]+y[62]<=3)

m.addConstr(y[2]+y[20]+y[35]+y[46]+y[52]+y[54]<=3)

m.addConstr(y[12]+y[21]+y[35]+y[46]+y[52]+y[55]<=3)

m.addConstr(y[2]+y[27]+y[28]+y[33]+y[58]+y[62]+y[67]<=3)

m.addConstr(y[10]+y[21]+y[33]+y[48]+y[54]+y[61]<=3)

m.addConstr(y[10]+y[21]+y[43]+y[48]+y[49]+y[58]<=3)

m.addConstr(y[6]+y[21]+y[33]+y[45]+y[49]+y[67]+y[58]<=3)

m.addConstr(y[11]+y[13]+y[33]+y[54]+y[59]+y[67]+y[14]<=3)

m.addConstr(y[2]+y[22]+y[29]+y[49]+y[51]+y[33]+y[67]<=3)

m.addConstr(y[2]+y[23]+y[28]+y[35]+y[52]+y[62]<=3)

m.addConstr(y[10]+y[13]+y[35]+y[42]+y[56]+y[59]<=3)

m.addConstr(y[10]+y[13]+y[39]+y[47]+y[54]+y[59]<=3)

m.addConstr(y[8]+y[25]+y[42]+y[48]+y[51]+y[54]<=3)



m.addConstr(y[46]==0)

m.addConstr(y[21]==0)

m.addConstr(y[2]==0)

m.addConstr(y[13]==0)

m.addConstr(y[10]==0)

m.addConstr(y[54]==0)

m.addConstr(y[48]==0)

m.addConstr(y[49]==0)

m.addConstr(y[33]==0)

m.addConstr(y[35]==0)







#Meal Composition Constraints:

# m.addConstr(y[6]+y[21]+y[35]+y[46]+y[51]+y[54]<=3)

# m.addConstr(y[3]+y[21]+y[46]+y[48]+y[51]+y[62]<=3)

# m.addConstr(y[1]+y[21]+y[35]+y[46]+y[51]+y[62]<=3)

# m.addConstr(y[2]+y[20]+y[46]+y[48]+y[58]+y[62]<=3)

# m.addConstr(y[12]+y[21]+y[46]+y[48]+y[52]+y[58]<=3)

# m.addConstr(y[2]+y[27]+y[28]+y[33]+y[58]+y[62]+y[67]<=3)

# m.addConstr(y[2]+y[22]+y[33]+y[46]+y[54]+y[55]+y[67]<=3)

# m.addConstr(y[6]+y[16]+y[21]+y[46]+y[49]+y[58]+y[67]<=3)

# m.addConstr(y[10]+y[21]+y[33]+y[48]+y[54]+y[61]<=3)

# m.addConstr(y[10]+y[21]+y[42]+y[46]+y[52]+y[54]<=3)







# Solving the model

m.optimize()



#  Print optimal solutions and optimal value

print("\n Optimal meal:")

for v in m.getVars():

    print(v.VarName, v.x)



print("\n Optimal palatability:")

print('Obj:', m.objVal)