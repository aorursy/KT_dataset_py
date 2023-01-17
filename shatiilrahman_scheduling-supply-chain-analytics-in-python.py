!pip install pulp

print('PuLP for Optimization Problem - start')

from pulp import *

model = LpProblem("Minimization problem - Scheduling: ", LpMinimize)

days = list(range(7))



x = LpVariable.dicts('staff_', days, lowBound=0, cat='Integer')

model += lpSum(x[i] for i in days)

model += x[0] + x[1] + x[2] + x[3] + x[4] + x[5] >= 20

model += x[0] + x[2] + x[3] + x[4] + x[5] + x[6] >= 14

model += x[0] + x[3] + x[4] + x[5] + x[6] + x[1] >= 11

model += x[0] + x[4] + x[5] + x[6] + x[1] + x[2] >= 15

model += x[0] + x[5] + x[6] + x[1] + x[2] + x[3] >= 22

model += x[0] + x[6] + x[1] + x[2] + x[3] + x[4] >= 12

model += x[1] + x[2] + x[3] + x[4] + x[5] + x[6] >= 25





model.solve()
print("Model Status:", LpStatus[model.status])

for v in model.variables():

    print(v.name, "=", v.varValue)

print("The optimised objective function= ", value(model.objective))