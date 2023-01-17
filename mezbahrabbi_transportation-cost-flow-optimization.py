from pulp import *
import numpy as np
from itertools import product

LANES = 30
CARRIERS = 6

cost = 100 * np.random.rand(LANES, CARRIERS) # c
demand = 10 * np.random.rand(LANES) # b_eq
capacity = [250, 300, 500, 750, 100, 200] # b_ub

prob = LpProblem("Transportation",LpMinimize)
x = LpVariable.dicts("Route", product(range(LANES), range(CARRIERS)), 0, None)

prob += lpSum(cost[l, c] * x[l, c] for l in range(LANES) for c in range(CARRIERS))

for l in range(LANES):
    prob += lpSum(cost[l, c] * x[l, c] for c in range(CARRIERS)) == demand[l]

for c in range(CARRIERS):
    prob += lpSum(cost[l, c] * x[l, c] for l in range(LANES)) <= capacity[c]

prob.solve()

# Get optimal solution
if LpStatus[prob.status] == "Optimal":
    x = {(l, c): value(x[l, c]) for l in range(LANES) for c in range(CARRIERS)}
else:
    print("Optimization failed.")
