!pip install -Iv pulp==1.6.8 --quiet
import pulp as plp
Lpproblem = plp.LpProblem('Problem', plp.LpMaximize) 
x = plp.LpVariable("x", lowBound = 0)   
y = plp.LpVariable("y", lowBound = 0)  
Lpproblem += 30 * x + 40 * y 
Lpproblem += 12 * x + 4 * y <= 300
Lpproblem += 4 * x + 4 * y <= 120
Lpproblem += 3 * x + 12 * y <= 252
print(Lpproblem)
status = Lpproblem.solve()    
print("Model status: ", plp.LpStatus[status])   
print("Model solution: \nProduct A: ", plp.value(x), "\nProduct B: ", plp.value(y), "\nProfit: ",plp.value(Lpproblem.objective)) 