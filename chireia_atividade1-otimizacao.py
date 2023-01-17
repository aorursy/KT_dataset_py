from ortools.linear_solver import pywraplp
solver = pywraplp.Solver('SolverSimpleSystem', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)



# Numero de brinquedos X e Y igual a A e B respectivamente:

# Não podem ser negativos

x = solver.NumVar(0, solver.infinity(),'x')

y = solver.NumVar(0, solver.infinity(),'y')



# Dono estima máximo de 2000 brinquedos, então:

# x + y <= 2000

obj1 = solver.Constraint(0, 2000)

obj1.SetCoefficient(x, 1)

obj1.SetCoefficient(y, 1)



# Cada brinquedo custa 8 e 14 respectivamente e o dono não vai

# investir mais que 20000, então:

# 8x + 14y <= 20000

obj2 = solver.Constraint(0, 20000)

obj2.SetCoefficient(x, 8)

obj2.SetCoefficient(y, 14)



# Eles vendem a 2 e 3 reais respectivamente os brinquedos,

# Para maximizar o lucro então:

objective = solver.Objective()

objective.SetCoefficient(x, 2)

objective.SetCoefficient(y, 3)

objective.SetMaximization()



solver.Solve()

opt_solution = 2 * x.solution_value() + 3 * y.solution_value()

print('Número de variáveis =', solver.NumVariables())

print('Número de restrições =', solver.NumConstraints())



print('Solução:')

print('x =', x.solution_value())

print('y =', y.solution_value())



print('Melhor valor =', opt_solution)