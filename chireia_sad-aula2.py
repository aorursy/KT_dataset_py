from ortools.linear_solver import pywraplp



solver = pywraplp.Solver('solver1', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)



# Cria variáveis do solver

x = solver.NumVar(0, 1, 'x')

y = solver.NumVar(0, 2, 'y')



# Cria a função objetivo, x + y

objective = solver.Objective()

objective.SetCoefficient(x, 1)

objective.SetCoefficient(y, -1)

objective.SetMaximization()



#Display do Resultado

solver.Solve()

print('Solução: ')

print('x = ', x.solution_value())

print('y = ', y.solution_value())
solver = pywraplp.Solver('solver2', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)





# Cria variáveis do solver

x = solver.NumVar(-solver.infinity(), solver.infinity(), 'x')

y = solver.NumVar(-solver.infinity(), solver.infinity(), 'y')



# Restrição 1: x + 2y <= 14

constraint1 = solver.Constraint(-solver.infinity(), 14)

constraint1.SetCoefficient(x, 1)

constraint1.SetCoefficient(y, 2)



# Restrição 2: 3x - y >= 0

constraint2 = solver.Constraint(0, solver.infinity())

constraint2.SetCoefficient(x, 3)

constraint2.SetCoefficient(y, -1)



# Restrição 3: x - y <= 2

constraint3 = solver.Constraint(-solver.infinity(), 2)

constraint3.SetCoefficient(x, 1)

constraint3.SetCoefficient(y, -1)



# Cria a função objetivo, 3x + 4y

objective = solver.Objective()

objective.SetCoefficient(x, 3)

objective.SetCoefficient(y, 4)

objective.SetMaximization()



# Soluciona o Problema

solver.Solve()

opt_solution = 3 * x.solution_value() + 4 * y.solution_value()

print('Número de variáveis: ', solver.NumVariables())

print('Número de restrições: ', solver.NumConstraints())



# Display do Resultado

print('Solução: ')

print('x = ', x.solution_value())

print('y = ', y.solution_value())



# A solução com melhor valor

print('Melhor valor: ', opt_solution)