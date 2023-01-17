from ortools.linear_solver import pywraplp
solver = pywraplp.Solver('SolverSimpleSystem', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

#criar variaveis x e y.



x = solver.NumVar(0,solver.infinity(), 'Mistura A')

y = solver.NumVar(0,1.2,'Mistura B')



#restrição 1: 1/2x + 1/3y ≤ 130

constraint1 = solver.Constraint(0, 130)

constraint1.SetCoefficient(x,1/2)

constraint1.SetCoefficient(y,1/3)



#restrição 2: 1/2x + 2/3y ≤ 170

constraint2 = solver.Constraint(0, 170)

constraint2.SetCoefficient(x,1/2)

constraint2.SetCoefficient(y,1/3)



#restrição 3: x ≥ 0

constraint3 = solver.Constraint(0,solver.infinity())

constraint3.SetCoefficient(x,1)



#restrição 4: y ≥ 0

constraint3 = solver.Constraint(0,solver.infinity())

constraint3.SetCoefficient(y,1)



#restrição 5: 20x + 12.50y >=0

objective = solver.Objective()

objective.SetCoefficient(x,20)

objective.SetCoefficient(y,12.50)

objective.SetMaximization()

#chamar resultados

solver.Solve()

opt_solution = 20 * x.solution_value() + 12.50 * y.solution_value()

print('Número de variáveis =', solver.NumVariables())

print('Número de restrições =', solver.NumConstraints())

print('Solução:')

print('Deve preparar',x.solution_value(),'KG da Mistura A e ', y.solution_value(),'KG da Mistura B')

#print('Mistura B =', y.solution_value())

#print('Melhor valor =', opt_solution)