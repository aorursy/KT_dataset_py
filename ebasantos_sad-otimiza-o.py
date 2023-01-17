from ortools.linear_solver import pywraplp
solver = pywraplp.Solver('SolverSimpleSystem', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)



x = solver.NumVar(0,solver.infinity(), 'X')

y = solver.NumVar(0,solver.infinity(), 'Y')



#R0 Cabem no máximo 100 artigos: x + y ≤ 100

constraint1 = solver.Constraint(0, 100)

constraint1.SetCoefficient(x,1)

constraint1.SetCoefficient(y,1)



#R1 Serão vendidos pelo menos 15 artigos A: x ≥ 15

constraint1 = solver.Constraint(0, 15)

constraint1.SetCoefficient(x,1)



#R2 Serão vendidos pelo menos 25 artigos B: y ≥ 25

constraint2 = solver.Constraint(0, 25)

constraint2.SetCoefficient(y,1)



#R3 O distribuidor entregará no máximo 60 artigos A: x ≤ 60.

constraint3 = solver.Constraint(0,60)

constraint3.SetCoefficient(x,1)



#R4 O distribuidor entregará no máximo 50 artigos B: y ≤ 50.



constraint3 = solver.Constraint(0,50)

constraint3.SetCoefficient(y,1)





# L =20x + 30y

objective = solver.Objective()

objective.SetCoefficient(x,20)

objective.SetCoefficient(y,30)

objective.SetMaximization()
solver.Solve()

opt_solution = 20 * x.solution_value() + 30 * y.solution_value()

print('Número de variáveis =', solver.NumVariables())

print('Número de restrições =', solver.NumConstraints())



print('Solução:')

print('x =', x.solution_value())

print('y =', y.solution_value())



print('Melhor valor =', opt_solution)