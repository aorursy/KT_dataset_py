from ortools.linear_solver import pywraplp
solver = pywraplp.Solver('SolverSimpleSystem', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)



x = solver.NumVar(0, 1, 'x')

y = solver.NumVar(0, 2, 'y')



objective = solver.Objective()

objective.SetCoefficient(x, 1)

objective.SetCoefficient(y, 1)

objective.SetMaximization()



solver.Solve()

print('Solução:')

print('x =', x.solution_value())

print('y =', y.solution_value())
from ortools.linear_solver import pywraplp
solver = pywraplp.Solver('SolverSimpleSystem', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)



x = solver.NumVar(-solver.infinity(), solver.infinity(), 'x')

y = solver.NumVar(-solver.infinity(), solver.infinity(), 'y')



objective1 = solver.Constraint(-solver.infinity(), 14)

objective1.SetCoefficient(x, 1)

objective1.SetCoefficient(y, 2)



objective2 = solver.Constraint(0, solver.infinity())

objective2.SetCoefficient(x, 3)

objective2.SetCoefficient(y, -1)



objective3 = solver.Constraint(-solver.infinity(), 2)

objective3.SetCoefficient(x, 1)

objective3.SetCoefficient(y, -1)



objective = solver.Objective()

objective.SetCoefficient(x, 3)

objective.SetCoefficient(y, 4)

objective.SetMaximization()



solver.Solve()

opt_solution = 3 * x.solution_value() + 4 * y.solution_value()

print('Número de variáveis =', solver.NumVariables())

print('Número de restrições =', solver.NumConstraints())







print('Solução:')

print('x =', x.solution_value())

print('y =', y.solution_value())



print('Melhor valor =', opt_solution)
from ortools.linear_solver import pywraplp
solver = pywraplp.Solver('Airline', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)



lugares = 100

valor_min = 100

valor_max = 1000



pass_normal = solver.NumVar(0, lugares, 'pass_normal')

tarif_normal = solver.NumVar(valor_min, solver.Infinity(), 'tarif_normal')

pass_econom = solver.NumVar(0, lugares, 'pass_econom')

tarif_econom = solver.NumVar(valor_min, solver.Infinity(), 'tarif_econom')