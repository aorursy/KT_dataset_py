from ortools.linear_solver import pywraplp



#Nome: Glauber Atanaka dos Santos - SI

#RGA: 201611316018



solver = pywraplp.Solver('SolverAtividadeSAD', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)



maca = solver.NumVar(0, solver.infinity(), 'maca')

banana = solver.NumVar(0, solver.infinity(), 'banana')

cerveja = solver.NumVar(0, solver.infinity(), 'cerveja')

coca = solver.NumVar(0, solver.infinity(), 'coca')



obj1 = solver.Constraint(10, 30)

obj1.SetCoefficient(maca, 3)

obj1.SetCoefficient(banana, 1)



obj2 = solver.Constraint(10, 40)

obj2.SetCoefficient(maca, 2)

obj2.SetCoefficient(coca, 1)



obj3 = solver.Constraint(10, 30)

obj3.SetCoefficient(cerveja, 1)

obj3.SetCoefficient(coca, 3)



objective = solver.Objective()

objective.SetCoefficient(maca, 3)

objective.SetCoefficient(banana, 2)

objective.SetCoefficient(cerveja, 2)

objective.SetCoefficient(coca, 3)

objective.SetMaximization()



solver.Solve()

print('Número de variáveis =', solver.NumVariables())

print('Número de restrições =', solver.NumConstraints())



print('Solução:')

print('maca =', maca.solution_value())

print('banana =', banana.solution_value())

print('cerveja =', cerveja.solution_value())

print('coca =', coca.solution_value())


