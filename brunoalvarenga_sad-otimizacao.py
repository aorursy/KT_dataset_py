from ortools.linear_solver import pywraplp



solver = pywraplp.Solver('Sad', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)





US = solver.NumVar(0, solver.infinity(), 'US')

Spain = solver.NumVar(0, solver.infinity(), 'Spain')

France = solver.NumVar(0, solver.infinity(), 'France')

Italy = solver.NumVar(0, solver.infinity(), 'Italy')

Bulgaria = solver.NumVar(0, solver.infinity(), 'Bulgaria')

Australia = solver.NumVar(0, solver.infinity(), 'Australia')



solverA = solver.Constraint(5, 40)

solverB = solver.Constraint(20, 80)

SolverC = solver.Constraint(15, 25)



solverA.SetCoefficient(US, 8)

solverB.SetCoefficient(Spain, 8)

SolverC.SetCoefficient(France, 2)



solverA.SetCoefficient(Bulgaria, 2)



SolverC.SetCoefficient(Italy, 1)





obj = solver.Objective()

obj.SetCoefficient(US, 2)

obj.SetCoefficient(Spain, 3)

obj.SetCoefficient(France, 3)

obj.SetCoefficient(Bulgaria, 2)

obj.SetMaximization()



solver.Solve()

print('US -', US.solution_value())

print('Spain -', Spain.solution_value())

print('France -', France.solution_value())

print('Bulgaria -', Bulgaria.solution_value())

print('Quantidade de variaveis', solver.NumVariables())

print('Quantidade de constraints', solver.NumConstraints())