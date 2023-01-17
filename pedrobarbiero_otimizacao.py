from ortools.linear_solver import pywraplp



solver = pywraplp.Solver('Sad', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)





gremio = solver.NumVar(0, solver.infinity(), 'gremio')

flamengo = solver.NumVar(0, solver.infinity(), 'flamengo')

inter = solver.NumVar(0, solver.infinity(), 'inter')

chape = solver.NumVar(0, solver.infinity(), 'chape')

cruzeiro = solver.NumVar(0, solver.infinity(), 'cruzeiro')

botafogo = solver.NumVar(0, solver.infinity(), 'botafogo')

goias = solver.NumVar(0, solver.infinity(), 'goias')



solverA = solver.Constraint(5, 40)

solverB = solver.Constraint(20, 80)

SolverC = solver.Constraint(15, 25)



solverA.SetCoefficient(gremio, 8)

solverB.SetCoefficient(gremio, 8)

SolverC.SetCoefficient(inter, 2)



solverA.SetCoefficient(flamengo, 2)

solverB.SetCoefficient(goias, 4)

SolverC.SetCoefficient(goias, 1)





obj = solver.Objective()

obj.SetCoefficient(gremio, 2)

obj.SetCoefficient(flamengo, 3)

obj.SetCoefficient(inter, 3)

obj.SetCoefficient(goias, 2)

obj.SetMaximization()



solver.Solve()

print('gremio -', gremio.solution_value())

print('flamengo -', flamengo.solution_value())

print('inter -', inter.solution_value())

print('goias -', goias.solution_value())

print('Quantidade de variaveis', solver.NumVariables())

print('Quantidade de constraints', solver.NumConstraints())
