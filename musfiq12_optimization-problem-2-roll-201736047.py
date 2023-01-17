#problem:Maximize 4x + 5y subject to the following constraints:

          #x + 3y≤15

          #4x – y≥0

          #x – y≤3

#help_source_code_https://developers.google.com/optimization/introduction/python#simple_example



"""Linear optimization example"""



from __future__ import print_function

from ortools.linear_solver import pywraplp



def main():

  # Instantiate a Glop solver, naming it LinearExample.

  solver = pywraplp.Solver('LinearExample',

                           pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)



# Create the two variables and let them take on any value.

  x = solver.NumVar(-solver.infinity(), solver.infinity(), 'x')

  y = solver.NumVar(-solver.infinity(), solver.infinity(), 'y')



  # Constraint 1: x + 3y <= 15.

  constraint1 = solver.Constraint(-solver.infinity(), 15)

  constraint1.SetCoefficient(x, 1)

  constraint1.SetCoefficient(y, 3)



  # Constraint 2: 4x - y >= 0.

  constraint2 = solver.Constraint(0, solver.infinity())

  constraint2.SetCoefficient(x, 4)

  constraint2.SetCoefficient(y, -1)



  # Constraint 3: x - y <= 3.

  constraint3 = solver.Constraint(-solver.infinity(), 3)

  constraint3.SetCoefficient(x, 1)

  constraint3.SetCoefficient(y, -1)



  # Objective function: 4x + 5y.

  objective = solver.Objective()

  objective.SetCoefficient(x, 4)

  objective.SetCoefficient(y, 5)

  objective.SetMaximization()



  # Solve the system.

  solver.Solve()

  opt_solution = 4 * x.solution_value() + 5 * y.solution_value()

  print('Number of variables =', solver.NumVariables())

  print('Number of constraints =', solver.NumConstraints())

  # The value of each variable in the solution.

  print('Solution:')

  print('x = ', x.solution_value())

  print('y = ', y.solution_value())

  # The objective value of the solution.

  print('Optimal objective value =', opt_solution)

if __name__ == '__main__':

  main()