#The following code declares the CP solver and creates two arrays of variables for the problem: shifts and nurses
from __future__ import print_function
import sys
from ortools.constraint_solver import pywrapcp

def main():
  # Creates the solver.
  solver = pywrapcp.Solver("schedule_shifts")

  num_workers = 4
  num_shifts = 4     # Workers assigned to shift 0 means not working that day.
  num_days = 7
  # [START]
  # Create shift variables.
  shifts = {}

  for j in range(num_workers):
    for i in range(num_days):
      shifts[(j, i)] = solver.IntVar(0, num_shifts - 1, "shifts(%i,%i)" % (j, i))
  shifts_flat = [shifts[(j, i)] for j in range(num_workers) for i in range(num_days)]

  # Create worker variables.
  workers = {}

  for j in range(num_shifts):
    for i in range(num_days):
      workers[(j, i)] = solver.IntVar(0, num_workers - 1, "shift%d day%d" % (j,i))
  # Set relationships between shifts and workers.
  for day in range(num_days):
    workers_for_day = [workers[(j, day)] for j in range(num_shifts)]

    for j in range(num_workers):
      s = shifts[(j, day)]
      solver.Add(s.IndexOf(workers_for_day) == j)
  # Make assignments different on each day
  for i in range(num_days):
    solver.Add(solver.AllDifferent([shifts[(j, i)] for j in range(num_workers)]))
    solver.Add(solver.AllDifferent([workers[(j, i)] for j in range(num_shifts)]))
  # Each worker works 5 or 6 days in a week.
  for j in range(num_workers):
    solver.Add(solver.Sum([shifts[(j, i)] > 0 for i in range(num_days)]) >= 5)
    solver.Add(solver.Sum([shifts[(j, i)] > 0 for i in range(num_days)]) <= 6)
  # Create works_shift variables. works_shift[(i, j)] is True if worker
  # i works shift j at least once during the week.
  works_shift = {}

  for i in range(num_workers):
    for j in range(num_shifts):
      works_shift[(i, j)] = solver.BoolVar('shift%d worker%d' % (i, j))

  for i in range(num_workers):
    for j in range(num_shifts):
      solver.Add(works_shift[(i, j)] == solver.Max([shifts[(i, k)] == j for k in range(num_days)]))

  # For each shift (other than 0), at most 2 workers are assigned to that shift
  # during the week.
  for j in range(1, num_shifts):
    solver.Add(solver.Sum([works_shift[(i, j)] for i in range(num_workers)]) <= 2)
  # If s workers works shifts 2 or 3 on, he must also work that shift the previous
  # day or the following day.
  solver.Add(solver.Max(workers[(2, 0)] == workers[(2, 1)], workers[(2, 1)] == workers[(2, 2)]) == 1)
  solver.Add(solver.Max(workers[(2, 1)] == workers[(2, 2)], workers[(2, 2)] == workers[(2, 3)]) == 1)
  solver.Add(solver.Max(workers[(2, 2)] == workers[(2, 3)], workers[(2, 3)] == workers[(2, 4)]) == 1)
  solver.Add(solver.Max(workers[(2, 3)] == workers[(2, 4)], workers[(2, 4)] == workers[(2, 5)]) == 1)
  solver.Add(solver.Max(workers[(2, 4)] == workers[(2, 5)], workers[(2, 5)] == workers[(2, 6)]) == 1)
  solver.Add(solver.Max(workers[(2, 5)] == workers[(2, 6)], workers[(2, 6)] == workers[(2, 0)]) == 1)
  solver.Add(solver.Max(workers[(2, 6)] == workers[(2, 0)], workers[(2, 0)] == workers[(2, 1)]) == 1)

  solver.Add(solver.Max(workers[(3, 0)] == workers[(3, 1)], workers[(3, 1)] == workers[(3, 2)]) == 1)
  solver.Add(solver.Max(workers[(3, 1)] == workers[(3, 2)], workers[(3, 2)] == workers[(3, 3)]) == 1)
  solver.Add(solver.Max(workers[(3, 2)] == workers[(3, 3)], workers[(3, 3)] == workers[(3, 4)]) == 1)
  solver.Add(solver.Max(workers[(3, 3)] == workers[(3, 4)], workers[(3, 4)] == workers[(3, 5)]) == 1)
  solver.Add(solver.Max(workers[(3, 4)] == workers[(3, 5)], workers[(3, 5)] == workers[(3, 6)]) == 1)
  solver.Add(solver.Max(workers[(3, 5)] == workers[(3, 6)], workers[(3, 6)] == workers[(3, 0)]) == 1)
  solver.Add(solver.Max(workers[(3, 6)] == workers[(3, 0)], workers[(3, 0)] == workers[(3, 1)]) == 1)
# Create the decision builder.
  db = solver.Phase(shifts_flat, solver.CHOOSE_FIRST_UNBOUND,
                    solver.ASSIGN_MIN_VALUE)
# Create the solution collector.
  solution = solver.Assignment()
  solution.Add(shifts_flat)
  collector = solver.AllSolutionCollector(solution)

  solver.Solve(db, [collector])
  print("Solutions found:", collector.SolutionCount())
  print("Time:", solver.WallTime(), "ms")
  print()
  # Display a few solutions picked at random.
  a_few_solutions = [859, 2034, 5091, 7003]

  for sol in a_few_solutions:
    print("Solution number" , sol, '\n')

    for i in range(num_days):
      print("Day", i)
      for j in range(num_workers):
        print("Worker", j, "assigned to task",
              collector.Value(sol, shifts[(j, i)]))
      print()

if __name__ == "__main__":
  main()
