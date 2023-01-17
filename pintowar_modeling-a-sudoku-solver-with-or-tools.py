import numpy as np

import pandas as pd

from datetime import datetime



sudoku = pd.read_csv("../input/sudoku/sudoku.csv") # Loading puzzles from csv

sample = sudoku.loc[2020] # row 2020

sample
def decode_sudoku(sample: str) -> np.matrix:

    '''Transform an encoded puzzle into an integer matrix.'''

    return np.matrix([np.array(list(sample[i:i+9])).astype(np.int) for i in range(0, len(sample), 9)])



decoded_puzzle = decode_sudoku(sample['puzzle'])

decoded_puzzle
def encode_sudoku(sudoku: np.matrix) -> str:

    '''Transform an integer matrix into an encoded string'''

    return ''.join([''.join(list(r.astype(str))) for r in np.asarray(sudoku)])



encoded_puzzle = encode_sudoku(decoded_puzzle)



assert encoded_puzzle == sample['puzzle'] # must be true, since the same puzzle was decoded and encoded

encoded_puzzle
from ortools.sat.python import cp_model



def solve_with_cp(grid: np.matrix) -> (np.matrix, float):

    '''Solve Sudoku instance (np.matrix) with CP modeling. Returns a tuple with the resulting matrix and the execution time in seconds.'''

    assert grid.shape == (9,9)

    

    grid_size = 9

    region_size = 3 #np.sqrt(grid_size).astype(np.int)

    model = cp_model.CpModel() # Step 1



    # Begin of Step2: Create and initialize variables.

    x = {}

    for i in range(grid_size):

        for j in range(grid_size):

            if grid[i, j] != 0:

                x[i, j] = grid[i, j] # Initial values (values already defined on the puzzle).

            else:

                x[i, j] = model.NewIntVar(1, grid_size, 'x[{},{}]'.format(i,j) ) # Values to be found (variyng from 1 to 9).

    # End of Step 2.



    # Begin of Step3: Values constraints.

    # AllDifferent on rows, to declare that all elements of all rows must be different.

    for i in range(grid_size):

        model.AddAllDifferent([x[i, j] for j in range(grid_size)])



    # AllDifferent on columns, to declare that all elements of all columns must be different.

    for j in range(grid_size):

        model.AddAllDifferent([x[i, j] for i in range(grid_size)])



    # AllDifferent on regions, to declare that all elements of all regions must be different.

    for row_idx in range(0, grid_size, region_size):

        for col_idx in range(0, grid_size, region_size):

            model.AddAllDifferent([x[row_idx + i, j] for j in range(col_idx, (col_idx + region_size)) for i in range(region_size)])

    # End of Step 3.



    solver = cp_model.CpSolver() # Step 4

    start = datetime.now()

    status = solver.Solve(model) # Step 5

    exec_time = datetime.now() - start

    result = np.zeros((grid_size, grid_size)).astype(np.int)



    # Begin of Step 6: Getting values defined by the solver

    if status == cp_model.FEASIBLE:

        for i in range(grid_size):

            for j in range(grid_size):

                result[i,j] = int(solver.Value(x[i,j]))

    else:

        raise Exception('Unfeasible Sudoku')

    # End of Step 6



    return result, exec_time.total_seconds()



res, _ = solve_with_cp(decoded_puzzle)

cp_solution = encode_sudoku(res) 



assert cp_solution == sample['solution'] # must show the same solution for the puzzle found on the dataset

res
from ortools.linear_solver import pywraplp



def solve_with_ip(grid: np.ndarray) -> (np.ndarray, float):

    '''Solve Sudoku instance (np.matrix) with IP modeling. Returns a tuple with the resulting matrix and the execution time in seconds.'''

    assert grid.shape == (9,9)

    

    grid_size = 9

    cell_size = 3 #np.sqrt(grid_size).astype(np.int)

    solver = pywraplp.Solver('Sudoku Solver', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING) # Step 1



    # Begin of Step2: Create variables.

    x = {}

    for i in range(grid_size):

        for j in range(grid_size):

            # Initial values.

            for k in range(grid_size):

                x[i, j, k] = solver.BoolVar('x[%i,%i,%i]' % (i, j, k))

    # End of Step2

    

    # Begin of Step3: Initialize variables in case of known (defined) values.

    for i in range(grid_size):

        for j in range(grid_size):

            defined = grid[i, j] != 0

            if defined:

                solver.Add(x[i,j,grid[i, j]-1] == 1)

    # End of Step3

    

    # Begin of Step4: Initialize variables in case of known (defined) values. 

    # All bins of a cell must have sum equals to 1

    for i in range(grid_size):

        for j in range(grid_size):

            solver.Add(solver.Sum([x[i, j, k] for k in range(grid_size)]) == 1)

    # End of Step4



    # Begin of Step5: Create variables.

    for k in range(grid_size):

        # AllDifferent on rows.

        for i in range(grid_size):

            solver.Add(solver.Sum([x[i, j, k] for j in range(grid_size)]) == 1)



        # AllDifferent on columns.

        for j in range(grid_size):

            solver.Add(solver.Sum([x[i, j, k] for i in range(grid_size)]) == 1)



        # AllDifferent on regions.

        for row_idx in range(0, grid_size, cell_size):

            for col_idx in range(0, grid_size, cell_size):

                solver.Add(solver.Sum([x[row_idx + i, j, k] for j in range(col_idx, (col_idx + cell_size)) for i in range(cell_size)]) == 1)

    # End of Step5



    # Solve and print out the solution.

    start = datetime.now()

    status = solver.Solve() # Step 6

    exec_time = datetime.now() - start

    statusdict = {0:'OPTIMAL', 1:'FEASIBLE', 2:'INFEASIBLE', 3:'UNBOUNDED', 

                  4:'ABNORMAL', 5:'MODEL_INVALID', 6:'NOT_SOLVED'}

    

    result = np.zeros((grid_size, grid_size)).astype(np.int)

    if status == pywraplp.Solver.OPTIMAL:

        for i in range(grid_size):

            for j in range(grid_size):

                result[i,j] = sum((k + 1) * int(x[i, j, k].solution_value()) for k in range(grid_size))

    else:

        raise Exception('Unfeasible Sudoku: {}'.format(statusdict[status]))



    return result, exec_time.total_seconds()



res, _ = solve_with_ip(decoded_puzzle)

ip_solution = encode_sudoku(res) 



assert ip_solution == sample['solution'] # must show the same solution for the puzzle found on the dataset

res
def solve_sudoku(instance: np.matrix, solver: str = 'ip') -> (np.matrix, float):

    if solver == 'ip':

        return solve_with_ip(instance)

    elif solver == 'cp':

        return solve_with_cp(instance)

    else:

        raise Exception('Unknown solver: {}'.format(solver))



solve_sudoku(decode_sudoku(sample['puzzle']))
from tqdm.notebook import tqdm



sample_size = 1000

seed = 2020

ip_exec_time = []

cp_exec_time = []



for index, row in tqdm(sudoku.sample(sample_size, random_state=seed).iterrows()):

    res, exec_time = solve_sudoku(decode_sudoku(row.puzzle), 'cp') # Solving with CP

    assert encode_sudoku(res) == row.solution # Assert if result equals to the expected solution

    cp_exec_time += [exec_time] # Register the solver execution time

    

    res, exec_time = solve_sudoku(decode_sudoku(row.puzzle), 'ip') # Solving with IP

    assert encode_sudoku(res) == row.solution # Assert if result equals to the expected solution

    ip_exec_time += [exec_time] # Register the solver execution time

    

performance_df = pd.DataFrame({'IP' : ip_exec_time, 'CP' : cp_exec_time})

performance_df.head()
performance_df.plot.hist(subplots=True, figsize=(12, 4), layout=(1,2))