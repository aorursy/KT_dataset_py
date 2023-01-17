import time

kaggle_timeout = time.perf_counter() + 8.9 * 60 * 60
# TODO: add z3-solver to kaggle-docker image

! pip3 install -q z3-solver
#!/usr/bin/env python3



import time

from itertools import chain  # flatten nested lists; chain(*[[a, b], [c, d], ...]) == [a, b, c, d, ...]

import z3

from z3 import *



rows = 'ABCDEFGHI'

cols = '123456789'

boxes = [[Int("{}{}".format(r, c)) for c in cols] for r in rows]  # declare variables for each box in the puzzle

square_units = [ [ x+y for x in A for y in B ] for A in ('ABC','DEF','GHI') for B in ('123','456','789') ]



def sudoku_solver(board):

    s_solver = Solver()  # create a solver instance



    # Add constraints that every box has a value between 1-9 (inclusive)

    s_solver.add([ And(1 <= box, box <= 9) for box in chain(*boxes) ])



    # Add constraints that every box in a row has a distinct value

    for i in range(len(boxes)): s_solver.add(Distinct(boxes[i]))



    # Add constraints that every box in a column has a distinct value

    for i in range(len(boxes)): s_solver.add(Distinct([ row[i] for row in boxes ]))



    # Add constraints so that every box in a 3x3 block has a distinct value

    for rows in [[0,1,2],[3,4,5],[6,7,8]]:

        for cols in [[0,1,2],[3,4,5],[6,7,8]]:

            s_solver.add(Distinct([ boxes[r][c] for r in rows for c in cols ]))



    # Add constraints for the problem defined in the input board

    for i in range(len(board)):

        for j in range(len(board[0])):

            if board[i][j] != 0:

                s_solver.add( boxes[i][j] == board[i][j] )



    return s_solver
import pandas as pd

from pathos.multiprocessing import ProcessPool





### Conversion Functions



def format_time(seconds):

    if seconds < 1:        return "{:.0f}ms".format(seconds*1000)

    if seconds < 60:       return "{:.2f}s".format(seconds)

    if seconds < 60*60:    return "{:.0f}m {:.0f}s".format(seconds//60, seconds%60)

    if seconds < 60*60*60: return "{:.0f}h {:.0f}m {:.0f}s".format(seconds//(60*60), (seconds//60)%60, seconds%60)    





def string_to_tuple(string):

    if isinstance(string, Solver): string = solver_to_tuple(string)



    string = string.replace('.','0')

    output = tuple( tuple(map(int, string[n*9:n*9+9])) for n in range(0,9) )

    return output





def tuple_to_string(board, zeros='.'):

    if isinstance(board, str):    board = string_to_tuple(board)    

    if isinstance(board, Solver): board = solver_to_tuple(board)



    output = "".join([ "".join(map(str,row)) for row in board ])

    output = output.replace('0', zeros)

    return output





def solver_to_tuple(s_solver):

    output = tuple(

        tuple(

            int(s_solver.model()[box].as_string())

            for col, box in enumerate(_boxes)

        )

        for row, _boxes in enumerate(boxes)

    )

    return output





def solver_to_string(s_solver, zeros='.'):

    output = "".join(

        "".join(

            s_solver.model()[box].as_string()

            for col, box in enumerate(_boxes)

        )

        for row, _boxes in enumerate(boxes)

    )

    return output





def series_to_inout_pair(series):

    input  = ''

    output = ''

    for key, value in series.iteritems():

        if isinstance(value, str) and len(value) == 9*9:

            if not input: input  = value

            else:         output = value

    return (input, output)







### Print Functions



def print_board(board):

    if isinstance(board, str):     board = string_to_tuple(board)

    if isinstance(board, Solver):  board = solver_to_tuple(board)

    for row, _boxes in enumerate(boxes):

        if row and row % 3 == 0:

            print('-'*9+"|"+'-'*9+"|"+'-'*9)

        for col, box in enumerate(_boxes):

            if col and col % 3 == 0:

                print('|', end='')

            print(' {} '.format((board[row][col] or '-')), end='')

        print()

    print()

    

        

def print_sudoku( board ):

    if isinstance(board, str): board = string_to_tuple(board)    



    print_board(board)



    time_start = time.perf_counter()            

    s_solver   = sudoku_solver(board)  

    time_end   = time.perf_counter()        

    if s_solver.check() != sat: print('Unsolvable'); return



    time_end   = time.perf_counter()        

    print_board(s_solver)

    print('solved in {:.2f}s'.format(time_end - time_start))

    



    

### Solve Functions

    

def solve_sudoku( board, format=str ):

    """This is really just a wrapper function that deals with type conversion"""

    if isinstance(board, str):     board = string_to_tuple(board)

    if isinstance(board, Solver):  board = solver_to_tuple(board)

    

    s_solver = sudoku_solver(board)

    

    if s_solver.check() != sat: 

        return None

    if format == str:

        return solver_to_string(s_solver)

    if format == tuple:

        return solver_to_tuple(s_solver)        

    return s_solver





from joblib import delayed

from joblib import Parallel



def solve_dataframe(dataframe, count=0, timeout=0, verbose=0):

    if isinstance(dataframe, str): dataframe = pd.read_csv(dataframe)

    time_start = time.perf_counter()    

    dataframe  = dataframe.copy() 

    

    solved = 0

    total  = 0

    count  = count or len(dataframe.index)

    count  = min(count,len(dataframe.index))

    

    pool   = ProcessPool(os.cpu_count())

    try:    pool.restart()

    except: pass

    try:

        if 'time_ms' not in dataframe.columns: dataframe['time_ms'] = 0

            

        all_idxs = ( idx for (idx, row) in dataframe.query('time_ms == 0').iterrows()  )  # generator

        all_rows = ( row for (idx, row) in dataframe.query('time_ms == 0').iterrows()  )  # generator

        while total < count:

            if timeout and timeout < time.perf_counter() - time_start: break

            batch_size = min(count-total, 1000) 

            idxs              =    (                        next(all_idxs)  for _ in range(batch_size)  )

            boards, expecteds = zip(*[ series_to_inout_pair(next(all_rows)) for _ in range(batch_size) ])

            def time_solve_sudoku(board):

                time_start = time.perf_counter()

                sudoku     = solve_sudoku(board, format=str)

                time_taken = time.perf_counter() - time_start

                return sudoku, time_taken 

            outputs = pool.map(time_solve_sudoku, boards)

            for idx, board, (output, time_taken), expected in zip(idxs, boards, outputs, expecteds):

                solved += int( print_output(board, output, expected, verbose=verbose) )

                total  += 1

                dataframe.at[idx,'time_ms']     = int(time_taken * 1000)

                if total >= count: break

                

    except (KeyboardInterrupt, TimeoutError): pass

    except Exception as exception: raise exception

    finally:

        pool.terminate()

    

    failed      = total - solved

    time_end    = time.perf_counter() 

    time_taken  = time_end-time_start

    time_sudoku = time_taken / total if total else 0

    print(f'Solved {solved}/{total} | failed: {failed} | in {format_time(time_taken)} ({format_time(time_sudoku)} per sudoku)')

    if verbose: print()

    return dataframe[ dataframe['time_ms'] > 0 ].sort_values(by='time_ms', ascending=False)

    

    

def print_output(board, output, expected, verbose=1):

    if isinstance(board, str): board = board.replace('0', '.')

    solved = False

    if output is None:

        if verbose: 

            print(f"Failed:    {board} -> {expected} != {output}")

        if verbose >= 2:

            print_board(board)

            print_board('Unsolvable')

    elif output != expected:

        solved = False            

        if verbose: 

            print(f"Different: {board} -> {expected} != {output}")                        

        if verbose >= 2:

            print_board(board)

            print_board(output)

    else:

        solved = True

        if verbose: 

            print(f"Solved:    {board} -> {output}")            

        if verbose >= 3:

            print_board(board)

            print_board(output)  

    return solved



test_board = "..149....642.31........8........67...54...9..9....5..8...6....5.......2...5.24.81"

assert test_board == tuple_to_string(string_to_tuple(test_board))
board = ((0, 0, 3, 0, 2, 0, 6, 0, 0),

         (9, 0, 0, 3, 0, 5, 0, 0, 1),

         (0, 0, 1, 8, 0, 6, 4, 0, 0),

         (0, 0, 8, 1, 0, 2, 9, 0, 0),

         (7, 0, 0, 0, 0, 0, 0, 0, 8),

         (0, 0, 6, 7, 0, 8, 2, 0, 0),

         (0, 0, 2, 6, 0, 9, 5, 0, 0),

         (8, 0, 0, 2, 0, 3, 0, 0, 9),

         (0, 0, 5, 0, 1, 0, 3, 0, 0))

print_sudoku(board)
board_hardest_sudoku = (

    (8, 0, 0, 0, 0, 0, 0, 0, 0),

    (0, 0, 3, 6, 0, 0, 0, 0, 0),

    (0, 7, 0, 0, 9, 0, 2, 0, 0),

    (0, 5, 0, 0, 0, 7, 0, 0, 0),

    (0, 0, 0, 0, 4, 5, 7, 0, 0),

    (0, 0, 0, 1, 0, 0, 0, 3, 0),

    (0, 0, 1, 0, 0, 0, 0, 6, 8),

    (0, 0, 8, 5, 0, 0, 0, 1, 0),

    (0, 9, 0, 0, 0, 0, 4, 0, 0)

)

print_sudoku(board_hardest_sudoku)
! find ../input -name '*.csv'
files = {

    "1 million": '../input/sudoku/sudoku.csv',

    "3 million": '../input/3-million-sudoku-puzzles-with-ratings/sudoku-3m.csv'

}

datasets = {

    "1 million": pd.read_csv(files["1 million"]),

    "3 million": pd.read_csv(files["3 million"], index_col='id').sort_values('difficulty', ascending=False),

}
display(datasets['1 million'].head())

# solve_dataframe(datasets['1 million'], count=1,   verbose=3)

# solve_dataframe(datasets['1 million'], count=100, verbose=0)
# solve_dataframe(datasets['1 million'], timeout=4*60*60, verbose=0)
solved_df = solve_dataframe(datasets['3 million'], count=1, verbose=3)
display(datasets['3 million'].head())

print("len(datasets['3 million']) = ", len(datasets['3 million']))

solve_dataframe(datasets['3 million'], count=5, verbose=1).head()
df = datasets['3 million']

solved_df = solve_dataframe(df, timeout=8.8*60*60, count=0, verbose=0)

solved_df.to_csv('3-million-sudoku-puzzles-with-ratings-solve-times.csv')

solved_df