# 1. Enable Internet in the Kernel (Settings side pane)



# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 

# !curl -X PURGE https://pypi.org/simple/kaggle-environments



# ConnectX environment was defined in v0.1.6

!pip install 'kaggle-environments>=0.1.6'
from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)

env.render()
def compute(col, matrix_board, rows, mark, inarow):

    heuristique = 0.

    if matrix_board[0][col] != 0:

        return -10.



    opponent = 2

    if mark == 2:

        opponent = 1

    opp_blocks = find_blocks(matrix_board, opponent, inarow)

    if opp_blocks[inarow - 1] > 0:

        return -10.



    new_matrix_board = play(col, matrix_board, rows, mark)



    #print("x"*20)

    #print("xxxxx Calcul heuristique")

    #print(new_matrix_board)



    blocks = find_blocks(new_matrix_board, mark, inarow)



    if blocks[inarow - 1] > 0:

        return 10.

    else:

        for i, b in enumerate(blocks):

            heuristique += b * (i+1) * (i+1)



        heuristique /=  50.

    #print("xx My heuristique : ", heuristique)



    # For the opponent

    opp_heuristique = 0.

    for i, b in enumerate(opp_blocks):

        opp_heuristique += b * (i+1) * (i+1)



    opp_heuristique /=  50.



    #print("xx Opp heuristique : ", opp_heuristique)

    #print("Final heurisique : ", heuristique - opp_heuristique)

    return heuristique - opp_heuristique
def play(col, matrix_board, rows, mark):

    new_matrix_board = matrix_board.copy()

    for r in reversed(range(rows)):

        if new_matrix_board[r][col] == 0:

            new_matrix_board[r][col] = mark

            break

    return new_matrix_board
def contains(small, big):

    for i in range(len(big)-len(small)+1):

        for j in range(len(small)):

            if big[i+j] != small[j]:

                break

        else:

            return i, i+len(small)

    return False
def find_blocks(matrix_board, mark, inarow):

    rows = matrix_board.shape[0]

    cols = matrix_board.shape[1]



    blocks = [0]*inarow



    # horizontal blocks

    for r in range(rows):

        b_size = 0



        if 0 not in matrix_board[r]:

            # line full

            victory_list = [mark]*inarow

            if not contains(victory_list, matrix_board[r]):

                break



        for c in range(cols):

            if matrix_board[r][c] == mark:

                b_size += 1

                # win

                if (b_size == inarow):

                    blocks[b_size - 1] += 1

                    break

                if (b_size > 0) and (c == cols - 1):

                    blocks[b_size - 1] += 1

            else:

                if b_size > 0:

                    blocks[b_size - 1] += 1

                b_size = 0



    # vertical blocks

    matrix_board_transp = matrix_board.transpose()

    rows = matrix_board_transp.shape[0]

    cols = matrix_board_transp.shape[1]

    for r in range(rows):

        b_size = 0

        for c in range(cols):

            if matrix_board_transp[r][c] == mark:

                b_size += 1

                # win

                if (b_size == inarow):

                    blocks[b_size - 1] += 1

                    break

                if (b_size > 0) and (c == cols - 1):

                    blocks[b_size - 1] += 1

            else:

                if b_size > 0:

                    blocks[b_size - 1] += 1

                # found opponent in column: no more weight for following blocks

                if matrix_board_transp[r][c] != 0:

                    break

                b_size = 0



    # diag blocks

    # computes only SouthWest to NorthEast direction diagonals: 

    # won't see winning or losing move with other diagonal direction

    matrix_board_diag = matrix_board.copy()

    rows = matrix_board_diag.shape[0]

    cols = matrix_board_diag.shape[1]

    for k in range(rows + cols -1):

        diag = []

        b_size = 0

        for j in range(k+1):

            i = k - j;

            if (i < rows and j < cols):

                diag.append(matrix_board_diag[i][j])

        diag.reverse()

        b_size = 0

        for i, val in enumerate(diag):              

            if val == mark:

                b_size += 1

                if (b_size > 1 and i == (len(diag) - 1)) or b_size == inarow:

                    blocks[b_size-1] += 1

                    break

            else:

                if b_size > 1:

                    blocks[b_size-1] += 1

                b_size = 0







    #print(matrix_board)

    #print("Blocks : ", blocks)

    return blocks
def get_col_to_play(h_matrix):

    for d in range(len(h_matrix[0])):

        for c in range(len(h_matrix)):

            if h_matrix[c][d].count(10.) > 0:

                return c



    col_to_play = 66

    max_found = -100

    for d in reversed(range(len(h_matrix[0]))):

        for c in range(len(h_matrix)):

            max_at_depth_for_col = max(h_matrix[c][d], default=-100)

            if max_at_depth_for_col > max_found:

                col_to_play = c

                max_found = max_at_depth_for_col

        if max_found > -10.:

            return col_to_play



    return col_to_play
def play_one_best_move(columns, new_matrix_board, rows, mark, inarow):

    import numpy as np

    h_matrix_opp = np.zeros((columns, 1, 0)).tolist()

    for c in range(columns):

        h_opp = compute(c, new_matrix_board, rows, mark, inarow)

        h_matrix_opp[c][0].append(h_opp)

    col_opp = get_col_to_play(h_matrix_opp)

    new_matrix_board = play(col_opp, new_matrix_board, rows, mark)

    #print("New board : {}".format(new_matrix_board))

    return new_matrix_board
def go_trough_moves(h_matrix, matrix_board, cols, rows, mark, inarow, depth, depth_max, h_col):

    opponent = 1 if mark == 2 else 2



    if depth == depth_max:

        return

    else:

        for col in range(cols):

            h = compute(col, matrix_board, rows, mark, inarow)



            if depth == 0:

                h_col = col



            h_matrix[h_col][depth].append(h)

            if h == -10.:

                continue



            new_matrix_board = play(col, matrix_board, rows, mark)



            # Opponent may certainly play...

            new_matrix_board = play_one_best_move(cols, new_matrix_board, rows, opponent, inarow)



            go_trough_moves(h_matrix, new_matrix_board, cols, rows, mark, inarow, depth+1, depth_max, h_col)
# This agent uses a sort of a* algorithm

def my_agent(observation, configuration):

    import numpy as np

    

    # Number of Columns on the Board.

    columns = configuration.columns

    # Number of Rows on the Board.

    rows = configuration.rows

    # Number of Checkers "in a row" needed to win.

    inarow = configuration.inarow

    # The current serialized Board (rows x columns).

    board = observation.board[:]

    # Which player the agent is playing as (1 or 2).

    mark = observation.mark

    

    if np.sum(board) == 0:

        return int(columns / 2)

    

    matrix_board = np.zeros((rows, columns))

    for row in range(rows):

        for col in range(columns):

            matrix_board[row][col] = board[(row*columns) + col]

    

    depth = 0

    max_depth = 3

    h = 0.01

    h_matrix = np.zeros((columns, max_depth-1, 0)).tolist()

    h_matrix = [[ [] for col in range(max_depth)] for row in range(columns)] 



    go_trough_moves(h_matrix, matrix_board, columns, rows, mark, inarow, depth, max_depth, 0)

    #print("Final h_matrix : ", h_matrix)

             

    return get_col_to_play(h_matrix)

env.reset()



# Play as the first agent against default "random" or negamax agent.

env.run([my_agent, my_agent])

env.render(mode="ipython", width=500, height=450)
# Play as first position against random agent.

trainer = env.train([None, "random"])



observation = trainer.reset()



while not env.done:

    my_action = my_agent(observation, env.configuration)

    print("My Action", my_action)

    observation, reward, done, info = trainer.step(my_action)

    env.render(mode="ipython", width=100, height=90, header=False, controls=False)

    print("Reward: ", reward)

env.render()
def mean_reward(rewards):

    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)



# Run multiple episodes to estimate its performance.

#print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))

#print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))
# "None" represents which agent you'll manually play as (first or second player).

env.play([my_agent, None], width=500, height=450)
import inspect

import os



# can get function reference through 'globals()[func_name]'

import_functions = [

    compute,

    play,

    contains,

    find_blocks,

    get_col_to_play,

    play_one_best_move,

    go_trough_moves

]



def write_agent_to_file(function, file, import_functions=[]):

    # get source and transform into list of lines

    function_source = inspect.getsource(function)

    function_source = function_source.split("\n")



    for func in import_functions:

        import_source = inspect.getsource(func)

        # add tab after every new line

        import_source = import_source.split("\n")

        import_source = ["    " + line for line in import_source]

        # insert new function after function definition

        function_source.insert(1, "\n".join(import_source))

    

    function_source = "\n".join(function_source)

    

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(function_source)

        print(function_source) # print written code



write_agent_to_file(my_agent, "submission.py", import_functions=import_functions)
# Note: Stdout replacement is a temporary workaround.

import sys

out = sys.stdout

submission = utils.read_file("/kaggle/working/submission.py")

agent = utils.get_last_callable(submission)

sys.stdout = out



env = make("connectx", debug=True)

env.run([agent, agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")