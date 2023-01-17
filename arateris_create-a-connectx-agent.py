from kaggle_environments import make, evaluate



# Create the game environment

# Set debug=True to see the errors if your agent refuses to run

env = make("connectx", debug=True)
def my_agent(obs, config):

    

    ################################

    # Imports and helper functions #

    ################################

    

    import numpy as np

    import random



    # Gets board at next step if agent drops piece in selected column

    def drop_piece(grid, col, piece, config):

        next_grid = grid.copy()

        for row in range(config.rows-1, -1, -1):

            if next_grid[row][col] == 0:

                break

        next_grid[row][col] = piece

        return next_grid



    # Returns True if dropping piece in column results in game win

    def check_winning_move(grid, config, col, piece):

        return  check_inarow_move(grid, config, col, piece, config.inarow)

    

    # Returns True if dropping piece in column results in inarow

    def check_inarow_move(grid, config, col, piece, inarow):

        # Convert the board to a 2D grid

#         grid = np.asarray(obs.board).reshape(config.rows, config.columns)

        next_grid = drop_piece(grid, col, piece, config)

        # horizontal

        for row in range(config.rows):

            for col in range(config.columns-(inarow-1)):

                window = list(next_grid[row,col:col+inarow])

                if window.count(piece) == inarow:

                    return True

        # vertical

        for row in range(config.rows-(inarow-1)):

            for col in range(config.columns):

                window = list(next_grid[row:row+inarow,col])

                if window.count(piece) == inarow:

                    return True

        # positive diagonal

        for row in range(config.rows-(inarow-1)):

            for col in range(config.columns-(inarow-1)):

                window = list(next_grid[range(row, row+inarow), range(col, col+inarow)])

                if window.count(piece) == inarow:

                    return True

        # negative diagonal

        for row in range(inarow-1, config.rows):

            for col in range(config.columns-(inarow-1)):

                window = list(next_grid[range(row, row-inarow, -1), range(col, col+inarow)])

                if window.count(piece) == inarow:

                    return True

        return False

    

    def get_valid_moves(config, grid):

        return [col for col in range(config.columns) if grid[0][col] == 0]

    

    def board_to_grid(config, board):

        return np.asarray(board).reshape(config.rows, config.columns)

    

    def grid_to_board(config, grid):

        return grid.reshape(-1).tolist()

    

    def gives_two_winning_moves(grid, config, col, piece):

        

        next_grid = drop_piece(grid, move, piece, config)

        next_valid_moves = get_valid_moves(config, next_grid)

        total_win_moves=0

        for next_move in next_valid_moves:

            if check_winning_move(next_grid, config, next_move, piece): 

                total_win_moves = total_win_moves+1

                if total_win_moves>1:

                    return True   

        return False

    

    def check_give_opp_winning_move(grid, config, move, piece):

        # Should return True if this move gives a winning position to the opponent

        

        next_grid = drop_piece(grid, move, piece, config)

        next_valid_moves = get_valid_moves(config, next_grid)

        opp_piece = 1 if piece==2 else 2

        

        for next_move in next_valid_moves:

            if check_winning_move(next_grid, config, next_move, opp_piece):

                return True

        return False

        

    def print_debug(str):

        if DEBUG: print(str)

    

    #########################

    # Agent makes selection #

    #########################

    

    DEBUG = False

    

    agent_mark = obs.mark

    opp_mark = 1 if agent_mark==2 else 2

    grid = board_to_grid(config, obs.board)

    

    valid_moves = get_valid_moves(config, grid)

    

    print_debug('valid moves:')

    print_debug(valid_moves)

    #if first move, play center !

    if grid.sum().sum()==0 : 

        print_debug('first move')

        return int((config.columns/2))

    

    # Check for winning move

    for move in valid_moves:

        if check_winning_move(grid, config, move, agent_mark):

            print_debug('win move')

            return move     

        

    # Check for opponent winning move

    for move in valid_moves:

        if check_winning_move(grid, config, move, opp_mark):

            print_debug('avoid opponent win move')

            return move     

        

    # Check if a valid play gives one wining move to opponent.

    selected_moves = []

    for move in valid_moves:

        if not check_give_opp_winning_move(grid, config, move, agent_mark):

            selected_moves.append(move)        



    print_debug('selected moves:')

    print_debug(selected_moves)            

    if len(selected_moves)==0:

        print_debug('no selected moves')

        return random.choice(valid_moves) #loosing move



    # Check for a place giving me 2 winning moves

        #if any, play that to win.

    for move in selected_moves:

        if gives_two_winning_moves(grid, config, move, agent_mark):

            print_debug('gives 2 win moves') 

            return move

    

    #try connect 3   

    # TODO : need to check all possible 3s and randomize/count how many 3s one piece gives

    for move in selected_moves:

        if check_inarow_move(grid, config, move, agent_mark, inarow=3):

            print_debug('connecting 3') 

            return move

        

    #avoid opp connect 3        

    # TODO : need to check all possible 3s and randomize/count how many 3s one piece gives

    for move in selected_moves:

        if check_inarow_move(grid, config, move, opp_mark, inarow=3):

            print_debug('avoid opp connecting 3') 

            return move

        

    #try connect 2    

    for move in selected_moves:

        if check_inarow_move(grid, config, move, agent_mark, inarow=2):

            print_debug('connecting 2') 

            return move

        

    #avoid opp connect 2 ?

    for move in selected_moves:

        if check_inarow_move(grid, config, move, opp_mark, inarow=2):

            print_debug('avoid opp connect 2') 

            return move

        

    if len(selected_moves)>0:

        print_debug('random') 

        return random.choice(selected_moves)

    

    print_debug('should not happen !') 

    return random.choice(valid_moves) #loosing move
# Play as first position against random agent.

trainer = env.train([None, "negamax"])



observation = trainer.reset()



while not env.done:

    my_action = my_agent(observation, env.configuration)

    print("My Action", my_action)

    observation, reward, done, info = trainer.step(my_action)

    env.render(mode="ipython", width=100, height=90, header=False, controls=False)

env.render()
# Agents play one game round

out = env.run([my_agent, 'negamax'])
# Show the game

env.render(mode="ipython")
# To learn more about the evaluate() function, check out the documentation here: (insert link here)

def get_win_percentages(agent1, agent2, n_rounds=100):

    import random

    import numpy as np

    # Use default Connect Four setup

    config = {'rows': 6, 'columns': 7, 'inarow': 4}

    # Agent 1 goes first (roughly) half the time          

    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)

    # Agent 2 goes first (roughly) half the time      

    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]

    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,0])/len(outcomes)*100, 1))

    print("Agent 2 Win Percentage:", np.round(outcomes.count([0,1])/len(outcomes)*100, 1))

    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0.5]))

    print("Number of Invalid Plays by Agent 2:", outcomes.count([0.5, None]))

    print("Number of Draws (in {} game rounds):".format(n_rounds), outcomes.count([0.5, 0.5]))
get_win_percentages(agent1=my_agent, agent2='random', n_rounds=100)
get_win_percentages(agent1=my_agent, agent2='negamax', n_rounds=100)
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")
import sys

from kaggle_environments import utils



out = sys.stdout

submission = utils.read_file("/kaggle/working/submission.py")

agent = utils.get_last_callable(submission)

sys.stdout = out



env = make("connectx", debug=True)

env.run([agent, agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")