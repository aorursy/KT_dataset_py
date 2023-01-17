# ConnectX environment was defined in v0.1.6

!pip install 'kaggle-environments>=0.1.6'
from kaggle_environments import evaluate, make, utils
def my_agent(observation, configuration):

    import time

    start = time.time()

    import numpy as np

    prob = np.zeros(configuration.columns)

    from random import choice

    #The amount of time you have to make a move. More time -> Better performance.

    time_limit = configuration.timeout-0.7

    

    def play(board, column, mark, config):

        columns = config.columns

        rows = config.rows

        row = max([r for r in range(rows) if board[column + (r * columns)] == 0])

        board[column + (row * columns)] = mark





    def is_win(board, column, mark, config, has_played=True):

        columns = config.columns

        rows = config.rows

        inarow = config.inarow - 1

        row = (

            min([r for r in range(rows) if board[column + (r * columns)] == mark])

            if has_played

            else max([r for r in range(rows) if board[column + (r * columns)] == 0])

        )



        def count(offset_row, offset_column):

            for i in range(1, inarow + 1):

                r = row + offset_row * i

                c = column + offset_column * i

                if (

                        r < 0

                        or r >= rows

                        or c < 0

                        or c >= columns

                        or board[c + (r * columns)] != mark

                ):

                    return i - 1

            return inarow



        return (

            count(1, 0) >= inarow  # vertical.

            or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.

            or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.

            or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.

        )



    def check_instant_win(_board, _mark, _configuration):

        for column_choice in [c for c in range(_configuration.columns) if _board[c] == 0]:

            won = is_win(_board, column_choice, _mark, _configuration, has_played=False)

            if won:

                return column_choice

        return -1.0

    

    def check_instant_loss(_board, _mark, _configuration):

        for column_choice in [c for c in range(_configuration.columns) if _board[c] == 0]:

            opponent_mark = 1

            if _mark == 1:

                opponent_mark = 2

            lost = is_win(_board, column_choice, opponent_mark, _configuration, has_played=False)

            if lost:

                return column_choice

        return -1.0



    def rand_agent(_board, _mark, _configuration):

        from random import choice

        options = [c for c in range(_configuration.columns) if _board[c] == 0]

        if not options:

            return -1

        return choice(options)



    win = check_instant_win(observation.board,observation.mark,configuration)

    if win != -1.0:

        return int(win)

    loss = check_instant_loss(observation.board,observation.mark,configuration)

    if loss != -1.0:

        return int(loss)



    end = time.time()

    while end-start < time_limit:

        for column_choice in [c for c in range(configuration.columns) if observation.board[c] == 0]:

            dummy_board = observation.board.copy()

            move = observation.mark

            play(dummy_board, column_choice, move, configuration)

            if move == 1:

                move = 2

            else:

                move = 1

            ongoing = True

            while ongoing:

                my_action = rand_agent(dummy_board,move, configuration)

                if my_action == -1:

                    winner = -1

                    ongoing = False

                    continue

                play(dummy_board,my_action,move,configuration)

                if is_win(dummy_board,my_action,move,configuration,has_played=True):

                    winner = move

                    ongoing = False

                    continue

                if move == 1:

                    move = 2

                else:

                    move = 1

            if observation.mark == winner:

                prob[column_choice] = prob[column_choice] + 1.5

            elif winner == -1:

                prob[column_choice] = prob[column_choice] + 1.0

            else:

                prob[column_choice] = prob[column_choice] + 0.0

        end = time.time()

    best_move = int(np.argmax(prob))

    if observation.board[best_move] == 0:

        return best_move

    else:

        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
def mean_reward(rewards):

    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)



# Run multiple episodes to estimate its performance.

print("Reward against negamax with negamax going first:", 1-mean_reward(evaluate("connectx", ["negamax", my_agent], num_episodes=10)))

print("Reward against negamax with MC method going first:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))
import os

import sys

import inspect



with open('submission.py', 'w') as file:

    file.write(inspect.getsource(my_agent))





out = sys.stdout

submission = utils.read_file('/kaggle/working/submission.py')

agent = utils.get_last_callable(submission)

sys.stdout = out



env = make('connectx', debug=True)

env.run([agent, agent])

print('Success!' if env.state[0].status == env.state[1].status == 'DONE' else 'Failed...')