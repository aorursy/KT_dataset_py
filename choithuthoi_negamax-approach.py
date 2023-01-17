!pip install kaggle_environments==0.1.6
from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)

env.render()
def negamax_agent(obs, config):

    import random

    columns = config.columns

    rows = config.rows

    size = rows * columns

    EMPTY = 0

    # Due to compute/time constraints the tree depth must be limited.

    max_depth = 4

    

    def choice(seq):

        return random.choice(seq)

    def play(board, column, mark, config):

        columns = config.columns

        rows = config.rows

        row = max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])

        board[column + (row * columns)] = mark

    def is_win(board, column, mark, config, has_played=True):

        inarow = config.inarow - 1

        row = (

            min([r for r in range(rows) if board[column + (r * columns)] == mark])

            if has_played

            else max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])

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



    def negamax(board, mark, depth):

        moves = sum(1 if cell != EMPTY else 0 for cell in board)



        # Tie Game

        if moves == size:

            return (0, None)



        # Can win next.

        for column in range(columns):

            if board[column] == EMPTY and is_win(board, column, mark, config, False):

                return ((size + 1 - moves) / 2, column)



        # Recursively check all columns.

        best_score = -size

        best_column = None

        for column in range(columns):

            if board[column] == EMPTY:

                # Max depth reached. Score based on cell proximity for a clustering effect.

                if depth <= 0:

                    row = max(

                        [

                            r

                            for r in range(rows)

                            if board[column + (r * columns)] == EMPTY

                        ]

                    )

                    score = (size + 1 - moves) / 2

                    if column > 0 and board[row * columns + column - 1] == mark:

                        score += 1

                    if (

                        column < columns - 1

                        and board[row * columns + column + 1] == mark

                    ):

                        score += 1

                    if row > 0 and board[(row - 1) * columns + column] == mark:

                        score += 1

                    if row < rows - 2 and board[(row + 1) * columns + column] == mark:

                        score += 1

                else:

                    next_board = board[:]

                    play(next_board, column, mark, config)

                    (score, _) = negamax(next_board,

                                         1 if mark == 2 else 2, depth - 1)

                    score = score * -1

                if score > best_score or (score == best_score and choice([True, False])):

                    best_score = score

                    best_column = column



        return (best_score, best_column)



    _, column = negamax(obs.board[:], obs.mark, max_depth)

    if column == None:

        column = choice([c for c in range(columns) if obs.board[c] == EMPTY])

    return column

env.reset()

# Play as the first agent against default "random" agent.

env.run([negamax_agent, "random"])

env.render(mode="ipython", width=500, height=450)
import os

import inspect

def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(negamax_agent, "submission.py")
# Note: Stdout replacement is a temporary workaround.

import sys

out = sys.stdout

submission = utils.read_file("/kaggle/working/submission.py")

agent = utils.get_last_callable(submission)

sys.stdout = out



env = make("connectx", debug=True)

env.run([agent, agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")