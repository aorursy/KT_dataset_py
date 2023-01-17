!pip install 'kaggle-environments>=0.1.6'
from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)

env.render()
LOG = True



# This agent random chooses a non-empty column.

def random_agent(observation, configuration):

    from random import choice

    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])



def constant_agent(observation, configuration):

    return 4





# This agent random chooses a non-empty column.

def first_free_agent(observation, configuration):

    import numpy as np

    def column_is_full(column, board, configuration):

        return board[0][column] != 0

        

    #board[row, column] simplifies data access

    board = np.reshape(observation.board, (configuration.rows, configuration.columns))

    

    first_not_full = 0

    for col in range(configuration.columns):   

        full = column_is_full(col, board, configuration)

        # print("Column", col, "is full", full)

        if not full:

            first_not_full = col

            break

    

    # print(board)

    return first_not_full

                



def timo_agent(observation, configuration):

    from random import choice

    import numpy as np

    

    # 0. Helper functions

    def is_win(board, column, columns, rows, inarow, mark, has_played=True):

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

    

    def column_is_full(column, board):

        return board[0][column] != 0

    

    # setup

    playable_columns = [c for c in range(configuration.columns) if observation.board[c] == 0]

    board = np.reshape(observation.board, (configuration.rows, configuration.columns))

    me = observation.mark

    other_player = 1 if observation.mark == 2 else 2

    

    #print("--> Playable", playable_columns)



    # 1. If you can win with any move, just make that move

    for col in playable_columns:

        if (is_win(observation.board, col, configuration.columns, configuration.rows, configuration.inarow - 1, me, False)):

            #print("==> (1) Play", col)

            return col

        

    # 2. If other player would win with that move, the play that one

    for col in playable_columns:

        if (is_win(observation.board, col, configuration.columns, configuration.rows, configuration.inarow - 1, other_player, False)):

            #print("==> (2) Play", col)

            return col

        

    # 3. Calculate a score for each column and return best score

    scores = []

    for col in playable_columns:

        score = 0

        

        for get_inarow in range(configuration.inarow - 1):

            if (is_win(observation.board, col, configuration.columns, configuration.rows, get_inarow, observation.mark, False)):

                #print("--> (2) Col", col, "get_inarow", get_inarow, "yay")

                score += (get_inarow + 1)

            #print("--> (2) Col", col, "get_inarow", get_inarow, "noo")

        

        scores.append([col, score])

        #print("--> (3) Col", col, "score", score)

    

    col = 0

    max_score = 0

    for score in scores:

        if score[1] > max_score:

            max_score = score[1]

            col = score[0]

    

    #print("==> (3) Play", col)

    return col

        

    # X. Otherwise return a random one

    #col = choice(playable_columns)

    #print("==> (X) Play", col)

    #return col



    





# Play one step

#trainer = env.train([None, "random"])

trainer = env.train([None, "negamax"])

observation = trainer.reset()



print(observation) 



env.render()



while not env.done:

    action = timo_agent(observation, env.configuration)

    print("Timo plays", action)

    observation, reward, done, info = trainer.step(action)

    env.render()

    

print("Timo got", reward);

env.render(mode="ipython", width=500, height=450)
env.reset()

env.run([timo_agent, "random"])

env.render(mode="ipython", width=500, height=450)
agents = {

    "Timo": timo_agent,

    "Constant": constant_agent,

    "Negamax": "negamax",

    "Random": "random"

}

    

print(agents)



num_episodes = 2

points = {}

for name in agents:

    points[name] = 0

print(points)



for name_1 in agents:

    for name_2 in agents:

        if (name_1 == name_2):

            continue

        env.reset()

        result = evaluate("connectx", [agents[name_1], agents[name_2]], num_episodes=num_episodes)

        

        #for r in result:

        #    print(r)

        name_1_points = sum([r[0] for r in result if r[0] is not None])

        name_2_points = sum([r[1] for r in result if r[1] is not None])

        

        print("Playing", num_episodes, ":", name_1, "=", name_1_points, "points /", name_2, "=", name_2_points, "points")

        points[name_1] += name_1_points

        points[name_2] += name_2_points



print(points)    
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(timo_agent, "submission.py")



# write other test file

with open("testfile.txt", "a" if os.path.exists("testfile.txt") else "w") as f:

        f.write("Teeest")

# Note: Stdout replacement is a temporary workaround.

import sys

out = sys.stdout

submission = utils.read_file("/kaggle/working/submission.py")

agent = utils.get_last_callable(submission)

sys.stdout = out



env = make("connectx", debug=True)

env.run([agent, agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")