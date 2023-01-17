# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 1. Enable Internet in the Kernel (Settings side pane)



# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 

# !curl -X PURGE https://pypi.org/simple/kaggle-environments



# ConnectX environment was defined in v0.1.6

!pip install 'kaggle-environments>=0.1.6'
from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)

env.render()


def swarm(obs, conf):

    def send_scout_carrier(x, y):

        """ send scout carrier to explore current cell and, if possible, cell above """

        points = send_scouts(x, y)

        # if cell above exists

        if y > 0:

            cell_above_points = send_scouts(x, y - 1)

            # cell above points have lower priority

            if points < m1 and points < (cell_above_points - 1):

                # current cell's points will be negative

                points -= cell_above_points

        return points

    

    def send_scouts(x, y):

        """ send scouts to get points from all axes of the cell """

        axes = explore_axes(x, y)

        points = combine_points(axes)

        return points

        

    def explore_axes(x, y):

        """

            find points, marks, zeros and amount of in_air cells of all axes of the cell,

            "NE" = North-East etc.

        """

        return {

            "NE -> SW": [

                explore_direction(x, lambda z : z + 1, y, lambda z : z - 1),

                explore_direction(x, lambda z : z - 1, y, lambda z : z + 1)

            ],

            "E -> W": [

                explore_direction(x, lambda z : z + 1, y, lambda z : z),

                explore_direction(x, lambda z : z - 1, y, lambda z : z)

            ],

            "SE -> NW": [

                explore_direction(x, lambda z : z + 1, y, lambda z : z + 1),

                explore_direction(x, lambda z : z - 1, y, lambda z : z - 1)

            ],

            "S -> N": [

                explore_direction(x, lambda z : z, y, lambda z : z + 1),

                explore_direction(x, lambda z : z, y, lambda z : z - 1)

            ]

        }

    

    def explore_direction(x, x_fun, y, y_fun):

        """ get points, mark, zeros and amount of in_air cells of this direction """

        # consider only opponents mark

        mark = 0

        points = 0

        zeros = 0

        in_air = 0

        for i in range(one_mark_to_win):

            x = x_fun(x)

            y = y_fun(y)

            # if board[x][y] is inside board's borders

            if y >= 0 and y < conf.rows and x >= 0 and x < conf.columns:

                # mark of the direction will be the mark of the first non-empty cell

                if mark == 0 and board[x][y] != 0:

                    mark = board[x][y]

                # if board[x][y] is empty

                if board[x][y] == 0:

                    zeros += 1

                    if (y + 1) < conf.rows and board[x][y + 1] == 0:

                        in_air += 1

                elif board[x][y] == mark:

                    points += 1

                # stop searching for marks in this direction

                else:

                    break

        return {

            "mark": mark,

            "points": points,

            "zeros": zeros,

            "in_air": in_air

        }

    

    def combine_points(axes):

        """ combine points of different axes """

        points = 0

        # loop through all axes

        for axis in axes:

            # if mark in both directions of the axis is the same

            # or mark is zero in one or both directions of the axis

            if (axes[axis][0]["mark"] == axes[axis][1]["mark"]

                    or axes[axis][0]["mark"] == 0 or axes[axis][1]["mark"] == 0):

                # combine points of the same axis

                points += evaluate_amount_of_points(

                              axes[axis][0]["points"] + axes[axis][1]["points"],

                              axes[axis][0]["zeros"] + axes[axis][1]["zeros"],

                              axes[axis][0]["in_air"] + axes[axis][1]["in_air"],

                              m1,

                              m2,

                              axes[axis][0]["mark"]

                          )

            else:

                # if marks in directions of the axis are different and none of those marks is 0

                for direction in axes[axis]:

                    points += evaluate_amount_of_points(

                                  direction["points"],

                                  direction["zeros"],

                                  direction["in_air"],

                                  m1,

                                  m2,

                                  direction["mark"]

                              )

        return points

    

    def evaluate_amount_of_points(points, zeros, in_air, m1, m2, mark):

        """ evaluate amount of points in one direction or entire axis """

        # if points + zeros in one direction or entire axis >= one_mark_to_win

        # multiply amount of points by one of the multipliers or keep amount of points as it is

        if (points + zeros) >= one_mark_to_win:

            if points >= one_mark_to_win:

                points *= m1

            elif points == two_marks_to_win:

                points = points * m2 + zeros - in_air

            else:

                points = points + zeros - in_air

        else:

            points = 0

        return points





    #################################################################################

    # one_mark_to_win points multiplier

    m1 = 100

    # two_marks_to_win points multiplier

    m2 = 10

    # define swarm's mark

    swarm_mark = obs.mark

    # define opponent's mark

    opp_mark = 2 if swarm_mark == 1 else 1

    # define one mark to victory

    one_mark_to_win = conf.inarow - 1

    # define two marks to victory

    two_marks_to_win = conf.inarow - 2

    # define board as two dimensional array

    board = []

    for column in range(conf.columns):

        board.append([])

        for row in range(conf.rows):

            board[column].append(obs.board[conf.columns * row + column])

    # define board center

    board_center = conf.columns // 2

    # start searching for the_column from board center

    x = board_center

    # shift to left/right from board center

    shift = 0

    # THE COLUMN !!!

    the_column = {

        "x": x,

        "points": float("-inf")

    }

    

    # searching for the_column

    while x >= 0 and x < conf.columns:

        # find first empty cell starting from bottom of the column

        y = conf.rows - 1

        while y >= 0 and board[x][y] != 0:

            y -= 1

        # if column is not full

        if y >= 0:

            # send scout carrier to get points

            points = send_scout_carrier(x, y)

            # evaluate which column is THE COLUMN !!!

            if points > the_column["points"]:

                the_column["x"] = x

                the_column["points"] = points

        # shift x to right or left from swarm center

        shift *= -1

        if shift >= 0:

            shift += 1

        x = board_center + shift

    

    return the_column["x"]
env.reset()

# Play as the first agent against "negamax" agent.

env.run([swarm, swarm])

#env.run([swarm, "negamax"])

env.render(mode="ipython", width=500, height=450)
# Play as first position against negamax agent.

trainer = env.train([None, "negamax"])



observation = trainer.reset()



while not env.done:

    my_action = swarm(observation, env.configuration)

    print("My Action", my_action)

    observation, reward, done, info = trainer.step(my_action)

    # env.render(mode="ipython", width=100, height=90, header=False, controls=False)

env.render()
def mean_reward(rewards):

    return "{0} episodes: won {1}, lost {2}, draw {3}".format(

                                                           len(rewards),

                                                           sum(1 if r[0] > 0 else 0 for r in rewards),

                                                           sum(1 if r[1] > 0 else 0 for r in rewards),

                                                           sum(r[0] == r[1] for r in rewards)

                                                       )



# Run multiple episodes to estimate its performance.

print("Swarm vs Random Agent", mean_reward(evaluate("connectx", [swarm, "random"], num_episodes=10)))

print("Swarm vs Negamax Agent", mean_reward(evaluate("connectx", [swarm, "negamax"], num_episodes=10)))
# "None" represents which agent you'll manually play as (first or second player).

env.play([swarm, None], width=500, height=450)

#env.play([None, swarm], width=500, height=450)
# Two random agents play one game round

env.run(["random", "random"])



# Show the game

env.render(mode="ipython")
def agent_random(obs, config):

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    return random.choice(valid_moves)



# Selects middle column

def agent_middle(obs, config):

    return config.columns//2



# Selects leftmost valid column

def agent_leftmost(obs, config):

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    return valid_moves[0]
# Agents play one game round

env.run([agent_leftmost, agent_random])



# Show the game

env.render(mode="ipython")
def get_win_percentages(agent1, agent2, n_rounds=100):

    # Use default Connect Four setup

    config = {'rows': 6, 'columns': 7, 'inarow': 4}

    # Agent 1 goes first (roughly) half the time          

    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)

    # Agent 2 goes first (roughly) half the time      

    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]

    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))

    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))

    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))

    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))
get_win_percentages(agent1=agent_middle, agent2=agent_random)