# 1. Enable Internet in the Kernel (Settings side pane)



# 2. Curl cache may need purged if v0.1.5 cannot be found (uncomment if needed). 

# !curl -X PURGE https://pypi.org/simple/kaggle-environments



# ConnectX environment was defined in v0.1.5

!pip install 'kaggle-environments>=0.1.6'
from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)

env.render()
# connect my 4 or stop opponemt's 4

# choose the central column if it is not full

# else, randomly choose a not-full column



def my_agent(observation, configuration):

    from random import choice # Note: import must be put inside function!

    empty = 0

    me = observation.mark # player mark: either 1 or 2

    enemy = 3 - me # 1 + 2 = 3   

    columns = configuration.columns # 7

    central_column = (columns - 1) // 2

    rows = configuration.rows # 6

    board = observation.board    

    col = 0 # initialized to 0

    

    # get available columns

    available_columns = []

    for col in range(columns):

        if (board[col] == 0):

            available_columns.append(col)

    

    # get available rows

    available_rows = [-1] * columns # -1 means no available row

    for col in available_columns:

        available_row = 0    

        for row in range(rows):            

            location = (row * columns) + col 

            if (board[location] == 0):

                available_row = row

            else:

                break

        available_rows[col] = available_row



    def get_up(number, col):

        row = available_rows[col]

        location = (row * columns) + col

        new_row = row - number

        if (0 <= new_row < rows):

            new_location = location - number*columns        

            new = board[new_location]

        else:

            # outside of board

            new = -1

        return new



    def get_down(number, col):

        row = available_rows[col]

        location = (row * columns) + col

        new_row = row + number

        if (new_row < rows):

            new_location = location + number*columns        

            new = board[new_location]

        else:

            # outside of board

            new = -1

        return new



    def get_right(number, col):

        row = available_rows[col]

        location = (row * columns) + col

        new_col = col + number

        if (new_col < columns):

            new_location = location + number        

            new = board[new_location]

        else:

            # outside of board

            new = -1

        return new



    def get_left(number, col):

        row = available_rows[col]

        location = (row * columns) + col

        new_col = col - number

        if (new_col >= 0):

            new_location = location - number        

            new = board[new_location]

        else:

            # outside of board

            new = -1

        return new



    def get_left_down(number, col):

        row = available_rows[col]

        location = (row * columns) + col

        new_col = col - number

        new_row = row + number

        if ((new_col >= 0) and (new_row < rows)):

            new_location = location + number*columns - number        

            new = board[new_location]

        else:

            # outside of board

            new = -1

        return new



    def get_right_down(number, col):

        row = available_rows[col]

        location = (row * columns) + col

        new_col = col + number

        new_row = row + number

        if ((new_col < columns) and (new_row < rows)):

            new_location = location + number*columns + number        

            new = board[new_location]

        else:

            # outside of board

            new = -1

        return new



    def get_left_up(number, col):

        row = available_rows[col]

        location = (row * columns) + col

        new_col = col - number

        new_row = row - number

        if ((new_col >= 0) and (new_row >= 0)):

            new_location = location - number*columns - number        

            new = board[new_location]

        else:

            # outside of board

            new = -1

        return new



    def get_right_up(number, col):

        row = available_rows[col]

        location = (row * columns) + col

        new_col = col + number

        new_row = row - number

        if ((new_col < columns) and (new_row >= 0)):

            new_location = location - number*columns + number        

            new = board[new_location]

        else:

            # outside of board

            new = -1

        return new



    def connect(marker, number):

        for col in available_columns:

            up1 = get_up(1, col)

            up2 = get_up(2, col)

            up3 = get_up(3, col)

        

            down1 = get_down(1, col)

            down2 = get_down(2, col)

            down3 = get_down(3, col)

        

            right1 = get_right(1, col)

            right2 = get_right(2, col)

            right3 = get_right(3, col)            

            

            left1 = get_left(1, col)

            left2 = get_left(2, col)

            left3 = get_left(3, col)            



            left_down1 = get_left_down(1, col)

            left_down2 = get_left_down(2, col)

            left_down3 = get_left_down(3, col)            



            right_down1 = get_right_down(1, col)

            right_down2 = get_right_down(2, col)

            right_down3 = get_right_down(3, col)            



            left_up1 = get_left_up(1, col)

            left_up2 = get_left_up(2, col)

            left_up3 = get_left_up(3, col)            



            right_up1 = get_right_up(1, col)

            right_up2 = get_right_up(2, col)

            right_up3 = get_right_up(3, col)            

            

            if (number == 4): # connect my 4 or prevent enemy 4

                # down 3

                if ((down1 == marker) and (down2 == marker) and (down3 == marker)):

                    return col

        

                # left 3

                if ((left1 == marker) and (left2 == marker) and (left3 == marker)):

                    return col

        

                # left 2 right 1

                if ((left1 == marker) and (left2 == marker) and (right1 == marker)):

                    return col

        

                # left 1 right 2

                if ((left1 == marker) and (right1 == marker) and (right2 == marker)):

                    return col



                # right 3

                if ((right1 == marker) and (right2 == marker) and (right3 == marker)):

                    return col

        

                # left_down 3

                if ((left_down1 == marker) and (left_down2 == marker) and (left_down3 == marker)):

                    return col

        

                # left_down 2 right_up 1

                if ((left_down1 == marker) and (left_down2 == marker) and (right_up1 == marker)):

                    return col

        

                # left_down 1 right_up 2

                if ((left_down1 == marker) and (right_down1 == marker) and (right_down2 == marker)):

                    return col

        

                # right_up 3

                if ((right_up1 == marker) and (right_up2 == marker) and (right_up3 == marker)):

                    return col

    

                # right_down 3

                if ((right_down1 == marker) and (right_down2 == marker) and (right_down3 == marker)):

                    return col



                # right_down 2 left_up 1

                if ((right_down1 == marker) and (right_down2 == marker) and (left_up1 == marker)):

                    return col



                # right_down 1 left_up 2

                if ((right_down1 == marker) and (left_up1 == marker) and (left_up2 == marker)):

                    return col



                # left_up 3

                if ((left_up1 == marker) and (left_up2 == marker) and (left_up3 == marker)):

                    return col

                

            elif (number == 3): # connect my 3 or prevent enemy 3

                # two-way 3

                # empty 1 left 2 empty 1

                # empty 1 left 2 empty 1 

                if ((left3 == empty) and (left2 == marker) and (left1 == marker) and (right1 == empty)):

                    return col



                # empty 1 right 2 empty 1

                if ((left1 == empty) and (right1 == marker) and (right2 == marker) and (right3 == empty)):

                    return col



                # empty 1 left 1 right 1 empty 1 

                if ((left2 == empty) and (left1 == marker) and (right1 == marker) and (right2 == empty)):

                    return col



                # empty 1 left_down 2 empty 1 

                if ((left_down3 == empty) and (left_down2 == marker) and (left_down1 == marker) and (right_up1 == empty)):

                    return col



                # empty 1 right_down 2 empty 1 

                if ((left_up1 == empty) and (right_down1 == marker) and (right_down2 == marker) and (right_down3 == empty)):

                    return col



                # empty 1 left_up 2 empty 1

                if ((left_up3 == empty) and (left_up2 == marker) and (left_up1 == marker) and (right_down1 == empty)):

                    return col

        

                # empty 1 right_up 2 empty 1

                if ((left_down1 == empty) and (right_up1 == marker) and (right_up2 == marker) and (right_up3 == empty)):

                    return col

                

                # one-way 3

                # empty 1 down 2

                if ((up1 == empty) and (down1 == marker) and (down2 == marker) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col



                # empty 1 left 2

                if ((left3 == empty) and (left2 == marker) and (left1 == marker)):

                    return col



                # left 2 empty 1 

                if ((left2 == marker) and (left1 == marker) and (right1 == empty) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col



                # empty 1 right 2

                if ((left1 == empty) and (right1 == marker) and (right2 == marker) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col



                # right 2 empty 1 

                if ((right1 == marker) and (right2 == marker) and (right3 == empty) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col



                # empty 1 left 1 right 1

                if ((left2 == empty) and (left1 == marker) and (right1 == marker) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col



                # left 1 right 1 empty 1 

                if ((left1 == marker) and (right1 == marker) and (right2 == empty) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col



                # empty 1 left_down 2

                if ((left_down3 == empty) and (left_down2 == marker) and (left_down1 == marker) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col



                # left_down 2 empty 1 

                if ((left_down2 == marker) and (left_down1 == marker) and (right_up1 == empty) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col



                # empty 1 right_down 2

                if ((left_up1 == empty) and (right_down1 == marker) and (right_down2 == marker) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col



                # right_down 2 empty 1 

                if ((right_down1 == marker) and (right_down2 == marker) and (right_down3 == empty) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col



                # empty 1 left_up 2

                if ((left_up3 == empty) and (left_up2 == marker) and (left_up1 == marker) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col

        

                # left_up 2 empty 1 

                if ((left_up2 == marker) and (left_up1 == marker) and (right_down1 == empty) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col

        

                # empty 1 right_up 2

                if ((left_down1 == empty) and (right_up1 == marker) and (right_up2 == marker) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col

                

                # right_up 2 empty 1 

                if ((right_up1 == marker) and (right_up2 == marker) and (right_up3 == empty) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col

                

            elif (number == 2): # connect my 2 or prevent enemy 2

                # empty 2 down 1

                if ((up2 == empty) and (up1 == empty) and (down1 == marker) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col

    

                # empty 2 left 1

                if ((left3 == empty) and (left2 == empty) and (left1 == marker) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col

        

                # left 1 empty 2 

                if ((left1 == marker) and (right1 == empty) and (right2 == empty) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col

        

                # empty 2 right 1

                if ((left2 == empty) and (left1 == empty) and (right1 == marker) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col

        

                # right 1 empty 2 

                if ((right1 == marker) and (right2 == empty) and (right3 == empty) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col

        

                # empty 2 left_down 1

                if ((left_down3 == empty) and (left_down2 == empty) and (left_down1 == marker) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col

    

                # left_down 1 empty 2 

                if ((left_down1 == marker) and (right_up1 == empty) and (right_up2 == empty) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col

    

                # empty 2 right_down 1

                if ((left_up2 == empty) and (left_up1 == empty) and (right_down1 == marker) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col



                # right_down 1 empty 2 

                if ((right_down1 == marker) and (right_down2 == empty) and (right_down3 == empty) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col



                # empty 2 left_up 1

                if ((left_up3 == empty) and (left_up2 == empty) and (left_up1 == marker) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col

        

                # left_up 1 empty 2 

                if ((left_up1 == marker) and (right_down1 == empty) and (right_down2 == empty) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col

        

                # empty 2 right_up 1

                if ((left_down2 == empty) and (left_down1 == empty) and (right_up1 == marker) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col

            

                # right_up 1 empty 2 

                if ((right_up1 == marker) and (right_up2 == empty) and (right_up3 == empty) and (left1 != (3 - marker)) and (right1 != (3 - marker))):

                    return col

            

        col = -1 # no col is found

        return col



    col = -1

    col = connect(me, 4) # connect my 4

    if (col != -1): 

        return col

    col = connect(enemy, 4) # prevent enemy 4

    if (col != -1): 

        return col

    col = connect(me, 3) # connect my 3

    if (col != -1): 

        return col

    col = connect(enemy, 3) # prevent enemy 3

    if (col != -1): 

        return col

    col = connect(me, 2) # connect my 2

    if (col != -1): 

        return col

    col = connect(enemy, 2) # prevent enemy 2

    if (col != -1): 

        return col



    # choose central column if it is not full

    if (board[central_column] == 0):

        return central_column

    

    # choose random column

    col = choice([c for c in available_columns])

    return col
env.reset()

# Play as the first agent against default "random" agent.

#env.run([my_agent, "random"])



# "negamax" is very strong!

env.run([my_agent, "negamax"])



# manually play against your agent!

# Play as the second play (in ipython notebooks only).

#env.play([my_agent, None])



env.render(mode="ipython", width=400, height=360)
# Play as first position against random agent.

trainer = env.train([None, "random"])



observation = trainer.reset()



while not env.done:

    my_action = my_agent(observation, env.configuration)

    print("My Action", my_action)

    observation, reward, done, info = trainer.step(my_action)

    env.render(mode="ipython", width=100, height=90, header=False, controls=False)

env.render()
def mean_reward(rewards):

    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)



# Run multiple episodes to estimate its performance.

print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))

print("Random Agent vs My Agent:", mean_reward(evaluate("connectx", ["random", my_agent], num_episodes=10)))

print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))

print("Negamax Agent vs My Agent:", mean_reward(evaluate("connectx", ["negamax", my_agent], num_episodes=10)))
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")
# Note: Stdout replacement is a temporary workaround.

import sys

out = sys.stdout

submission = utils.read_file("/kaggle/working/submission.py")

agent = utils.get_last_callable(submission)

sys.stdout = out



env = make("connectx", debug=True)

env.run([agent, agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")