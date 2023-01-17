# 1. Enable Internet in the Kernel (Settings side pane)



# 2. Curl cache may need purged if v0.1.4 cannot be found (uncomment if needed). 

# !curl -X PURGE https://pypi.org/simple/kaggle-environments



# ConnectX environment was defined in v0.1.4

!pip install 'kaggle-environments>=0.1.4'
from kaggle_environments import evaluate, make



env = make("connectx", debug=True)

env.render()
# with this comand you can check values of enviroment

env.configuration

def get_input(user):

    """ 

    This funtion sent to our environment the position Select

       Input: User Name

       Output: value of select column

       

    """

    n=7 # Max number of columns

    input1 = 'Input from player {}: '.format(user)

    while True:

        try:

            print("Enter Value from 1 to 7")

            user_input = int(input(input1))

        except Value_Error:

            print('Invalid input:', user_input)

            continue

        if   user_input <= 0 or user_input > n:

            print('invalid input:', user_input)

       

        else:

            return user_input -1
#https://stackoverflow.com/questions/21641807/python-connect-4?answertab=active#tab-top



def check_winner(observation):

    """

    This function return the value of the winner.

    

    INPUT:  observation 

    OOUTPUT: 1 for user Winner or 2 for Computer Winner 

    """

    

  

    line1 = observation.board[0:7] # bottom row

    line2 = observation.board[7:14]

    line3 = observation.board[14:21]

    line4 = observation.board[21:28]

    line5 = observation.board[28:35]

    line6 = observation.board[35:42]



    board = [line1, line2 , line3, line4, line5, line6] 



    # Check rows for winner

    for row in range(6):

        for col in range(4):

            if (board[row][col] == board[row][col + 1] == board[row][col + 2] ==\

                board[row][col + 3]) and (board[row][col] != 0):

                return board[row][col]  #Return Number that match row



    # Check columns for winner

    for col in range(7):

        for row in range(3):

            if (board[row][col] == board[row + 1][col] == board[row + 2][col] ==\

                board[row + 3][col]) and (board[row][col] != 0):

                return board[row][col]  #Return Number that match column



    # Check diagonal (top-left to bottom-right) for winner



    for row in range(3):

        for col in range(4):

            if (board[row][col] == board[row + 1][col + 1] == board[row + 2][col + 2] ==\

                board[row + 3][col + 3]) and (board[row][col] != 0):

                return board[row][col] #Return Number that match diagonal





    # Check diagonal (bottom-left to top-right) for winner



    for row in range(5, 2, -1):

        for col in range(4):

            if (board[row][col] == board[row - 1][col + 1] == board[row - 2][col + 2] ==\

                board[row - 3][col + 3]) and (board[row][col] != 0):

                return board[row][col] #Return Number that match diagonal



    # No winner: return None

    return None
from IPython.display import display, Image

display(Image(filename='/kaggle/input/playimagen/play.jpg'))
# This agent random chooses a non-empty column.

def my_agent(observation, configuration):

    from random import choice

    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
# observation.board this is a list of 42 elements 7X6
play = False # Change this line to Start Play



# Play as first position against random agent.

trainer = env.train([None, "random"])

observation = trainer.reset()



while not env.done:

    if play:

        my_action = get_input(user = "marco")

        print("My Action", my_action)

        observation, reward, done, info = trainer.step(my_action)

        #print(observation, reward, done, info)

        if (check_winner(observation) == 1):

            print ("You Won, Amazing! \nGAME OVER")

            

        elif (check_winner(observation) == 2):

            print ("The Computer Won! \nGAME OVER")

        env.render(mode="ipython", width=300, height=200, header=False, controls=False)

    else: # Run Ramdom

        my_action = my_agent(observation, env.configuration)

        print("My Action", my_action)

        observation, reward, done, info = trainer.step(my_action)

        if (check_winner(observation) == 1):

            print ("You Won, Amazing! \nGAME OVER")

            

        elif (check_winner(observation) == 2):

            print ("The Computer Won! \nGAME OVER")

        env.render(mode="ipython", width=100, height=90, header=False, controls=False)


env.render(mode="ipython", width=500, height=450)