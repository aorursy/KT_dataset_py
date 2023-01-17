%matplotlib notebook
# enable animations
from shutil import copyfile
copyfile(src="../input/q-funcs/qlearning_functions.py",
         dst="../working/qlearning_functions.py")
from qlearning_functions import *
my_board = new_board()

show(my_board)
my_board = new_board()
print(my_board)
show(my_board,
     helpers=True)
test_boards = ["         ",
               "X O XO OX",
               "OXOXXOXOX" ]

for board in test_boards:
    show(board)
    print('evaluator says:', evaluate(board))
my_board = 'X  OOO XX'
flip_board(my_board)
e_init = 0.7     # how much to explore at the start (1 => all exploration)
e_terminal = 0   # how much to exploit at the end   (0 => all exploitation)

simulate_e_greedy(e_init, e_terminal)
get_move(new_board(), epsilon=1)
simulate_game( epsilon_x = 1 ,
               epsilon_o = 1 ,
               slow_down = 3 ) # increase this to make the game slower
qlearn_flow()
steps, winner = simulate_game(verb=True)

backpropagate(steps, winner, verb=True, wait_seconds=3) # <- increase wait_seconds to slow down
# run as-is, or set your own hyperparameters:
visualize_learning(lrate=.1,          # how quickly new values replace old values
                   discount=.9,       # how important are future rewards 
                   init_e=.8,         # maximum epsilon value
                   batches=5,         # number of times to shrink epsilon
                   sims_per_batch=50, # games per batch, per game type (3)
                   update_freq=5,     # how often to redraw plots (lower=>slower)
                   boards=['         ', # observe X's first move
                           '    X    ']) # observe O's response 
full_training()
print(f'The agent has encountered {round(100*len(q_table)/4520, 2)}% of all possible board states')
versus_agent()
