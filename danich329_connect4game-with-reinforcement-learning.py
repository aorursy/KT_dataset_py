# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

if 1==0:

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import gym

from kaggle_environments import make, evaluate



# Create the game environment

# Set debug=True to see the errors if your agent refuses to run

env = make("connectx", debug=True)



# List of available default agents

print(list(env.agents))
# Selects random valid column

def agent_random(obs, config):

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    return random.choice(valid_moves)



# Selects middle column

def agent_middle(obs, config):

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    if config.columns//2 in valid_moves:

        return config.columns//2

    else:

        return random.choice(valid_moves)



# Selects leftmost valid column

def agent_leftmost(obs, config):

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    return valid_moves[0]
def mcts_agent(obs, config):

    import math

    import time

    from kaggle_environments.envs.connectx import connectx

    import random

    #random.seed(100)

    

    global current_state

    

    init_time = time.time()

    

    # time per move, left some overhead

    T_max = config.timeout - 0.5



    #Global variables

    PENALTY = -100

    TIE = 50

    WIN = 100

    dampen_factor = 1

    

    # Check if board is finished

    def check_board_state(board, config, piece, selected_column):

        if connectx.is_win(

            board, selected_column, 

            piece, config

        ):

            return (True, WIN)



        elif not(any(spot == 0 for spot in board)):

            return (True, TIE)



        else:

            return (False, None)

    

    # Find the opponent's move in order to speed up 

    # move finding

    def find_action_taken_by_opponent(new_board, old_board, config):

        for i, piece in enumerate(new_board):

            if piece != old_board[i]:

                return i % config.columns

        return -1  # shouldn't get here



    # Calulcate opponent score

    def opponent_score(score):

        return score + PENALTY

    

    # Pick a random move

    def random_choice(board, config):

        return random.choice([

            c 

            for c in range(config.columns) 

            if board[c] == 0

        ])

    

    # Simulate play by editing a copy of the provided board

    def simulate_play(board, config, piece, selected_column):

        row = max([

            r 

            for r in range(config.rows) 

            if board[selected_column + (r * config.columns)] == 0

        ])



        board[selected_column + (row * config.columns)] = piece

    

    # Simulate a full game to evaluate child

    def run_full_simulation(board, config, piece):

        

        # Make sure to to keep track of the pieces

        original_piece = piece

        

        current_board = board.copy()

        is_finished = False

        current_piece = piece

        

        while not is_finished:

            # play the selected move,

            # check if finished and reverse player

            # if not finished

            selected_column = random_choice(current_board, config)

            simulate_play(current_board, config, current_piece, selected_column)

            is_finished, terminal_score = check_board_state(

                current_board, config,

                current_piece, selected_column

            )

            

            if not is_finished:

                current_piece = 3 - current_piece



        if current_piece == original_piece:

            return terminal_score

        

        # Opponent wins so this is not ideal

        return opponent_score(terminal_score)

    

    # The score that the algorithm uses to

    # evaluate node values

    def uct_score(

        node_score, node_times_visited, 

        parent_score, dampen_factor=1

    ):

        if node_times_visited == 0:

            return math.inf

        else:

            # W_i/n_i

            first_half = (node_score / node_times_visited)

            

            # cp * sqrt(ln(N_i)/n_i)

            second_half = (

                dampen_factor * 

                math.sqrt(

                    math.log(parent_score) /

                    node_times_visited

                )

            )

            

            return first_half + second_half

    

    # Tree search

    class NodeState:

        def __init__(

            self,

            board, config, piece,

            parent=None, is_terminal=False, 

            terminal_score=None, selected_column=None

        ): 



            self.available_moves = [

                c 

                for c in range(config.columns) 

                if board[c] == 0

            ]

            self.board = board.copy()

            self.children = []

            self.config = config

            self.expandable_moves = self.available_moves.copy()

            self.is_terminal = is_terminal

            self.parent = parent

            self.piece = piece

            self.score = 0

            self.selected_column = selected_column

            self.terminal_score = terminal_score

            self.times_visisted = 0

        

        # Back Prop score up nodes

        def backpropagate(self, simulated_score):

            self.score += simulated_score

            self.times_visisted += 1

            if self.parent is not None:

                # Calculate Opponent's score

                self.parent.backpropagate(

                    opponent_score(simulated_score)

                )

        

        # If child already exists return already seen

        # action. Save memory

        def choose_explored_path(self, action):

            for child in self.children:

                if child.selected_column == action:

                    return child

            return None



        

        # Choice child that maxes UCB1 SCORE

        def choose_strongest_child(self, dampen_factor):

            children_scores = [

                uct_score(

                    child.score,

                    child.times_visisted,

                    self.times_visisted,

                    dampen_factor

                ) 

                for child in self.children

            ]

            

            max_score = max(children_scores)

            best_child_index = children_scores.index(max_score)

            return self.children[best_child_index]

        

        # Expand child and simulate run

        def expand_child_then_simulate(self):

            # Select move from legal list of moves

            column_move = random.choice(self.expandable_moves)



            # Copy and Play selected move

            child_board_state = self.board.copy()

            

            simulate_play(

                child_board_state, self.config, 

                self.piece, column_move

            )

            

            is_terminal, terminal_score = check_board_state(

                child_board_state, config,

                self.piece, column_move

            )

            

            #Insert selected move into the search tree

            self.children.append(

                NodeState(

                    board=child_board_state,

                    piece=3-self.piece,

                    config=self.config,

                    parent=self,

                    is_terminal=is_terminal,

                    terminal_score=terminal_score,

                    selected_column=column_move

                )

            )

            

            # Simulate game until end

            simulated_score = self.children[-1].simulate()

            

            # Backpropogate score

            self.children[-1].backpropagate(simulated_score)

            

            # Remove explored list

            self.expandable_moves.remove(column_move)

        

        # Check if there are no children at respective node

        def is_expandable(self):

            return (

                (not self.is_terminal) 

                and 

                (len(self.expandable_moves) > 0)

            )

        

        # Simulate score

        def simulate(self):

            if self.is_terminal:

                return self.terminal_score

            

            return opponent_score(

                run_full_simulation(

                    self.board,  

                    self.config,

                    self.piece

                )

            )

        

        def tree_single_run(self):

            """

            A single iteration of the 4 stages of the MCTS algorithm.

            """

            if self.is_terminal:

                self.backpropagate(self.terminal_score)

                return

            if self.is_expandable():

                self.expand_child_then_simulate()

                return

            self.choose_strongest_child(dampen_factor).tree_single_run()

        

    # Main function

    # If current_state already exists, recycle it based on action taken by opponent

    try:  

        current_state = (

            current_state

            .choose_explored_path(

                find_action_taken_by_opponent(

                    board, current_state.board,

                    configuration

                )

            )

        )

        

        # make current_state the root node, 

        # dereference parents and siblings

        current_state.parent = None  

        

    except:  # new game or other error in recycling attempt due to Kaggle mechanism

        current_state = NodeState(

            obs.board, config, 

            obs.mark,

            parent=None, 

            is_terminal=False, 

            terminal_score=None, 

            selected_column=None

        )

   

    # Run MCTS iterations until time limit is reached.

    while time.time() - init_time <= T_max:

        current_state.tree_single_run()

    

    child_scores = [

        child.score 

        for child in current_state.children

    ]



    max_score = max(child_scores)

    best_child = child_scores.index(max_score)

    current_state = current_state.children[best_child]

    

    return current_state.selected_column
env.run(['random', mcts_agent])

env.render(mode="ipython")
# check if column contains key

def three_in_col(col, key):

    found=False

    for i in range(0,len(col)-len(key)+1):

        if str(key) == str(col[i:len(key)+i]):

            return(True)

    return(False)
# check if row contains a winning spot

def three_in_row(row, my_mark, numtowin):

    win_spot_col=[]

    for i in range(0,len(row)-numtowin+1):

        # subset to num to win

        sub_row = row[i:numtowin+i]

        # check if there are 3's of my_mark

        num_mymark = np.count_nonzero(sub_row == my_mark)

        if num_mymark == numtowin - 1:

            if 0 in row[i:numtowin+i]:

                wi_rep_idx = np.where(sub_row==0)[0][0]

                win_spot_col.append(

                    i + wi_rep_idx)

    return(win_spot_col)
# This returns either:

# * The columns that will allow you to win

# * The column to block the opponent from winning

# * -1 meaning that no one is winning in the next round

def essential_move(mark, board, valid_moves, config):

    # parameters:

    # * threekey: combination to win. For example if we 

    #   want to connect 4 the combination to win is [1,1,1,0]

    #   (given that we are player 1)

    # * board: the matrix containing the board (the bottom row

    #   should be index 0)

    # * valid moves: open columns on board

    # * config: configuration of the board

    

    threekey = [int(x) for x in list(np.ones(config.inarow-1) * mark) +[0]]

    

    # connect 4 in column

    for move in valid_moves:

        threeInCol = three_in_col(board[:,move],np.array(threekey))

        if threeInCol:

            return(move)

    winning_col=[]

    winning_row=[]

    

    for row_i in range(0,config.rows):

        row=board[row_i,:]

        

        # connect 4 in row

        win_cols_in_row_i=three_in_row(row, mark, config.inarow)

        if len(win_cols_in_row_i) > 0:

            winning_col+=win_cols_in_row_i

            winning_row+=([row_i]*len(win_cols_in_row_i))

        

        # connect 4 in a diagonal

        d_start = np.where(row==mark)

        for ds in list(d_start[0]):

            # right diagonal

            if (row_i + (config.inarow - 1) < config.rows

                and (ds + (config.inarow - 1) < config.columns)):

                good=0

                good_row=-1

                good_col=-1

                for k in range(1,config.inarow):

                    if board[row_i+k,ds+k]==mark:

                        good+=1

                    elif board[row_i+k,ds+k]==0:

                        good_row=row_i+k

                        good_col=ds+k

                if good==config.inarow - 1:

                    winning_col+=[good_col]

                    winning_row+=[good_row]

            # left diagonal

            if (row_i + (config.inarow - 1) < config.rows

                and (ds - (config.inarow - 1) > -1)):

                good=0

                good_row=-1

                good_col=-1

                for k in range(1,len(threekey)):

                    if board[row_i+k,ds-k]==mark:

                        good+=1

                    elif board[row_i+k,ds-k]==0:

                        good_row=row_i+k

                        good_col=ds-k

                if good==config.inarow - 1:

                    winning_col+=[good_col]

                    winning_row+=[good_row]

            

            

    # check if winning spot is available

    for i in range(0,len(winning_col)):

        wcol=winning_col[i]

        wrow=winning_row[i]

        if wrow - 1 < 0 or board[wrow-1,wcol]!=0:

            if wcol in valid_moves:

                return(int(wcol))

    return(-1)
import sys

# Selects random valid column

def agent_without_a_plan(obs, config):

    # `obs` contains 2 peices of information

    b_m=np.flip(

        np.resize(obs.board,(config.rows,config.columns)),0)

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    

    # get opponents mark

    opp_mark=1

    if obs.mark==1:

        opp_mark=2

        

    # always perform these first two moves if available

    #if b_m[2,3]==0 and (b_m[0,3]==0 or b_m[0,3]==obs.mark):

    #    return int(3)

    

    # LAST MOVE TO WIN

    # combo2win

    #combo2win = [int(x) for x in list(np.ones(config.inarow-1) * obs.mark)+[0]]

    winning_move = essential_move(obs.mark,b_m, 

                                  valid_moves, config)

    if winning_move != -1:

        return(winning_move)

    

    # BLOCK OPPONENT FROM WINNING

    # combo2lose

    #combo2lose = [int(x) for x in list(np.ones(config.inarow-1) * opp_mark)+[0]]

    block_winning_move = essential_move(opp_mark,b_m,

                                        valid_moves, config)

    if block_winning_move != -1:

        return(block_winning_move)

    

    return random.choice(valid_moves)
# returns row that mark will be placed at

# in the given column

def drop_chip_row(board, column, nrows):

    drop=False

    for row_i in range(0,nrows):

        if board[row_i,column]==0:

            return(row_i)

    return(-1)

            
# This function is returns the score of dropping 

# a chip in column `drop_col`

def score_drop(board, drop_col, my_mark,config):

    opp_mark=1

    if my_mark==1:

        opp_mark=2

    drop_row = drop_chip_row(board, drop_col, config.rows)

    # 1. get score of marks in a column

    ## automatic .25 is given since dropping a chip down is .25 points

    col_score=.25 

    ## given .25 points to each chip that is consecutively below

    ## the one you plan to drop

    if drop_row > 0:            

        next_row = drop_row - 1

        while next_row >= 0 and next_row > drop_row - config.inarow:

            if board[next_row,drop_col] == my_mark:

                col_score+=.25

            else:

                next_row = -1

            next_row = next_row - 1

    ## if the chip is near the top it gets 0 points

    ## if we cannot make a column of 4

    if (drop_row > config.rows - config.inarow):

        score_threshold = (config.inarow -

                          (config.rows - drop_row)) * .25

        if .25 + score_threshold > col_score:

            col_score = 0

    score = col_score

        

    # 2. get score of marks in a row

    if score != 1:

        row_score=0

        leftmost=drop_col-(config.inarow-1)

        if leftmost < 0:

            leftmost=0

        while leftmost <= config.columns - config.inarow:

            rightmost=leftmost+config.inarow



            row_of_four = board[drop_row,leftmost:rightmost]

            if not np.isin(opp_mark,row_of_four):

                new_row_score=.25 + (np.count_nonzero(row_of_four == my_mark)* .25)

                if new_row_score > row_score:

                    row_score = new_row_score

            leftmost+=1

        if row_score==.25:

            row_score=0

        if row_score > score:

            score = row_score

        

    # 3. get score of marks in a RIGHT diagonal

    if score != 1:

        right_d_score=0



        ## get the lowest possible diagonal from this point

        leftmost_col=drop_col-(config.inarow-1)

        lowest_row=drop_row-(config.inarow-1)

        ## check if the row or column is below zero

        ## if this is so, we need to adjust the bottom

        ## of the diagonal

        col_below_zero=0

        if leftmost_col < 0:

            col_below_zero=leftmost_col*-1

        row_below_zero=0

        if lowest_row < 0:

            row_below_zero=lowest_row*-1

        if col_below_zero > row_below_zero:

            leftmost_col = 0

            lowest_row = lowest_row + col_below_zero

        elif col_below_zero < row_below_zero:

            lowest_row = 0

            leftmost_col = leftmost_col + row_below_zero

        elif row_below_zero > 0:

            leftmost_col = 0

            lowest_row = 0



        diag_of_four = np.array([])

        end_of_diag_col=leftmost_col + (config.inarow - 1)

        end_of_diag_row=lowest_row + (config.inarow - 1)

        while (leftmost_col <= drop_col and

              end_of_diag_col < config.columns and

              end_of_diag_row < config.rows):

            ## get values of sub-diagonal in diag_of_four

            if len(diag_of_four)==0:

                for i in range(0,config.inarow):

                    diag_of_four=np.append(diag_of_four,board[lowest_row+i,leftmost_col+i])

            else:

                diag_of_four = np.delete(diag_of_four, 0)

                i = config.inarow - 1

                diag_of_four = np.append(diag_of_four,board[lowest_row+i,leftmost_col+i])



            ## get score given diag_of_four

            if not np.isin(opp_mark,diag_of_four):

                new_rdiag_score=.25 + (np.count_nonzero(diag_of_four == my_mark)* .25)

                if new_rdiag_score > right_d_score:

                    right_d_score = new_rdiag_score

            leftmost_col+=1

            lowest_row+=1

            end_of_diag_col=leftmost_col + (config.inarow - 1)

            end_of_diag_row=lowest_row + (config.inarow - 1)

        if right_d_score > score:

            score = right_d_score

            

    # 4. get score of marks in a LEFT diagonal

    if score != 1:

        left_d_score=0



        ## get the lowest possible diagonal from this point

        rightmost_col=drop_col+(config.inarow-1)

        lowest_row=drop_row-(config.inarow-1)

        #print("og rightmost col: " +str(rightmost_col))

        #print("og lowest_row: " +str(lowest_row))

        ## check if the column is greater than 

        ## the # of columns OR row is below zero

        ## if this is so, we need to adjust the bottom

        ## of the diagonal

        col_gt_numrows=0

        if rightmost_col >= config.columns:

            col_gt_numrows=rightmost_col-(config.rows-1)

        row_below_zero=0

        if lowest_row < 0:

            row_below_zero=lowest_row*-1

        if col_gt_numrows > row_below_zero:

            rightmost_col = config.rows-1

            lowest_row = lowest_row + col_gt_numrows

        elif col_gt_numrows < row_below_zero:

            lowest_row = 0

            rightmost_col = rightmost_col - row_below_zero

        elif row_below_zero > 0:

            rightmost_col = config.rows-1

            lowest_row = 0

        diag_of_four = np.array([])

        end_of_diag_col=rightmost_col - (config.inarow - 1)

        end_of_diag_row=lowest_row + (config.inarow - 1)

        while (rightmost_col >= drop_col and

              end_of_diag_col >= 0 and

              end_of_diag_row < config.rows):

            # get values of sub-diagonal in diag_of_four

            if len(diag_of_four)==0:

                for i in range(0,config.inarow):

                    diag_of_four=np.append(diag_of_four,board[lowest_row+i,rightmost_col-i])

            else:

                diag_of_four = np.delete(diag_of_four, 0)

                i = config.inarow - 1

                diag_of_four = np.append(diag_of_four,board[lowest_row+i,rightmost_col-i])



            # get score given diag_of_four

            if not np.isin(opp_mark,diag_of_four):

                new_ldiag_score=.25 + (np.count_nonzero(diag_of_four == my_mark)* .25)

                if new_ldiag_score > left_d_score:

                    left_d_score = new_ldiag_score

            rightmost_col=rightmost_col-1

            lowest_row+=1

            end_of_diag_col=rightmost_col - (config.inarow - 1)

            end_of_diag_row=lowest_row + (config.inarow - 1)

        if left_d_score > score:

            score = left_d_score

    if score == 1:

        return(4)

    else:

        return(col_score+row_score+left_d_score+right_d_score)
if 1==0:

    a_m=np.array([[0, 2, 1, 2, 1, 2, 0],

       [0, 1, 1, 1, 0, 0, 0],

       [0, 2, 1, 2, 0, 0, 0],

       [0, 0, 2, 1, 0, 0, 0],

       [0, 0, 1, 0, 0, 0, 0],

       [0, 0, 0, 0, 0, 0, 0]])

    print(a_m)



    import pandas as pd

    class a:

        def __init__(self,columns,rows,inarow):

            self.columns = columns

            self.rows = rows

            self.inarow = inarow



    conf=a(7,6,4)

    my_num=1

    opp_num=2

    for i in range(0,7):

        if i == 4:

            print("col: "+ str(i))

            print("my score")

            print(score_drop(a_m,i,my_num,conf))

            print("block score")

            score_drop(a_m,i,opp_num,conf)

            # add my mark in column to fake board

            fake_board_a=a_m.copy()

            fake_chip_ya=drop_chip_row(fake_board_a, i, conf.rows)

            print("------")

            print("give score")

            if fake_chip_ya == -1:

                    print(0)

            else:

                fake_board_a[fake_chip_ya,i]=my_num

                print(fake_board_a)

                print(score_drop(fake_board_a, i, opp_num,conf))

            print("---")
# Selects random valid column

def custom_agent(obs, config):

    # `obs` contains 2 peices of information

    b_m=np.flip(

        np.resize(obs.board,(config.rows,config.columns)),0)

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    

    # get opponents mark

    opp_mark=1

    if obs.mark==1:

        opp_mark=2



    # GET SCORE OF EACH POSSIBLITY

    score_moves=[]

    for move in valid_moves:

        add_score_mymark= score_drop(b_m, move, obs.mark,config)

        add_score_oppmark= score_drop(b_m, move, opp_mark,config) * .8

        fake_board=b_m.copy()

        # add my mark in column to fake board

        fake_chip_y=drop_chip_row(fake_board, move, config.rows)

        if fake_chip_y == -1:

            sub_score = 0

        else:

            fake_board[fake_chip_y,move]=obs.mark

            sub_score= score_drop(fake_board, move, opp_mark,config)

        score_moves.append(

            (add_score_mymark + 

             add_score_oppmark)

            - sub_score)

        

    best_col=[valid_moves[0]]

    best_score=score_moves[0]

    if len(score_moves) > 1:

        for i in range(1,len(score_moves)):

            if score_moves[i] > best_score:

                best_col=[valid_moves[i]]

                best_score=score_moves[i]

            elif score_moves[i] == best_score:

                best_col.append(valid_moves[i])

    return(random.choice(best_col))
# Agents play one game round

env.run([custom_agent, agent_without_a_plan])



# Show the game

env.render(mode="ipython")
# To learn more about the evaluate() function, check out the documentation here: (insert link here)

def get_win_percentages(agent1, agent2, n_rounds=100):

    # Use default Connect Four setup

    config = {'rows': 6, 'columns': 7, 'inarow': 4}

    # Agent 1 goes first (roughly) half the time          

    agent1_first_outcomes = evaluate("connectx", 

                        [agent1, agent2],

                        config, [], n_rounds//2)

    # Agent 2 goes first (roughly) half the time  

    agent2_first_outcomes = [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]

    outcomes = agent1_first_outcomes + agent2_first_outcomes

    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))

    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))

    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))

    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))

    print("Number of Draws (in {} game rounds):".format(n_rounds), outcomes.count([0, 0]))
get_win_percentages(agent1=agent_middle, agent2=agent_random)
get_win_percentages(agent1=agent_leftmost, agent2=agent_random)
get_win_percentages(agent1=agent_without_a_plan, agent2=agent_random)
get_win_percentages(agent1=custom_agent, agent2=agent_random)
get_win_percentages(agent1='', agent2=agent_without_a_plan)
get_win_percentages(agent1='random', agent2=mcts_agent)