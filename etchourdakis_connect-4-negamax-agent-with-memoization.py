# Kaggle specific imports

from kaggle_environments import evaluate, make, utils



# Numpy

import numpy as np



# Here the memoization table will be a defaultdict

from collections import defaultdict



# We introduce some randomness to not have completely deterministic play every time

import random



# Finally, tqdm is such a nice package to not import

import tqdm.notebook
def negamax(

        state, 

        depth, 

        alpha, 

        beta, 

        color, 

        ttlut

    ):

    """

        A negamax function with alpha-beta pruning and memoization.

        

        :param tuple state: A tuple where state[0] is the game board

                            at that state and state[1] the last column

                            played.

        :param int depth:   The depth examined. While visually we say

                            the tree is traversed form lower to higher

                            depth, here it is the opposite, so the leaves

                            will be at depth 0.

        :param float alpha: Parameter alpha for alpha beta pruning.

        :param float beta:  Parameter beta for alpha beta pruning.

        :param int color:   The player color (e.g. red or blue). Here its 

                            always 1 for the player, and -1 for the opponent.

    """

    

    # Store original alpha value, will be used later.

    alphaOrig = alpha

    

    # Tranform the board to a hashable tuple to be able to be

    # looked up in ttlut.

    node = tuple(state[0].flatten().tolist())

    

    # If we have already a record about node at a higher depth

    # retrieve its value. The node might be storing the exact

    # value, or an upper or lower bound.

    if ttlut[node]['valid'] and ttlut[node]['depth'] >= depth:

        if ttlut[node]['flag'] == 'EXACT':

            return ttlut[node]['value']

        elif ttlut[node]['flag'] == 'LOWERBOUND':

            alpha = max(alpha, ttlut[node]['value'])

        elif ttlut[node]['flag'] == 'UPPERBOUND':

            beta = min(alpha, ttlut[node]['value'])

            

        if alpha >= beta:

            return ttlut[node]['value']



    # Check whether the node is terminal and if so return a heuristic

    # value of it.

    if  is_terminal(state, ttlut):

        val =  color * eval_function(state, ttlut)

        return val

    

    # Check if we are at the maximum (here minimum) depth we can look ahead

    # and if so, return a heuristic value of it.

    if depth <= 0:

        return color * eval_function(state, ttlut)

    

    # Set initial value as -infinity

    value = -np.inf

    

    # Get the children of the current state. We do not really need to pass color here

    # however it speeds up computation (we do not need to count pieces in the board)

    # to see whether the number of piuieces is odd or even.

    children = get_children(state)

    

    # Some values are going to be equal, add some randomization so that when sorting 

    # those will not always be sorted the same way.

    random.shuffle(children)    

    

    # Sort the children according to increasing value

    children = sorted(children, key=lambda x: eval_function(x, ttlut))

    for child in children:

        

        # The rational opponent's play would maximize their negamax value so we should

        # always assume they pick the move which maximizes it so we should choose the 

        # same value when calculating our overall value. Since the opponent's value is going

        # to be of opposite sign, multiply it with -1.

        value = max(value, -negamax(child, depth-1, -beta, -alpha, -color, ttlut))

        alpha = max(alpha, value)

        

        # If it's over an upper bound, break the loop and do not examine any more states.

        if alpha >= beta:

            break

            

            

    # Since we already did the labour of getting the values for this state, save them

    # to a look-up table for future use.

    ttlut[node]['value'] = value

    if value <= alphaOrig:

        ttlut[node]['flag'] = 'UPPERBOUND'

    elif value >= beta:

        ttlut[node]['flag'] = 'LOWERBOUND'

    else:

        ttlut[node]['flag'] = 'EXACT'

        

    ttlut[node]['depth'] = depth

    ttlut[node]['valid'] = True

            

    return value
def eval_function(state, ttlut, X=4):

    """

    The evaluation function for state `state`

    

    :param tuple state: A state tuple where state[0] is the

                        game board at that state as a numpy array

                        and state[1] is the last action at that 

                        state.

    :param defaultdict ttlut: 

                        A look up table.

    """

    

    # Again, convert the state to a hashable form.

    node = tuple(state[0].flatten().tolist())

    

    # If the heuristic value returned by h does not exist,

    # add it to the table, else return it.

    if ttlut[node]['h'] is not None:

        val =  ttlut[node]['h']

        return val

    else:

        val = h(state, X=4)

        ttlut[node]['h'] = val

        return val
def is_terminal(state, ttlut, X=4):

    

    # Again, convert to hashable.

    node = tuple(state[0].flatten().tolist())

    

    # If we already know it is a terminal state, skip the

    # calculations.

    if ttlut[node]['terminal'] == True:

        return True



    board, _, _ = state

    

    # Check whether the number of nonzeros (occupied places)

    # in the top row is less than the size of that row. Since 

    # the colors are either -1 or 1 the following should suffice:

    if np.abs(board[0,:]).sum() ==  board.shape[1]:

        ttlut[node]['terminal'] = True

        return True

    

    # Return if the value returned by the evaluation function is 

    # + or - infinity.

    return np.isinf(np.abs(eval_function(state, ttlut, X=X)))

def add_piece_column(board, color, c):

    """ 

        Adds piece of color `color` on column `c`. Returns the row of that piece.



        :param np.ndarray board: the game board at that state

        :param int color: 1 or -1, the piece color

    """

    if c >= 0:    

        column = board[:, c]

        for cc in range(len(column)-1,-1,-1):

            if column[cc] == 0:

                column[cc] = color

                return cc



    return -1





def get_children(state, first_move=False):

    """

        Returns a list of children-states.

        

        :param tuple state: the state tuple

        :param int color: the current player, 1 or -1.

    """



    board, j, i = state

    color = board[i][j]

        

    # If the board is empty, the agent plays the first move, therefore:

    if first_move:

        color = 1

    else:

        color = -color

    

    children = []

    

    for c in range(board.shape[1]):

        # We can insert a new chesspiece in a non-full column. Since the bottom positions

        # of a column get filled first, we can check whether a column is empty by just checking the top

        # row (0).

        if board[0,c] == 0:

            

            # Copy the current board. 

            child_board = board.copy()

            #import pdb; pdb.set_trace()

            

            # Add a mark in column c.

            row = add_piece_column(child_board, color, c)

            

            # Create a new state for the child and add

            # it to the list of children.

            child_state = (child_board, c, row)

            children.append(child_state)



    

    return children
def h(state, X=4):

    """

        A heuristic function for Connect-X.

        

        :param tuple state: the (board, action) tuple state.

        :param int X: The number of chess pieces we expect a group of to

                      win. In Connect-4 this is X=4.

    """

    

    board, piece_j, piece_i = state

    

    # For a full group (therefore a win or loss), assign infinite value.

    GROUPX = np.inf 



    # Start with a value of 0

    hval = 0



    # Select heuristic value based on Player A's POV. In case the last action

    # was the player's opponent (color=-1) we need to invert the result since

    # the state is now damaging for the player.

    color = board[piece_i][piece_j]



        

    # Add to the value according to which column the piece was put into. Emphasize middle

    # columns more.

    if piece_j == 3:

        hval = 200

    elif piece_j in [2,4]:

        hval = 120

    elif piece_j in [1,5]:

        hval = 70

    else:

        hval = 40

    

    # Get the number of rows and columns of the board.

    nCols = board.shape[1]

    nRows = board.shape[0]

    

    # Below we count whether the action led to full groups, full-1 groups, etc.

    # We count whether the groups were formed in horizontal lines, vertical lines,

    # diagonally, or counter diagonally.

    

    # For horizontal lines. Count initial piece.

    count = 1

    

    # This checks whether the two edges are open

    openPoints = 0

    

    # Count right

    if piece_j < nCols - 1:

        for j in range(piece_j+1, min(piece_j+X, nCols)):

            

            # If already formed a full group, no need to

            # calculate further.

            if count == X:

                return color*GROUPX



            # If the pieces are all the same

            # color, keep counting, else break.

            if board[piece_i][j] == color:

                count += 1

            else:

                if board[piece_i][j] == 0:

                    openPoints += 1

                break

                

    # Do we have a full group already?

    if count == X:

        return color*GROUPX      

    

    # Count left

    if piece_j > 0:

        for j in range(piece_j-1, max(0, piece_j-X) -1, -1):

            if count == X:

                #import pdb; pdb.set_trace()

                return color*GROUPX



            if board[piece_i][j] == color:

                count += 1

            else:

                if board[piece_i][j] == 0:

                    openPoints += 1                

                break

                

    if count == X:

        return color*GROUPX     

    

    # IF we do not have a full group but we have a X-1 group, add

    # to the value.

    if count == X-1:

        if openPoints == 2:

            hval += 900000

        elif openPoints == 1:

            hval += 50000



    if count == X-2:

        if openPoints == 2:

            hval += 4000

        elif openPoints == 1:

            hval += 3000

    

    # Count vertically same as above but on the vertical dimension.

    # Note that here we do not need to count upwards since there is no

    # way to place a piece "below" the rest.

    count = 1

    openPoints = 0

        

    if piece_i < nRows - 1:

        for i in range(piece_i+1, min(piece_i+X, nRows)):

            if count == X:

                return color*GROUPX



            if board[i][piece_j] == color:

                count += 1

            else:

                if board[i][piece_j] == 0:

                    openPoints += 1

                break

                

    if count == X:

        return color*GROUPX        

    

    if count == X-1:

        # This case is always open.

        hval +=50000

    

    if count == X-2:

        hval +=3000    

    

    # Count on the bottom-left to up-right diagonals

    

    count = 1        

    openPoints = 0

    

    # Count up-right diagonal

    if piece_i > 0 and piece_j < nCols - 1:

        for d in range(1, X):

            if count == X:

                return color*GROUPX



            # Check if we hit the edges, break if so.

            if piece_i - d < 0:

                break

            if piece_j + d > nCols - 1:

                break



            if board[piece_i-d, piece_j+d] == color:

                count += 1

            else:

                if board[piece_i-d, piece_j+d] == 0:

                    openPoints += 1

                break

                

    if count == X:

        return color*GROUPX                

    

    # Count bottom-left diagonal

    if piece_i < nRows - 1 and piece_j > 0:

        for d in range(1,X):

            if count == X:

                return color*GROUPX



            if piece_i + d > nRows - 1:

                break

            if piece_j - d <0:

                break



            if board[piece_i+d, piece_j-d] == color:

                count += 1

            else:

                if board[piece_i+d, piece_j-d] == 0:

                    openPoints += 1

                break   

                

    if count == X:

        return color*GROUPX   

    

    if count == X-1:

        if openPoints == 2:

            hval +=800000

        elif openPoints == 1:

            hval +=40000

            

    if count == X-2:

        if openPoints == 2:

            hval +=4000

        elif openPoints == 1:

            hval +=3000            

    

    # Count on the up-left, bottom-right diagonal

    count = 1   

    openPoints = 0

    

    # Count up-left diagonal

    if piece_i > 0  and piece_j > 0:

        for d in range(1,X):

            if count == X:

                return color*GROUPX



            if piece_i - d < 0:

                break

            if piece_j - d < 0:

                break



            if board[piece_i-d, piece_j-d] == color:

                count += 1

            else:

                if board[piece_i-d, piece_j-d] == 0:

                    openPoints += 1

                break

                

    if count == X:

        return color*GROUPX                

                

    # Count bottom-right diagonal

    if piece_i < nRows - 1 and piece_j < nCols - 1:

        for d in range(1,X):

            if count == X:

                return color*GROUPX



            if piece_i + d > nRows - 1:

                break

            if piece_j + d > nCols - 1:

                break



            if board[piece_i+d, piece_j+d] == color:

                count += 1

            else:

                if board[piece_i+d, piece_j+d] == 0:

                    openPoints += 1

                break    

    

    if count == X:

        return color*GROUPX    

    

    if count == X-1:

        if openPoints == 2:

            hval +=800000

        elif openPoints == 1:

            hval +=40000

            

            

    if count == X-2:

        if openPoints == 2:

            hval +=4000

        elif openPoints == 1:

            hval +=3000      

            

    # Return the calculated hval multiplied with the color

    # to transform it for player A's point of view.

    return  color*hval
# This agent random chooses a non-empty column.



def nega_max_decision(state, depth, ttlut):

    # Get possible actions as children states

    children = get_children(state,  first_move=True)    

    

    

    # Shuffle them so that when picking the maximum value we do not just pick the same action over and over

    # when some actions are equivalent (e.g. the symmetic of a column with the column are expected to have

    # the same value)

    random.shuffle(children)

    

    # Pick an initial decision just in case we cannot find a better one.

    child_board, decision, decision_row = children[0]    

    

    # This step is similar to what's in negamax with the addition we return the best decision.

    value = -np.inf

    for child in children:

        new_value = -negamax(child, depth-1, -np.inf, np.inf, -1, ttlut)

        if new_value > value:

            decision = child[1]

            value = new_value         

    

    return decision



def nega_max_agent(observation, configuration):    

    col = 0



    # Get player mark integer and board from observation. Also transform board to numpy array.

    player = observation.mark

    board = np.array(observation.board).reshape(configuration.rows, configuration.columns)

    

    # If we play second, the last action is the column of the only non zero element in the board.

    if np.abs(board).sum() == 1:

        

        nz = np.nonzero(board)

        last_action = nz[0][0] - 1

        last_action_row = nz[1][0] - 1

    else:

        last_action = -1

        last_action_row = -1

    

    # Convert the colours from the observed colour marks (usually 1 and 2) to 1 for player and -1 for opponent.

    board = (board != player)*(board != 0)*-1 + (board == player)*1    

    

    # Initialize the look up table.

    ttlut = defaultdict(lambda:{'terminal':False, 'depth':-1000, 'flag': '', 'value': -10000, 'valid':False, 'h':None})



    # Set up the root node. 

    state = (board, last_action, last_action_row)

    

    # Initially there is no need to have the depth too high, 

    # increase it slowly.

    

    # Figure out what step we are in.

    step = np.abs(board).sum() + 1

    

    # Set the maximum depth. It has to be chosen as such it does not return a timeout. 

    max_depth = 5

    

    decision = nega_max_decision(state, max_depth, ttlut)



            

    # Finally, return the decision that maximizes the -negamax value shown above, or the default from the first child.

    return decision
env = make("connectx", debug=True)

env.reset()

env.run(["random", nega_max_agent])

env.render(mode="ipython", width=500, height=450)
# The state that caused the error is the second to last since the last one will just

# have the message 'ERROR'

observation = env.state[-2].observation 



# Or you can choose whichever else you need

observation = env.state[0].observation 



configuration = env.configuration



# Run and debug your agent for one step. To step further you can use the 

# python debugger with adding

# import pdb; pdb.set_trace() 

# to the error-causing line (it might be in another cell) and run it again!

nega_max_agent(observation, configuration)
%load_ext line_profiler

%lprun -f get_children nega_max_agent(observation, configuration)
def mean_reward(rewards):

    sum_A = 0

    nan_A = 0

    sum_B = 0

    nan_B = 0    

    for n, r in enumerate(rewards):

        if r[0] is None:

            nan_A += 1

        else:

            sum_A += r[0]

            

        if r[1] is None:

            nan_B += 1

        else:

            sum_B += r[1]

            

    avg_A = sum_A/float(len(rewards))

    avg_B = sum_B/float(len(rewards))

    

    outp = "Player A Total: {:.2f}, Avg: {:.2f}, Errors: {}\n".format(sum_A, avg_A, nan_A)

    outp += "Player B Total: {:.2f}, Avg: {:.2f}, Errors: {}\n".format(sum_B, avg_B, nan_B)

    

    return outp



def print_mean_rewards(agent, against, num_episodes=10):

    for name, opponent in tqdm.notebook.tqdm(against):

        print(

            "{} vs {}:\n{}".format(

                agent[0],

                name,

                mean_reward(

                    evaluate("connectx", [agent[1], opponent], num_episodes=num_episodes)

                ),

            )

        )

        print(

            "{} vs {}:\n{}".format(

                name,

                agent[0],

                mean_reward(

                    evaluate("connectx", [ opponent, agent[1]], num_episodes=num_episodes)

                ),

            )

        )    

print_mean_rewards(("KneeDeepInTheDoot", nega_max_agent), [("random", "random"), ("negamax","negamax")])

        
def nega_max_agent(observation, configuration):    

    

    # Numpy

    import numpy as np



    # Here the memoization table will be a defaultdict

    from collections import defaultdict



    # We introduce some randomness to not have completely deterministic play every time

    import random    

    

    def negamax(

            state, 

            depth, 

            alpha, 

            beta, 

            color, 

            ttlut

        ):

        """

            A negamax function with alpha-beta pruning and memoization.



            :param tuple state: A tuple where state[0] is the game board

                                at that state and state[1] the last column

                                played.

            :param int depth:   The depth examined. While visually we say

                                the tree is traversed form lower to higher

                                depth, here it is the opposite, so the leaves

                                will be at depth 0.

            :param float alpha: Parameter alpha for alpha beta pruning.

            :param float beta:  Parameter beta for alpha beta pruning.

            :param int color:   The player color (e.g. red or blue). Here its 

                                always 1 for the player, and -1 for the opponent.

        """



        # Store original alpha value, will be used later.

        alphaOrig = alpha



        # Tranform the board to a hashable tuple to be able to be

        # looked up in ttlut.

        node = tuple(state[0].flatten().tolist())



        # If we have already a record about node at a higher depth

        # retrieve its value. The node might be storing the exact

        # value, or an upper or lower bound.

        if ttlut[node]['valid'] and ttlut[node]['depth'] >= depth:

            if ttlut[node]['flag'] == 'EXACT':

                return ttlut[node]['value']

            elif ttlut[node]['flag'] == 'LOWERBOUND':

                alpha = max(alpha, ttlut[node]['value'])

            elif ttlut[node]['flag'] == 'UPPERBOUND':

                beta = min(alpha, ttlut[node]['value'])



            if alpha >= beta:

                return ttlut[node]['value']



        # Check whether the node is terminal and if so return a heuristic

        # value of it.

        if  is_terminal(state, ttlut):

            val =  color * eval_function(state, ttlut)

            return val



        # Check if we are at the maximum (here minimum) depth we can look ahead

        # and if so, return a heuristic value of it.

        if depth <= 0:

            return color * eval_function(state, ttlut)



        # Set initial value as -infinity

        value = -np.inf



        # Get the children of the current state. We do not really need to pass color here

        # however it speeds up computation (we do not need to count pieces in the board)

        # to see whether the number of piuieces is odd or even.

        children = get_children(state)



        # Some values are going to be equal, add some randomization so that when sorting 

        # those will not always be sorted the same way.

        random.shuffle(children)    



        # Sort the children according to increasing value

        children = sorted(children, key=lambda x: eval_function(x, ttlut))

        for child in children:



            # The rational opponent's play would maximize their negamax value so we should

            # always assume they pick the move which maximizes it so we should choose the 

            # same value when calculating our overall value. Since the opponent's value is going

            # to be of opposite sign, multiply it with -1.

            value = max(value, -negamax(child, depth-1, -beta, -alpha, -color, ttlut))

            alpha = max(alpha, value)



            # If it's over an upper bound, break the loop and do not examine any more states.

            if alpha >= beta:

                break





        # Since we already did the labour of getting the values for this state, save them

        # to a look-up table for future use.

        ttlut[node]['value'] = value

        if value <= alphaOrig:

            ttlut[node]['flag'] = 'UPPERBOUND'

        elif value >= beta:

            ttlut[node]['flag'] = 'LOWERBOUND'

        else:

            ttlut[node]['flag'] = 'EXACT'



        ttlut[node]['depth'] = depth

        ttlut[node]['valid'] = True



        return value    

    

    def nega_max_decision(state, depth, ttlut):

        # Get possible actions as children states

        children = get_children(state,  first_move=True)    





        # Shuffle them so that when picking the maximum value we do not just pick the same action over and over

        # when some actions are equivalent (e.g. the symmetic of a column with the column are expected to have

        # the same value)

        random.shuffle(children)



        # Pick an initial decision just in case we cannot find a better one.

        child_board, decision, decision_row = children[0]    



        # This step is similar to what's in negamax with the addition we return the best decision.

        value = -np.inf

        for child in children:

            new_value = -negamax(child, depth-1, -np.inf, np.inf, -1, ttlut)

            if new_value > value:

                decision = child[1]

                value = new_value         



        return decision    

    

    def eval_function(state, ttlut, X=4):

        """

        The evaluation function for state `state`



        :param tuple state: A state tuple where state[0] is the

                            game board at that state as a numpy array

                            and state[1] is the last action at that 

                            state.

        :param defaultdict ttlut: 

                            A look up table.

        """



        # Again, convert the state to a hashable form.

        node = tuple(state[0].flatten().tolist())



        # If the heuristic value returned by h does not exist,

        # add it to the table, else return it.

        if ttlut[node]['h'] is not None:

            val =  ttlut[node]['h']

            return val

        else:

            val = h(state, X=4)

            ttlut[node]['h'] = val

            return val    

    

    def is_terminal(state, ttlut, X=4):



        # Again, convert to hashable.

        node = tuple(state[0].flatten().tolist())



        # If we already know it is a terminal state, skip the

        # calculations.

        if ttlut[node]['terminal'] == True:

            return True



        board, _, _ = state



        # Check whether the number of nonzeros (occupied places)

        # in the top row is less than the size of that row. Since 

        # the colors are either -1 or 1 the following should suffice:

        if np.abs(board[0,:]).sum() ==  board.shape[1]:

            ttlut[node]['terminal'] = True

            return True



        # Return if the value returned by the evaluation function is 

        # + or - infinity.

        return np.isinf(np.abs(eval_function(state, ttlut, X=X)))

    

    

    def add_piece_column(board, color, c):

        """ 

            Adds piece of color `color` on column `c`. Returns the row of that piece.



            :param np.ndarray board: the game board at that state

            :param int color: 1 or -1, the piece color

        """

        if c >= 0:    

            column = board[:, c]

            for cc in range(len(column)-1,-1,-1):

                if column[cc] == 0:

                    column[cc] = color

                    return cc



        return -1





    def get_children(state, first_move=False):

        """

            Returns a list of children-states.



            :param tuple state: the state tuple

            :param int color: the current player, 1 or -1.

        """



        board, j, i = state

        color = board[i][j]



        # If the board is empty, the agent plays the first move, therefore:

        if first_move:

            color = 1

        else:

            color = -color



        children = []



        for c in range(board.shape[1]):

            # We can insert a new chesspiece in a non-full column. Since the bottom positions

            # of a column get filled first, we can check whether a column is empty by just checking the top

            # row (0).

            if board[0,c] == 0:



                # Copy the current board. 

                child_board = board.copy()

                #import pdb; pdb.set_trace()



                # Add a mark in column c.

                row = add_piece_column(child_board, color, c)



                # Create a new state for the child and add

                # it to the list of children.

                child_state = (child_board, c, row)

                children.append(child_state)





        return children    

    

    def h(state, X=4):

        """

            A heuristic function for Connect-X.



            :param tuple state: the (board, action) tuple state.

            :param int X: The number of chess pieces we expect a group of to

                          win. In Connect-4 this is X=4.

        """



        board, piece_j, piece_i = state



        # For a full group (therefore a win or loss), assign infinite value.

        GROUPX = np.inf 



        # Start with a value of 0

        hval = 0



        # Select heuristic value based on Player A's POV. In case the last action

        # was the player's opponent (color=-1) we need to invert the result since

        # the state is now damaging for the player.

        color = board[piece_i][piece_j]





        # Add to the value according to which column the piece was put into. Emphasize middle

        # columns more.

        if piece_j == 3:

            hval = 200

        elif piece_j in [2,4]:

            hval = 120

        elif piece_j in [1,5]:

            hval = 70

        else:

            hval = 40



        # Get the number of rows and columns of the board.

        nCols = board.shape[1]

        nRows = board.shape[0]



        # Below we count whether the action led to full groups, full-1 groups, etc.

        # We count whether the groups were formed in horizontal lines, vertical lines,

        # diagonally, or counter diagonally.



        # For horizontal lines. Count initial piece.

        count = 1



        # This checks whether the two edges are open

        openPoints = 0



        # Count right

        if piece_j < nCols - 1:

            for j in range(piece_j+1, min(piece_j+X, nCols)):



                # If already formed a full group, no need to

                # calculate further.

                if count == X:

                    return color*GROUPX



                # If the pieces are all the same

                # color, keep counting, else break.

                if board[piece_i][j] == color:

                    count += 1

                else:

                    if board[piece_i][j] == 0:

                        openPoints += 1

                    break



        # Do we have a full group already?

        if count == X:

            return color*GROUPX      



        # Count left

        if piece_j > 0:

            for j in range(piece_j-1, max(0, piece_j-X) -1, -1):

                if count == X:

                    #import pdb; pdb.set_trace()

                    return color*GROUPX



                if board[piece_i][j] == color:

                    count += 1

                else:

                    if board[piece_i][j] == 0:

                        openPoints += 1                

                    break



        if count == X:

            return color*GROUPX     



        # IF we do not have a full group but we have a X-1 group, add

        # to the value.

        if count == X-1:

            if openPoints == 2:

                hval += 900000

            elif openPoints == 1:

                hval += 50000



        if count == X-2:

            if openPoints == 2:

                hval += 4000

            elif openPoints == 1:

                hval += 3000



        # Count vertically same as above but on the vertical dimension.

        # Note that here we do not need to count upwards since there is no

        # way to place a piece "below" the rest.

        count = 1

        openPoints = 0



        if piece_i < nRows - 1:

            for i in range(piece_i+1, min(piece_i+X, nRows)):

                if count == X:

                    return color*GROUPX



                if board[i][piece_j] == color:

                    count += 1

                else:

                    if board[i][piece_j] == 0:

                        openPoints += 1

                    break



        if count == X:

            return color*GROUPX        



        if count == X-1:

            # This case is always open.

            hval +=50000



        if count == X-2:

            hval +=3000    



        # Count on the bottom-left to up-right diagonals



        count = 1        

        openPoints = 0



        # Count up-right diagonal

        if piece_i > 0 and piece_j < nCols - 1:

            for d in range(1, X):

                if count == X:

                    return color*GROUPX



                # Check if we hit the edges, break if so.

                if piece_i - d < 0:

                    break

                if piece_j + d > nCols - 1:

                    break



                if board[piece_i-d, piece_j+d] == color:

                    count += 1

                else:

                    if board[piece_i-d, piece_j+d] == 0:

                        openPoints += 1

                    break



        if count == X:

            return color*GROUPX                



        # Count bottom-left diagonal

        if piece_i < nRows - 1 and piece_j > 0:

            for d in range(1,X):

                if count == X:

                    return color*GROUPX



                if piece_i + d > nRows - 1:

                    break

                if piece_j - d <0:

                    break



                if board[piece_i+d, piece_j-d] == color:

                    count += 1

                else:

                    if board[piece_i+d, piece_j-d] == 0:

                        openPoints += 1

                    break   



        if count == X:

            return color*GROUPX   



        if count == X-1:

            if openPoints == 2:

                hval +=800000

            elif openPoints == 1:

                hval +=40000



        if count == X-2:

            if openPoints == 2:

                hval +=4000

            elif openPoints == 1:

                hval +=3000            



        # Count on the up-left, bottom-right diagonal

        count = 1   

        openPoints = 0



        # Count up-left diagonal

        if piece_i > 0  and piece_j > 0:

            for d in range(1,X):

                if count == X:

                    return color*GROUPX



                if piece_i - d < 0:

                    break

                if piece_j - d < 0:

                    break



                if board[piece_i-d, piece_j-d] == color:

                    count += 1

                else:

                    if board[piece_i-d, piece_j-d] == 0:

                        openPoints += 1

                    break



        if count == X:

            return color*GROUPX                



        # Count bottom-right diagonal

        if piece_i < nRows - 1 and piece_j < nCols - 1:

            for d in range(1,X):

                if count == X:

                    return color*GROUPX



                if piece_i + d > nRows - 1:

                    break

                if piece_j + d > nCols - 1:

                    break



                if board[piece_i+d, piece_j+d] == color:

                    count += 1

                else:

                    if board[piece_i+d, piece_j+d] == 0:

                        openPoints += 1

                    break    



        if count == X:

            return color*GROUPX    



        if count == X-1:

            if openPoints == 2:

                hval +=800000

            elif openPoints == 1:

                hval +=40000





        if count == X-2:

            if openPoints == 2:

                hval +=4000

            elif openPoints == 1:

                hval +=3000      



        # Return the calculated hval multiplied with the color

        # to transform it for player A's point of view.

        return  color*hval    

    

    col = 0



    # Get player mark integer and board from observation. Also transform board to numpy array.

    player = observation.mark

    board = np.array(observation.board).reshape(configuration.rows, configuration.columns)

    

    # If we play second, the last action is the column of the only non zero element in the board.

    if np.abs(board).sum() == 1:

        

        nz = np.nonzero(board)

        last_action = nz[0][0] - 1

        last_action_row = nz[1][0] - 1

    else:

        last_action = -1

        last_action_row = -1

    

    # Convert the colours from the observed colour marks (usually 1 and 2) to 1 for player and -1 for opponent.

    board = (board != player)*(board != 0)*-1 + (board == player)*1    

    

    # Initialize the look up table.

    ttlut = defaultdict(lambda:{'terminal':False, 'depth':-1000, 'flag': '', 'value': -10000, 'valid':False, 'h':None})



    # Set up the root node. 

    state = (board, last_action, last_action_row)

    

    # Initially there is no need to have the depth too high, 

    # increase it slowly.

    

    # Figure out what step we are in.

    step = np.abs(board).sum() + 1

    

    # Set the maximum depth. It has to be chosen as such it does not return a timeout. 

    max_depth = 5

    

    decision = nega_max_decision(state, max_depth, ttlut)



            

    # Finally, return the decision that maximizes the -negamax value shown above, or the default from the first child.

    return decision
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(nega_max_agent, "submission.py")
# Note: Stdout replacement is a temporary workaround.

import sys

out = sys.stdout

submission = utils.read_file("/kaggle/working/submission.py")

agent = utils.get_last_callable(submission)

sys.stdout = out



env = make("connectx", debug=True)

env.run([agent, agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")