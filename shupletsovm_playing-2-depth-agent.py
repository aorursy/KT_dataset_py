# 1. Enable Internet in the Kernel (Settings side pane)



# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 

# !curl -X PURGE https://pypi.org/simple/kaggle-environments



# ConnectX environment was defined in v0.1.6

!pip install 'kaggle-environments>=0.1.6'
from kaggle_environments import evaluate, make, utils

from copy import deepcopy

import numpy as np



env = make("connectx", debug=True)

env.render()
class agents:

    #returns cheker on boor coordinate. Top left coord is (0, 0)

    def get_checker_by_coord(x, y, observation, configuration):

        return observation.board[x + (configuration.columns * y)]



    #returns the most downward row in a columns

    def find_empty_row_in_column(x, observation, configuration):

        column = agents.get_column(x, observation, configuration)

        y = -1

        coord = 0

        for checker in column:

            if checker == 0: 

                y = coord

                coord += 1

        return y



    #returns a column by X coordinate. Left X = 0

    def get_column(x, observation, configuration):

        column = []

        for y in range(0 ,configuration.rows):

            checker = agents.get_checker_by_coord(x, y, observation, configuration)

            column.append(checker)

        return column



    #returns a row by Y coordinate. Top Y = 0

    def get_row(y, observation, configuration):

        row = []

        for x in range(0,configuration.columns):

            checker = agents.get_checker_by_coord(x, y, observation, configuration)

            row.append(checker)

        return row



    #returns a diagonal (left_top -> bottom_right), inputs coordinates should be in 'Top row' or 'Left collumn'

    def get_main_diagonal(x, y, observation, configuration):

        diag = []

        if x * y != 0:

            raise Exception('Incorrect coordinates: either x or y should be 0')

        while x <= configuration.columns -1 and y <= configuration.rows - 1:

            checker = agents.get_checker_by_coord(x, y, observation, configuration)

            diag.append(checker)

            x += 1

            y += 1

        return diag



    #returns a diagonal (left_bot -> top_right), inputs coordinates should be in 'Bottom row' or 'Left collumn'

    def get_secondary_diagonal(x, y, observation, configuration):

        diag = []

        if not (x == 0 or y == (configuration.rows - 1)):

            raise Exception('Incorrect coordinates: either x should be 0 or y should be equal to: ', configuration.rows - 1)

        while x <= configuration.columns -1 and y >= 0:

            checker = agents.get_checker_by_coord(x, y, observation, configuration)

            diag.append(checker)

            x += 1

            y += -1

        return diag



    #checks for a winner in an array. Returns 0 if none, 1 or 2 if there is a winner.

    def check_array_for_winner(array):

        winner = 0

        counter = 0

        prev_checker = 0

        for checker in array:

            if checker !=0 and checker == prev_checker:

                counter += 1

            else:

                counter = 0

            prev_checker = checker

            if counter == 3:

                return checker

        return winner



    #CHANGED

    #returns list of possible legal moves (column number)

    def list_of_possible_moves(observation, configuration):

        return [c for c in range(configuration.columns) if observation.board[c] == 0]



    #CHANGED

    #sorts list of possible moves in preffered order (reversed)

    def sorted_list_of_possible_moves(observation, configuration):

        #move order the agent should prioritize, given multiply equal options

        move_order = [6,0,5,1,4,2,3]

        legal_moves = agents.list_of_possible_moves(observation, configuration)

        sorted_legal_moves = []

        for move in move_order:

            if move in legal_moves:

                sorted_legal_moves.append(move)

        return sorted_legal_moves



    #makes a move given column number

    def simulate_move(x, observation, configuration):

        y = agents.find_empty_row_in_column(x, observation, configuration)

        if y == -1:

            raise Exception('Illegal move! Column: ', y, ' has no empty spots!')

        observation = agents.set_checker_by_coord(x, y, observation, configuration)

        return observation



    #sets checker to a value given board state and column

    def set_checker_by_coord(x, y, observation, configuration):

        current_mark = observation.mark

        observation.board[x + (configuration.columns * y)] = current_mark

        return observation



    #changes mark of the current player

    def change_current_player(observation, configuration):

        if observation.mark == 1:

            observation.mark = 2

        else:

            observation.mark = 1

        return observation



    #simulates all possible legal moves, returns a winning move if possible, non winning moves for opponent on depth n=1 or a list of legal moves if any move leads to a loss

    def simulate_legal_moves(observation, configuration):

        moves = agents.list_of_possible_moves(observation, configuration)

        observations = []

        winners = []

        for move in moves:

            temp_observation = deepcopy(observation)

            temp_observation = agents.simulate_move(move, temp_observation, configuration)

            temp_observation = agents.change_current_player(temp_observation, configuration)

            observations.append(temp_observation)

            winners.append(agents.check_winner(temp_observation, configuration))

        return moves, observations, winners



    #returns 0 if board has no winner, 1 or 2 if there is only one, raises an exeption if both 1 and 2 are winners

    def check_winner(observation, configuration):

        winner_list = [0]



        #checking all columns for a winner

        for x in range(0, configuration.columns):

            winner = agents.check_array_for_winner(agents.get_column(x, observation, configuration))

            winner_list.append(winner)



        #checking all rows for a winner

        for y in range(0, configuration.rows):

            winner = agents.check_array_for_winner(agents.get_row(y, observation, configuration))

            winner_list.append(winner)



        #checking main diagonals for a winner

        top_row = [[x, 0] for x in  range(1, configuration.columns-3)]

        left_column = [[0, y] for y in range(0, configuration.rows-3)]

        for coord in (top_row+left_column):

            x = coord[0]

            y = coord[1]

            winner = agents.check_array_for_winner(agents.get_main_diagonal(x, y, observation, configuration))

            winner_list.append(winner)



        #checking secondary diagonals for a winner

        bottom_row = [[x, configuration.rows - 1] for x in  range(1, configuration.columns-3)]

        left_column = [[0, y] for y in range(3, configuration.rows)]

        for coord in (bottom_row+left_column):

            x = coord[0]

            y = coord[1]

            winner = agents.check_array_for_winner(agents.get_secondary_diagonal(x, y, observation, configuration))

            winner_list.append(winner)



        #get a winner 

        winner_list = np.unique(np.array(winner_list))

        if len(winner_list) == 3:

            raise Exception('More than two winners were found on the board state!')

        if len(winner_list) == 1:

            return 0

        return np.amax(winner_list)



    #returns expected reward for a board state: 0 if no one wins in depth 1, 1 if current player wins, -1 if opponent wins

    def evaluate_board_on_our_move(observation, configuration):

        #check if we win next turn

        moves, observations, winners = agents.simulate_legal_moves(observation, configuration)

        if observation.mark in winners:

            return 1



        #check if opponents wins every turn after ours

        opponent_wins_depth_1 = -1

        for depth_1_observation in observations:

            moves_depth_1, observations_depth_1, winners_depth_1 = agents.simulate_legal_moves(depth_1_observation, configuration)

            if depth_1_observation.mark not in winners_depth_1:

                opponent_wins_depth_1 = 0

                break



        return opponent_wins_depth_1

    

########################################

#### Editing by Alexander Levin ########

########################################

    max_depth = 2  # declares inside the agent

    current_depth = 0 # declares inside the agent

    def choose_best_option(observation, configuration, current_depth, max_depth):

        possible_moves = agents.list_of_possible_moves(observation, configuration)

        def check_for_ones(observation, configuration):

            moves, observations, winners = agents.simulate_legal_moves(observation, configuration)

            owner_id = observation.mark

            if owner_id in winners:

                return True, moves[winners.index(owner_id)]

            else:

                return False, [moves, observations]



        ones_flag, ones_content =  check_for_ones(observation, configuration)

        if ones_flag == True:  # in up to 7 next boards, function:

            result_move = ones_content

            return result_move

        else:

            moves, observations = ones_content[0], ones_content[1]

            

        def get_non_minus_one_moves(moves, observations):

            zero_moves, zero_observations = [], []

            #### Check for lists working properly

            for move, observation in zip(moves, observations):

                next_moves, next_observations, next_winners = agents.simulate_legal_moves(observation, configuration)

                opponent_id = (-owner_id+3)

                while (opponent_id in next_winners):

                    next_moves.pop(next_winners.index(opponent_id))

                    next_observations.pop(next_winners.index(opponent_id))

                    next_winners.pop(next_winners.index(opponent_id))

                zero_moves.append(next_moves)

                zero_observations.append(next_observations)

                return zero_moves, zero_observations

            #### Finish this

#         zero_moves, zero_observations = get_non_minus_one_moves() # in up to 49 next boards, function, returns moves without loosing

        current_depth += 1

#         if zero_moves != []:

#             possible_moves = zeros_moves

#         else:

#             print('We are fucked')

#             random_move = random.choice(agents.list_of_possible_moves(observation, configuration))

#             return random_move

#         if depth < max_depth:

#             result_move = choose_best_option(observation, possible_moves, current_depth, max_depth) # going deeper

#         else:

#             result_move = zeros_moves

        return result_move

########################################

########################################

    

    #input is a column name, output is increased chance for moving into middle

    def increase_moves_weight_to_middle(column):

        weight = {0:0

                  ,1:0.05

                  ,2:0.10

                  ,3:0.15

                  ,4:0.10

                  ,5:0.05

                  ,6:0}

        return weight[column]

    

    

    # A little bit smarter agent

    def agent_id_4(observation, configuration):



        # Number of Columns on the Board.

        columns = configuration.columns

        # Number of Rows on the Board.

        rows = configuration.rows

        # Number of Checkers "in a row" needed to win.

        inarow = configuration.inarow

        # The current serialized Board (rows x columns).

        board = observation.board

        # Which player the agent is playing as (1 or 2).

        mark = observation.mark



        #CHANGED

        #Main

        moves, new_observations, winners = agents.simulate_legal_moves(observation, configuration)



        bad_moves = []

        for move, new_observation in zip(moves, new_observations):

            moves_depth2, new_observations_depth2, winners_depth2 = agents.simulate_legal_moves(new_observation, configuration) 

            for winner_depth2 in winners_depth2:

                if winner_depth2 != observation.mark and winner_depth2 != 0:

                    bad_moves.append(move)

                    break



        potential_next_move = agents.sorted_list_of_possible_moves(observation, configuration)[0]

        # Return which column to drop a checker (action).

        for x in agents.sorted_list_of_possible_moves(observation, configuration):

            if x not in bad_moves:

                potential_next_move = x



        return potential_next_move

    

    def agent_id_5(observation, configuration):

        from copy import deepcopy

        import numpy as np

        import random



        ################ Main ###################

        moves, new_observations, winners = agents.simulate_legal_moves(observation, configuration)



        #checks for a winning move and returns it if need be

        for move, winner in zip(moves, winners):

            if winner == observation.mark:

                return move





        moves_values = []

        for move, new_observation in zip(moves, new_observations):

            moves_depth2, observations_depth2, winners_depth2 = agents.simulate_legal_moves(new_observation, configuration)

            move_reward = []



            #iterate over depth 1 boards

            for move_depth2, observation_depth2, winner_depth2 in zip(moves_depth2, observations_depth2, winners_depth2):



                #opponent wins?

                if winner_depth2 != observation.mark and winner_depth2 != 0:

                    move_reward.append(-1)

                    break



                #if opponent does not win evaluate board

                board_state = agents.evaluate_board_on_our_move(observation_depth2, configuration)

                move_reward.append(board_state)





            try: 

                moves_values.append([move, np.min(move_reward), np.mean(move_reward)])

            except:

                moves_values = [[move, 0, 0.0]]



        #chose best score

        current_opponent_wins = -2

        current_move_score = -2

        for possible_move in moves_values:

            possible_move[2] = possible_move[2] + agents.increase_moves_weight_to_middle(possible_move[0])

#             possible_move[2] = possible_move[2] + random.uniform(0, 0.3)



            if possible_move[1] > current_opponent_wins:

                current_best_move = possible_move[0]

                current_opponent_wins = possible_move[1]

                current_move_score = possible_move[2]

            if possible_move[2]> current_move_score and possible_move[1] >= current_opponent_wins:

                current_best_move = possible_move[0]

                current_opponent_wins = possible_move[1]

                current_move_score = possible_move[2]

        

        return current_best_move
# "None" represents which agent you'll manually play as (first or second player).

env.play([agents.agent_id_5, None], width=500, height=450)