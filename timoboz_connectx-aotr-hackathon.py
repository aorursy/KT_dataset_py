!pip install 'kaggle-environments>=0.1.6'
from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)

env.render()
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
def dev_agent(observation, configuration):

    from random import choice

    def score_action(observation, configuration, action):

        new_board = look_next(observation, configuration, action)

        score = 0

        if observation.mark == 1:

            enemy = 2

        else:

            enemy = 1

        #count row with most-in a row

        max_row_score = 0

        running_score = 0

        for r in range(configuration.rows):

            for c in range(configuration.columns):

                if new_board[r][c] == observation.mark:

                    running_score += 1

                    if running_score > score:

                        max_row_score = running_score

                if new_board[r][c] == enemy:

                    running_score = 0

                if running_score > max_row_score:

                    max_row_score = running_score

                else:

                    running_score = 0

            running_score = 0

        max_column_score = 0

        for c in range(configuration.columns):

            for r in range(configuration.rows):

                if new_board[r][c] == observation.mark:

                    running_score += 1

                if new_board[r][c] == enemy:

                    running_score = 0

                if running_score > max_column_score:

                    max_column_score = running_score

                else:

                    running_score = 0

            running_score = 0

        return max(max_column_score, max_row_score)

    def look_next(observation, configuration, col):

        board = np.reshape(observation.board, (configuration.rows, configuration.columns))

        new = board.copy()

        for r in range(configuration.rows-1, -1, -1):

            if new[r][col] == 0:

                new[r][col] = observation.mark

                break

        return new

    playable_columns = [c for c in range(configuration.columns) if observation.board[c] == 0]

    scores_dict = {}

    high_score = 0

    best_move = 0

    for c in playable_columns:

        temp_score = score_action(observation, configuration, c)

        if temp_score > high_score:

            high_score = temp_score

            best_move = c

    return best_move

    #return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
def ilya_agent(observation, configuration):

    import numpy as np

    mark = observation.mark # who am I? 0 or 1

    columns = configuration.columns

    rows = configuration.rows

    inarow = configuration.inarow

    board = observation.board

    brd = np.array(board).reshape(rows, columns)

    if np.max(brd[rows - 1]) == 0:

        return columns // 2

    tmp = np.sum(brd, axis=0)

    action = int(np.argmax(tmp))

    if np.max(brd[0]) > 0:

        # top filled

        action = int(np.argmin(tmp))

    return action
def mark_agent(observation, configuration):

    from random import choice

    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
def victor_agent(observation, configuration):

    from random import choice

    import numpy as np

    from functools import reduce

    #print(observation.board)

    board = np.reshape(observation.board, (configuration.rows, configuration.columns))

    top_player = reduce(lambda x, y: [y[c] if x[c] == 0 else x[c] for c in range(configuration.columns)], board)

    #print(top_player)

    for i in range(len(top_player)):

        if top_player[i] == observation.mark and observation.board[i] == 0:

            return i

    if sum(map(lambda x: x == 0, board[-1])) == 0:

        return choice([c for c in range(configuration.columns) if board[-1] == 0])

    else:

        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
def peijen_agent(observation, configuration):

    import numpy as np

    def can_I_win(column, board):

        for r in range(configuration.rows):

            if board[r][column] != 0 and board[r][column] != observation.mark:

                if r >= 4:

                    return True

                else:

                    return False

            else:

                # can't really win but block the other player

                if r > 4 and board[r][column] != observation.mark:

                    return True

        return True

    # setup

    playable_columns = [c for c in range(configuration.columns) if observation.board[c] == 0]

    board = np.reshape(observation.board, (configuration.rows, configuration.columns))

    for x in range(configuration.columns):

        column = (4 + x) % configuration.columns

        if column in playable_columns and can_I_win(column, board):

            return column

    # cannot win? play 4 anyway

    return 4
def jacky_agent(observation, configuration):

    return 2
agents = {

    "Timo": timo_agent,

    "Dev": dev_agent,

    "Ilya": ilya_agent,

    "Mark": mark_agent,

    "Victor": victor_agent,

    "Peijen": peijen_agent,

    "Jacky": jacky_agent,

    "Random": "random",

#    "Negamax": "negamax"

}

    

print(agents)

num_episodes = 10

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