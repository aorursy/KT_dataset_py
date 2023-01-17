def agent_blocker(obs, config):

    my_mark = obs.mark

    for area in range(obs.board):

        if (obs.board[area] == 0):

            if (not (obs.board[area + 1] == 0 or obs.board[area + 1] == my_mark) or not (obs.board[area - 1] == 0 or obs.board[area - 1] == 1) or not (obs.board[area + 8] == 0 or obs.board[area + 8] == 1 or obs.board[area - 9])):

                return area
def agent_blocker(obs, config):

    my_mark = obs.mark

    for area in range(obs.board):

        if (obs.board[area] == 0):

            if (not (obs.board[area + 1] == 0 or obs.board[area + 1] == my_mark) or not (obs.board[area - 1] == 0 or obs.board[area - 1] == 1) or not (obs.board[area + 8] == 0 or obs.board[area + 8] == 1 or obs.board[area - 9])):

                return area

    for area in range(obs.board):

        if (obs.board[area] == 0):

            if (obs.board[area + 1] == my_mark or obs.board[area - 1] == my_mark or obs.board[config.columns + 1 + area] == my_mark):

                return area
def agent_blocker(obs, config):

    my_mark = obs.mark

    for area in range(obs.board):

        if (obs.board[area] == 0):

            if (not (obs.board[area + 1] == 0 or obs.board[area + 1] == my_mark) or not (obs.board[area - 1] == 0 or obs.board[area - 1] == 1) or not (obs.board[area + 8] == 0 or obs.board[area + 8] == 1 or obs.board[area - 9])):

                return area

    for area in range(obs.board):

        if (obs.board[area] == 0):

            if (obs.board[area + 1] == my_mark or obs.board[area - 1] == my_mark or obs.board[config.columns + 1 + area] == my_mark):

                return area

    for area in range(obs.board):

        if (obs.board[area] == 0):

            return area
%%writefile "submission.py"



def agent_blocker(obs, config):

    my_mark = obs.mark

    for area in range(obs.board):

        if (obs.board[area] == 0):

            if (not (obs.board[area + 1] == 0 or obs.board[area + 1] == my_mark) or not (obs.board[area - 1] == 0 or obs.board[area - 1] == 1) or not (obs.board[area + 8] == 0 or obs.board[area + 8] == 1 or obs.board[area - 9])):

                return area

    for area in range(obs.board):

        if (obs.board[area] == 0):

            if (obs.board[area + 1] == my_mark or obs.board[area - 1] == my_mark or obs.board[config.columns + 1 + area] == my_mark):

                return area

    for area in range(obs.board):

        if (obs.board[area] == 0):

            return area