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
from kaggle_environments import make, evaluate

# Create the game environment
# Set debug=True to see the errors if your agent refuses to run
env = make("connectx", debug=True)

# List of available default agents
print(list(env.agents))
def agentOne(obs, config):
    board_pos = obs.board
    my_mark = obs.mark
    num_cols = config.columns
    num_rows = config.rows
    windiscs = config.inarow

    marks = [1, 2]
    marks.remove(my_mark)

    np_board_pos = np.array(board_pos).reshape((num_rows, num_cols))

    # count repeat of 1
    repeat_1 = 0
    for c in board_pos:
        if c == 1:
            repeat_1 += 1

    if repeat_1 == 0:
        return num_cols // 2

    opp_mark = marks[0]
    defense_col = []

    for occ in range(1, windiscs):
        for i in range(num_rows):
            for j in range(num_cols):
                if np_board_pos[i, j] != 0:
                    if np_board_pos[i, j] == my_mark:
                        # check row for consecutive occurences of 1
                        if i != 0:
                            if j + windiscs - occ < num_cols - 1:
                                if np_board_pos[i - 1, j + windiscs - occ] == 0:
                                    cons_occ_row = 0
                                    for k in range(windiscs - occ):
                                        if j + k > num_cols - 1:
                                            break
                                        if np_board_pos[i, j + k] == my_mark:
                                            cons_occ_row += 1
                                        elif np_board_pos[i, j + k] == 0:
                                            if j + k + 1 < num_cols:
                                                if np_board_pos[i, j + k + 1] == my_mark:
                                                    if np_board_pos[i - 1, j + k] == 0:
                                                        return j + k
                                        else:
                                            break
                                    if cons_occ_row == windiscs - occ:
                                        return j + cons_occ_row

                        # check column for consecutive occurences of 1
                        if i != 0:
                            if np_board_pos[i - 1, j] == 0:
                                cons_occ_col = 0
                                for k in range(windiscs - occ):
                                    if i + k > num_rows - 1:
                                        break
                                    if np_board_pos[i + k, j] == my_mark:
                                        cons_occ_col += 1
                                    else:
                                        break
                                if cons_occ_col == windiscs - occ:
                                    return j

                        # check diagonal for consecutive occurences of 1
                        if 0 < i < num_rows - 1 and 0 < j < num_cols - 1:
                            if np_board_pos[i - 1, j - 1] == 0:
                                cons_occ_dia_negative = 0
                                for k in range(windiscs - occ):
                                    if i + k > num_rows - 1:
                                        break
                                    if j + k > num_cols - 1:
                                        break
                                    if np_board_pos[i + k, j + k] == my_mark:
                                        cons_occ_dia_negative += 1
                                    elif np_board_pos[i + k, j + k] == 0:
                                        if i + k + 1 < num_rows and j + k + 1 < num_cols:
                                            if np_board_pos[i + k + 1, j + k + 1] == my_mark:
                                                if np_board_pos[i + k - 1, j + k] == 0:
                                                    return j + k
                                    else:
                                        break
                                if cons_occ_dia_negative == windiscs - occ:
                                    return j - 1
                            if np_board_pos[i - 1, j + 1] == 0:
                                cons_occ_dia_positive = 0
                                for k in range(windiscs - occ):
                                    if i + k > num_rows - 1:
                                        break
                                    if j - k < 0:
                                        break
                                    if np_board_pos[i + k, j - k] == my_mark:
                                        cons_occ_dia_positive += 1
                                    elif np_board_pos[i + k, j - k] == 0:
                                        if i + k + 1 < num_rows and j - k - 1 >= 0:
                                            if np_board_pos[i + k + 1, j - k - 1] == my_mark:
                                                if np_board_pos[i + k - 1, j - k] == 0:
                                                    return j - k
                                    else:
                                        break
                                if cons_occ_dia_positive == windiscs - occ:
                                    return j + 1

                    elif np_board_pos[i, j] == opp_mark:
                        # check row for consecutive occurences of 2
                        if i != 0:
                            if j + windiscs - occ < num_cols - 1:
                                if np_board_pos[i - 1, j + windiscs - occ] == 0:
                                    cons_occ_row = 0
                                    for k in range(windiscs - occ):
                                        if j + k > num_cols - 1:
                                            break
                                        if np_board_pos[i, j + k] == opp_mark:
                                            cons_occ_row += 1
                                        elif np_board_pos[i, j + k] == 0:
                                            if j + k + 1 < num_cols:
                                                if np_board_pos[i, j + k + 1] == opp_mark:
                                                    if np_board_pos[i - 1, j + k] == 0:
                                                        defense_col.append(j + k)
                                        else:
                                            break

                                    if cons_occ_row == windiscs - occ:
                                        if j + cons_occ_row <= num_cols:
                                            defense_col.append(j + cons_occ_row)

                        # check column for consecutive occurences of 2
                        if i != 0:
                            if np_board_pos[i - 1, j] == 0:
                                cons_occ_col = 0
                                for k in range(windiscs - occ):
                                    if i + k > num_rows - 1:
                                        break
                                    if np_board_pos[i + k, j] == opp_mark:
                                        cons_occ_col += 1
                                    else:
                                        break
                                if cons_occ_col == windiscs - occ:
                                    defense_col.append(j)

                        # check diagonal for consecutive occurences of 2
                        if 0 < i < num_rows - 1 and 0 < j < num_cols - 1:
                            if np_board_pos[i - 1, j - 1] == 0:
                                cons_occ_dia_negative = 0
                                for k in range(windiscs - occ):
                                    if i + k > num_rows - 1:
                                        break
                                    if j + k > num_cols - 1:
                                        break
                                    if np_board_pos[i + k, j + k] == opp_mark:
                                        cons_occ_dia_negative += 1
                                    elif np_board_pos[i + k, j + k] == 0:
                                        if i + k + 1 < num_rows and j + k + 1 < num_cols:
                                            if np_board_pos[i + k + 1, j + k + 1] == opp_mark:
                                                if np_board_pos[i + k - 1, j + k] == 0:
                                                    defense_col.append(j + k)
                                    else:
                                        break
                                if cons_occ_dia_negative == windiscs - occ:
                                    defense_col.append(j - 1)

                            if np_board_pos[i - 1, j + 1] == 0:
                                cons_occ_dia_positive = 0
                                for k in range(windiscs - occ):
                                    if i + k > num_rows - 1:
                                        break
                                    if j - k < 0:
                                        break
                                    if np_board_pos[i + k, j - k] == opp_mark:
                                        cons_occ_dia_positive += 1
                                    elif np_board_pos[i + k, j - k] == 0:
                                        if i + k + 1 < num_rows and j - k - 1 >= 0:
                                            if np_board_pos[i + k + 1, j - k - 1] == opp_mark:
                                                if np_board_pos[i + k - 1, j - k] == 0:
                                                    defense_col.append(j - k)
                                    else:
                                        break
                                if cons_occ_dia_positive == windiscs - occ:
                                    defense_col.append(j)

            if len(defense_col) > 0:
                return defense_col[0]


def agentOO7(obs, config):
    board_pos = obs.board
    my_mark = obs.mark
    num_cols = config.columns
    num_rows = config.rows
    windiscs = config.inarow

    marks = [1, 2]
    marks.remove(my_mark)
    opp_mark = marks[0]

    np_board_pos = np.array(board_pos).reshape((num_rows, num_cols))

    # count repeat of 1
    repeat_my_mark = 0
    for c in board_pos:
        if c == my_mark:
            repeat_my_mark += 1

    if repeat_my_mark == 0:
        return num_cols // 2

    col_fill_prob = {i: j for i in range(num_cols) for j in [0]}

    win = 49
    defense = 40
    critical_pos = []

    # Check columns
    for col in range(num_cols):
        space_1d = np_board_pos[:, col]
        if 0 in space_1d and (my_mark in space_1d or opp_mark in space_1d):
            num_zeroes = 0
            slice_size = windiscs
            for element in space_1d:
                if element == 0:
                    num_zeroes += 1
                else:
                    break

            check_space = space_1d[num_zeroes - 1: num_zeroes + slice_size - 1]
            len_check_space = len(check_space)
            i = 1
            probability_win = 0
            while i == 1:
                if check_space[i] == my_mark:
                    probability_win += 1 / (slice_size - 1)
                    if len_check_space > 2:
                        if check_space[i + 1] == my_mark:
                            probability_win += (1 / (slice_size - 1))
                            if len_check_space > 3:
                                if check_space[i + 2] == my_mark:
                                    return col
                                else:
                                    break
                        else:
                            break
                elif check_space[i] == opp_mark:
                    probability_win += 1 / (slice_size - 1)
                    if len_check_space > 2:
                        if check_space[i + 1] == opp_mark:
                            probability_win += 1 / (slice_size - 1)
                            if len_check_space > 3:
                                if check_space[i + 2] == opp_mark:
                                    critical_pos.append(col)
                                else:
                                    probability_win = 0
                                    break
                        else:
                            probability_win = 0
                            break

                i += 1

            col_fill_prob[col] += probability_win

        elif 0 not in space_1d:
            del col_fill_prob[col]

    # Check rows
    for row in range(num_rows - 1, -1, -1):
        space_1d = np_board_pos[row, :]
        if (0 in space_1d and my_mark in space_1d) or (0 in space_1d and opp_mark in space_1d):
            space_size = len(space_1d)
            slice_size = windiscs

            n = 0
            while slice_size + n <= space_size:
                slice = space_1d[n: slice_size + n]
                if 0 in slice:
                    pos0 = []
                    pos_my_mark = []
                    pos_opp_mark = []
                    size = 0
                    while size < slice_size:
                        if slice[size] == 0:
                            pos0.append(size)
                        elif slice[size] == my_mark:
                            pos_my_mark.append(size)
                        elif slice[size] == opp_mark:
                            pos_opp_mark.append(size)
                        size += 1

                    unique_element_len = len(set(slice))
                    if row < num_rows - 1:
                        if np_board_pos[row + 1, n + pos0[0]] == 0:
                            n += 1
                            continue

                    if unique_element_len == 2:
                        for mark in set(slice):
                            if mark == my_mark:
                                if len(pos_my_mark) == 3:
                                    return n + pos0[0]
                                else:
                                    probability_win = len(pos_my_mark) / (slice_size - 1)
                                col_fill_prob[n + pos0[0]] += probability_win
                            elif mark == opp_mark:
                                if len(pos_opp_mark) == 3:
                                    critical_pos.append(n + pos0[0])
                                else:
                                    probability_win = len(pos_opp_mark) / (slice_size - 1)
                                    col_fill_prob[n + pos0[0]] += probability_win
                    elif unique_element_len == 3:
                        for mark in set(slice):
                            if mark == my_mark:
                                pos_advantage = -1
                                for posimymark in pos_my_mark:
                                    for posi0 in pos0:
                                        distance = abs(posimymark - posi0)
                                        if distance == 1:
                                            if pos_advantage == -1:
                                                pos_advantage = 0.1
                                            else:
                                                pos_advantage += 0.1
                                for posimymark1 in pos_my_mark:
                                    for posimymark2 in pos_my_mark:
                                        distance = abs(posimymark1 - posimymark2)
                                        if distance == 1:
                                            if pos_advantage == -1:
                                                pos_advantage = 0.1
                                            else:
                                                pos_advantage += 0.1

                                probability_win = (len(pos_my_mark) / (slice_size - 1)) + pos_advantage
                                col_fill_prob[n + pos0[0]] += probability_win
                            elif mark == opp_mark:
                                pos_advantage = -1
                                for posioppmark in pos_opp_mark:
                                    for posi0 in pos0:
                                        distance = abs(posioppmark - posi0)
                                        if distance == 1:
                                            if pos_advantage == -1:
                                                pos_advantage = 0.1
                                            else:
                                                pos_advantage += 0.1
                                probability_win = (len(pos_opp_mark) / slice_size - 1) + pos_advantage
                                col_fill_prob[n + pos0[0]] += probability_win

                n += 1

    # Check diagonal
    for row in range(num_rows - 1, -1, -1):
        for col in range(num_cols - 1, -1, -1):
            if np_board_pos[row, col] >= 0:
                # right-side diagonal
                if abs(num_cols - 1 - col) >= windiscs - 1 and row >= windiscs - 1:
                    check_space = []
                    for d in range(windiscs):
                        check_space.append(np_board_pos[row - d, col + d])
                    pos0 = []
                    pos_my_mark = []
                    pos_opp_mark = []
                    size = 0
                    while size < windiscs:
                        if check_space[size] == 0:
                            pos0.append(size)
                        elif check_space[size] == my_mark:
                            pos_my_mark.append(size)
                        elif check_space[size] == opp_mark:
                            pos_opp_mark.append(size)
                        size += 1

                    if len(pos0) == 1:
                        if row == num_rows - 1 and pos0[0] == 0:
                            row -= 1
                        if np_board_pos[row - pos0[0] + 1, col + pos0[0]] != 0:
                            unique_element_len = len(set(check_space))
                            if unique_element_len == 2:
                                for mark in set(check_space):
                                    if mark == my_mark:
                                        return col + pos0[0]
                                    elif mark == opp_mark:
                                        critical_pos.append(col + pos0[0])
                            elif unique_element_len == 3:
                                for mark in set(check_space):
                                    if mark == my_mark:
                                        pos_advantage = -1
                                        for posimymark in pos_my_mark:
                                            for posi0 in pos0:
                                                distance = abs(posimymark - posi0)
                                                if distance == 1:
                                                    if pos_advantage == -1:
                                                        pos_advantage = 0.1
                                                    else:
                                                        pos_advantage += 0.1
                                        for posimymark1 in pos_my_mark:
                                            for posimymark2 in pos_my_mark:
                                                distance = abs(posimymark1 - posimymark2)
                                                if distance == 1:
                                                    if pos_advantage == -1:
                                                        pos_advantage = 0.1
                                                    else:
                                                        pos_advantage += 0.1

                                        probability_win = (len(pos_my_mark) / (windiscs - 1)) + pos_advantage
                                        col_fill_prob[col + pos0[0]] += probability_win
                                    elif mark == opp_mark:
                                        pos_advantage = -1
                                        for posioppmark in pos_opp_mark:
                                            for posi0 in pos0:
                                                distance = abs(posioppmark - posi0)
                                                if distance == 1:
                                                    if pos_advantage == -1:
                                                        pos_advantage = 0.1
                                                    else:
                                                        pos_advantage += 0.1
                                        probability_win = (len(pos_opp_mark) / windiscs - 1) + pos_advantage
                                        col_fill_prob[col + pos0[0]] += probability_win

                        else:
                            unique_element_len = len(set(check_space))
                            if unique_element_len == 2:
                                for mark in set(check_space):
                                    if mark == my_mark:
                                        probability_win = (len(pos_my_mark) / (windiscs - 1)) + 3
                                        col_fill_prob[col + pos0[0]] += probability_win
                                    elif mark == opp_mark:
                                        probability_win = -2
                                        col_fill_prob[col + pos0[0]] += probability_win

                # left-side diagonal
                if col >= windiscs - 1 and row >= windiscs - 1:
                    check_space = []
                    for d in range(windiscs):
                        check_space.append(np_board_pos[row - d, col - d])
                    pos0 = []
                    pos_my_mark = []
                    pos_opp_mark = []
                    size = 0
                    while size < windiscs:
                        if check_space[size] == 0:
                            pos0.append(size)
                        elif check_space[size] == my_mark:
                            pos_my_mark.append(size)
                        elif check_space[size] == opp_mark:
                            pos_opp_mark.append(size)
                        size += 1

                    if len(pos0) == 1:
                        if row == num_rows - 1 and pos0[0] == 0:
                            row -= 1
                        if np_board_pos[row - pos0[0] + 1, col - pos0[0]] != 0:
                            unique_element_len = len(set(check_space))
                            if unique_element_len == 2:
                                for mark in set(check_space):
                                    if mark == my_mark:
                                        return col - pos0[0]
                                    elif mark == opp_mark:
                                        critical_pos.append(col - pos0[0])
                            elif unique_element_len == 3:
                                for mark in set(check_space):
                                    if mark == my_mark:
                                        pos_advantage = -1
                                        for posimymark in pos_my_mark:
                                            for posi0 in pos0:
                                                distance = abs(posimymark - posi0)
                                                if distance == 1:
                                                    if pos_advantage == -1:
                                                        pos_advantage = 0.1
                                                    else:
                                                        pos_advantage += 0.1
                                        for posimymark1 in pos_my_mark:
                                            for posimymark2 in pos_my_mark:
                                                distance = abs(posimymark1 - posimymark2)
                                                if distance == 1:
                                                    if pos_advantage == -1:
                                                        pos_advantage = 0.1
                                                    else:
                                                        pos_advantage += 0.1

                                        probability_win = (len(pos_my_mark) / (windiscs - 1)) + pos_advantage
                                        col_fill_prob[col - pos0[0]] += probability_win
                                    elif mark == opp_mark:
                                        pos_advantage = -1
                                        for posioppmark in pos_opp_mark:
                                            for posi0 in pos0:
                                                distance = abs(posioppmark - posi0)
                                                if distance == 1:
                                                    if pos_advantage == -1:
                                                        pos_advantage = 0.1
                                                    else:
                                                        pos_advantage += 0.1
                                        probability_win = (len(pos_opp_mark) / windiscs - 1) + pos_advantage
                                        col_fill_prob[col - pos0[0]] += probability_win

                        else:
                            unique_element_len = len(set(check_space))
                            if unique_element_len == 2:
                                for mark in set(check_space):
                                    if mark == my_mark:
                                        probability_win = (len(pos_my_mark) / (windiscs - 1)) + 3
                                        col_fill_prob[col - pos0[0]] += probability_win
                                    elif mark == opp_mark:
                                        probability_win = -2
                                        col_fill_prob[col - pos0[0]] += probability_win

    if len(critical_pos) > 0:
        return critical_pos[0]

    max_prob = max(list(col_fill_prob.values()))
    for key in col_fill_prob.keys():
        if col_fill_prob[key] == max_prob:
            return key
        
        

def agentVinod(obs, config):
    board_pos = obs.board
    my_mark = obs.mark
    num_cols = config.columns
    num_rows = config.rows
    windiscs = config.inarow

    marks = [1, 2]
    marks.remove(my_mark)
    opp_mark = marks[0]

    for _ in range(num_cols):
        board_pos.append(my_mark)

    np_board_pos = np.array(board_pos).reshape((num_rows + 1, num_cols))

    def probability_distribution(num_parts, consecutive_diff=0.2, total_prob=3):
        pd = []
        if num_parts % 2 == 1:
            centre_prob = (total_prob / num_parts) + \
                          (consecutive_diff / 4) * ((num_parts - 1) * (num_parts + 1) / num_parts)
            pd.append(centre_prob)
            next_prob = centre_prob
            for _ in range(int((num_parts - 1) / 2)):
                next_prob -= consecutive_diff
                pd.append(next_prob)
                pd.insert(0, next_prob)
        else:
            centre_prob = (total_prob / num_parts) + (consecutive_diff / 4) * (num_parts - 2)
            next_prob = centre_prob
            for _ in range(int(num_parts / 2)):
                pd.append(next_prob)
                pd.insert(0, next_prob)
                next_prob -= consecutive_diff

        pd_dict = {k: pd[k] for k in range(num_parts)}
        return pd_dict

    col_fill_prob = probability_distribution(num_cols)
    critical_pos = []
    worst_col = []

    # Check columns
    for col in range(num_cols):
        space_1d = np_board_pos[:, col]
        if 0 in space_1d:
            num_zeroes = 0
            for element in space_1d:
                if element == 0:
                    num_zeroes += 1

            if num_zeroes < num_rows:
                check_space = space_1d[num_zeroes - 1: num_zeroes + windiscs - 1]
                len_check_space = len(check_space)
            # i = 1
            # probability_win = 0
            # if len_check_space == windiscs:
            #     if num_zeroes < windiscs - 1:
            #         if (check_space[1] == my_mark and check_space[2] == opp_mark) or \
            #                 (check_space[1] == opp_mark and check_space[2] == my_mark):
            #             continue

                unique_marks = set(check_space)

                if len(unique_marks) == 2:
                    for mark in unique_marks:
                        num_my_mark = 0
                        for nmm in check_space:
                            if nmm == my_mark:
                                num_my_mark += 1
                                
                        if mark == my_mark:
                            if num_my_mark == 3:
                                critical_pos.append(col)
                            else:
                                probability_win = (num_my_mark + 1) / windiscs
                                col_fill_prob[col] += probability_win
                        elif mark == opp_mark:
                            num_opp_mark = 0
                            for nom in check_space:
                                if nom == opp_mark:
                                    num_opp_mark += 1
                                    
                            if num_opp_mark == 3:
                                critical_pos.append(col)
                            elif len_check_space - 1 == 2:
                                probability_win = (num_opp_mark + 1) / windiscs
                                col_fill_prob[col] += probability_win

            # while i == 1:
            #     if check_space[i] == my_mark:
            #         probability_win += 2 / windiscs
            #         if len_check_space > 2:
            #             if check_space[i + 1] == my_mark:
            #                 probability_win += (1 / windiscs)
            #                 if len_check_space > 3:
            #                     if check_space[i + 2] == my_mark:
            #                         return col
            #
            #     elif check_space[i] == opp_mark:
            #         if len_check_space > 2:
            #             if check_space[i + 1] == opp_mark:
            #                 probability_win += 3 / windiscs
            #                 if len_check_space > 3:
            #                     if check_space[i + 2] == opp_mark:
            #                         critical_pos.append(col)
            #
            #     i += 1

        elif 0 not in space_1d:
            del col_fill_prob[col]

    # Check rows
    for row in range(num_rows - 1, -1, -1):
        space_1d = np_board_pos[row, :]
        if (0 in space_1d and my_mark in space_1d) or (0 in space_1d and opp_mark in space_1d):
            space_size = len(space_1d)

            n = 0
            while windiscs + n <= space_size:
                slice_1d = space_1d[n: windiscs + n]
                if 0 in slice_1d:
                    pos0 = []
                    pos_my_mark = []
                    pos_opp_mark = []
                    size = 0
                    while size < windiscs:
                        if slice_1d[size] == 0:
                            pos0.append(size)
                        elif slice_1d[size] == my_mark:
                            pos_my_mark.append(size)
                        elif slice_1d[size] == opp_mark:
                            pos_opp_mark.append(size)
                        size += 1

                    unique_element_len = len(set(slice_1d))

                    for p in range(len(pos0)):
                        if np_board_pos[row + 1, n + pos0[p]] != 0:
                            if unique_element_len == 2:
                                for mark in set(slice_1d):
                                    if mark == my_mark:
                                        if len(pos_my_mark) == 3:
                                            return n + pos0[p]
                                        else:
                                            probability_win = (len(pos_my_mark) + 1) / windiscs
                                            col_fill_prob[n + pos0[p]] += probability_win
                                    elif mark == opp_mark:
                                        if len(pos_opp_mark) == 3:
                                            critical_pos.append(n + pos0[p])
                                        elif len(pos_opp_mark) == 2:
                                            probability_win = (len(pos_opp_mark) + 1) / windiscs
                                            col_fill_prob[n + pos0[p]] += probability_win

                        else:
                            unique_element_len = len(set(slice_1d))
                            if unique_element_len == 2:
                                for mark in set(slice_1d):
                                    if mark == opp_mark:
                                        if len(pos_opp_mark) == 3:
                                            worst_col.append(n + pos0[0])
                n += 1

    # Check diagonal
    for row in range(num_rows - 1, -1, -1):
        for col in range(num_cols - 1, -1, -1):
            if np_board_pos[row, col] >= 0:
                # right-side diagonal
                if abs(num_cols - 1 - col) >= windiscs - 1 and row >= windiscs - 1:
                    check_space = []
                    for d in range(windiscs):
                        check_space.append(np_board_pos[row - d, col + d])
                    pos0 = []
                    pos_my_mark = []
                    pos_opp_mark = []
                    size = 0
                    while size < windiscs:
                        if check_space[size] == 0:
                            pos0.append(size)
                        elif check_space[size] == my_mark:
                            pos_my_mark.append(size)
                        elif check_space[size] == opp_mark:
                            pos_opp_mark.append(size)
                        size += 1

                    if 1 <= len(pos0) <= 2:
                        for p in range(len(pos0)):
                            if np_board_pos[row - pos0[p] + 1, col + pos0[p]] != 0:
                                unique_element_len = len(set(check_space))
                                if unique_element_len == 2:
                                    for mark in set(check_space):
                                        if mark == my_mark:
                                            if len(pos_my_mark) == 3:
                                                return col + pos0[p]
                                            else:
                                                probability_win = (len(pos_my_mark) + 1) / windiscs
                                                col_fill_prob[col + pos0[p]] += probability_win
                                        elif mark == opp_mark:
                                            if len(pos_opp_mark) == 3:
                                                critical_pos.append(col + pos0[p])
                                            else:
                                                probability_win = (len(pos_opp_mark) + 1) / windiscs
                                                col_fill_prob[col + pos0[p]] += probability_win

                            else:
                                unique_element_len = len(set(check_space))
                                if unique_element_len == 2:
                                    for mark in set(check_space):
                                        if mark == opp_mark:
                                            if len(pos_opp_mark) == 3:
                                                worst_col.append(col + pos0[0])

                    else:
                        for p in range(len(pos0)):
                            if row == num_rows - 1 and pos0[p] == 0:
                                row -= 1
                            if np_board_pos[row - pos0[p] + 1, col + pos0[p]] != 0:
                                unique_element_len = len(set(check_space))
                                if unique_element_len == 2:
                                    for mark in set(check_space):
                                        if mark == my_mark:
                                            probability_win = (len(pos_my_mark) + 1) / windiscs
                                            col_fill_prob[col + pos0[p]] += probability_win

                # left-side diagonal
                if col >= windiscs - 1 and row >= windiscs - 1:
                    check_space = []
                    for d in range(windiscs):
                        check_space.append(np_board_pos[row - d, col - d])
                    pos0 = []
                    pos_my_mark = []
                    pos_opp_mark = []
                    size = 0
                    while size < windiscs:
                        if check_space[size] == 0:
                            pos0.append(size)
                        elif check_space[size] == my_mark:
                            pos_my_mark.append(size)
                        elif check_space[size] == opp_mark:
                            pos_opp_mark.append(size)
                        size += 1

                    if 1 <= len(pos0) <= 2:
                        for p in range(1, len(pos0) + 1):
                            if np_board_pos[row - pos0[-p] + 1, col - pos0[-p]] != 0:
                                unique_element_len = len(set(check_space))
                                if unique_element_len == 2:
                                    for mark in set(check_space):
                                        if mark == my_mark:
                                            if len(pos_my_mark) == 3:
                                                return col - pos0[-p]
                                            else:
                                                probability_win = (len(pos_my_mark) + 1) / windiscs
                                                col_fill_prob[col - pos0[-p]] += probability_win
                                        elif mark == opp_mark:
                                            if len(pos_opp_mark) == 3:
                                                critical_pos.append(col - pos0[0])
                                            else:
                                                probability_win = (len(pos_opp_mark) + 1) / windiscs
                                                col_fill_prob[col - pos0[-p]] += probability_win

                            else:
                                unique_element_len = len(set(check_space))
                                if unique_element_len == 2:
                                    for mark in set(check_space):
                                        if mark == opp_mark:
                                            if len(pos_opp_mark) == 3:
                                                worst_col.append(col - pos0[0])
                    else:
                        for p in range(1, len(pos0) + 1):
                            if row == num_rows - 1 and pos0[-p] == 0:
                                row -= 1
                            if np_board_pos[row - pos0[-p] + 1, col - pos0[-p]] != 0:
                                unique_element_len = len(set(check_space))
                                if unique_element_len == 2:
                                    for mark in set(check_space):
                                        if mark == my_mark:
                                            probability_win = (len(pos_my_mark) / (windiscs - 1))
                                            col_fill_prob[col - pos0[-p]] += probability_win
                                        elif mark == opp_mark:
                                            probability_win = 0
                                            col_fill_prob[col - pos0[-p]] += probability_win

    if len(critical_pos) > 0:
        return critical_pos[0]

    key_near2centre = 0
    distance = num_cols // 2
    max_prob = max(list(col_fill_prob.values()))

    for worst_key in set(worst_col):
        del col_fill_prob[worst_key]

    for key in col_fill_prob.keys():
        if col_fill_prob[key] == max_prob:
            dist = abs((num_cols//2) - key)
            if dist <= distance:
                key_near2centre = key

    return key_near2centre
# Agents play one game round
env.run([agentVinod, agentVinod])

# Show the game
env.render(mode="ipython")
