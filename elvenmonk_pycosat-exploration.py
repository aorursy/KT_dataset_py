import numpy as np

import pandas as pd

from numba import njit, prange

from scipy.signal import convolve2d

from tqdm.auto import tqdm

import matplotlib.pyplot as plt

import pycosat

import random



window = np.ones((3, 3))



def plot_3d(solution_3d: np.ndarray, title, size=3, max_cols=10, has_target=False):

    N = len(solution_3d)

    if N <= 0:

        return

    cols = min(N, max_cols)

    rows = (N - 1) // max_cols + 1

    plt.figure(figsize=(cols*size, rows*size))

    plt.suptitle(title)

    for t in range(N):

        board = solution_3d[t]

        plt.subplot(rows, cols, t + 1)

        plt.imshow(board, cmap='binary')

        plt.title('target' if t == 0 and has_target else f'state {t}')

    plt.show()



def csv_to_numpy_list(df) -> np.ndarray:

    return df[[ f'stop_{n}' for n in range(25**2) ]].values.reshape(-1,25,25)



def life_step(X: np.ndarray):

    C = convolve2d(X, window, mode='same', boundary='wrap')

    return (C == 3) | (X & (C == 4))
sample_submission_df = pd.read_csv('../input/conways-reverse-game-of-life-2020/sample_submission.csv', index_col='id')

test_df = pd.read_csv('../input/conways-reverse-game-of-life-2020/test.csv', index_col='id')

deltas = test_df['delta'].values

boards = csv_to_numpy_list(test_df)
for index in tqdm(range(1), total=1):

    board = np.tile(boards[index], (2, 1, 1))

    board[1] = life_step(board[0])

    plot_3d(board, 'sample board')
SIZE = 25



empty_board = np.zeros((SIZE,SIZE), dtype=bool)



# Computes number of variables.

# I negated variable states after some testing, because solver

# seems to prefer True as default state for solution variables,

# while I want cells to be empty whenever possible

def v(c):

    return -(SIZE * c[0] + c[1] + 1)



def dead_clauses(res, c, x):

    # if cell is dead, there was not exactly 3 alive neighbours (56 clauses)

    for i1 in range(0, 6):

        for i2 in range(i1+1, 7):

            for i3 in range(i2+1, 8):

                a = [v(x[i]) for i in range(8)]

                a[i1], a[i2], a[i3] = -a[i1], -a[i2], -a[i3]

                res.append(a)

    # if cell is dead and was alive, was not 2-3 alive neighbours (28 clauses)

    for i1 in range(0, 7):

        for i2 in range(i1+1, 8):

            a = [v(x[i]) if i < 8 else -v(c) for i in range(9)]

            a[i1], a[i2] = -a[i1], -a[i2]

            res.append(a)



def live_clauses(res, c, x):

    # if cell is alive, there was less then 4 alive neighbours (70 clauses)

    for i1 in range(0, 5):

        for i2 in range(i1+1, 6):

            for i3 in range(i2+1, 7):

                for i4 in range(i3+1, 8):

                    #from each 4 at least 1 was dead

                    res.append([-v(x[i1]), -v(x[i2]), -v(x[i3]), -v(x[i4])])

    # if cell is alive and was dead, there was more than 2 alive (less than 6 dead) neighbours (28 clauses)

    for i1 in range(0, 7):

        for i2 in range(i1+1, 8):

            a = [v(x[i]) if i < 8 else v(c) for i in range(9) if i != i1 and i != i2]

            res.append(a)

    # if cell is alive, there was more than 1 alive (less than 7 dead) neighbours (8 clauses)

    for i1 in range(0, 8):

        a = [v(x[i]) for i in range(8) if i != i1]

        res.append(a)



def board_clauses(board, use_opt = True):

    res, opt1, opt2 = [], [], []

    for i in range(SIZE):

        for j in range(SIZE):

            x = [((i + k % 3 - 1) % SIZE, (j + k // 3 - 1) % SIZE) for k in range(9) if k != 4]

            if board[i,j]:

                live_clauses(res, (i, j), x) # 106 clauses

            else:

                dead_clauses(res, (i, j), x) # 84 clauses

                if use_opt:

                    y = [((i + k % 5 - 2) % SIZE, (j + k // 5 - 2) % SIZE) for k in range(25)]

                    if sum(board[ii,jj] for ii,jj in y) < 1: # No alive neighbours

                        res.append([-v((i, j))]) # Very dead space MUST stay dead! (1 clause)

                    elif sum(board[ii,jj] for ii,jj in x) < 1: # No alive neighbours

                        opt1.append([-v((i, j))]) # Dead space should stay dead! (1 clause)

                    elif sum(board[ii,jj] for ii,jj in x) < 2: # Too few alive neighbours

                        opt2.append([-v((i, j))]) # Dead space should stay dead! (1 clause)



    return res, opt1, opt2
N = len(deltas)

score = 0

for n in tqdm(range(N), total=N):

    clauses, opt1, opt2 = board_clauses(boards[n], use_opt = False)

    solution = pycosat.solve(clauses)

    if isinstance(solution, str):

        print(f'{n} not solved!')

        continue

    board = np.array(solution[:SIZE**2]) < 0

    sample_submission_df.loc[test_df.index[n]] = 1 * board

    board = life_step(board.reshape(SIZE,SIZE))

    d = np.sum(board ^ boards[n])

    score += d / 625

print(score/N)
N = 1

for n in tqdm(range(N), total=N):

    T = min(deltas[n], 3)

    board = np.tile(empty_board, (T+1, 1, 1))

    board[0] = boards[n]

    solvers = [None for _ in range(T)]

    opt = [None for _ in range(T)]

    os = [0 for _ in range(T)]

    oe = [0 for _ in range(T)]

    t = 0

    while 0 <= t and t < T:

        if solvers[t] is None:

            clauses, opt1, opt2 = board_clauses(board[t])

            solution = pycosat.solve(clauses)

            if not isinstance(solution, str):

                if t == T - 1:

                    print(t, '!!', end=" ")

                    t += 1

                    board[t] = np.array(solution[:SIZE**2]).reshape(SIZE,SIZE) < 0

                    continue

                else:

                    print(t, '??', end=" ")

                    random.shuffle(opt1)

                    random.shuffle(opt2)

                    opt[t] = opt1 + opt2

                    os[t] = len(opt[t])+1

                    oe[t] = len(opt[t])+1

                    solvers[t] = pycosat.itersolve(clauses + opt[t])

                    print(len(opt[t]), end=" ")

        try:

            solution = next(solvers[t])

            if oe[t] - os[t] > 1:

                os[t] = (os[t]+oe[t])//2

                solvers[t] = pycosat.itersolve(clauses + opt[t][:(os[t]+oe[t])//2])

                print((os[t]+oe[t])//2, end=" ")

            else:

                print(t, '++', end=" ")

                t += 1

                board[t] = np.array(solution[:SIZE**2]).reshape(SIZE,SIZE) < 0

        except Exception as err:

            if solvers[t] is not None and (os[t]+oe[t])//2 > 0:

                if os[t] == oe[t]:

                    os[t] = 0

                    oe[t] = len(opt[t])+1

                elif oe[t] - os[t] > 1:

                    oe[t] = (os[t]+oe[t])//2

                else:

                    os[t] -= 1

                    oe[t] -= 1

                solvers[t] = pycosat.itersolve(clauses + opt[t][:(os[t]+oe[t])//2])

                print((os[t]+oe[t])//2, end=" ")

            else:

                print(t, '--', end=" ")

                solvers[t] = None

                opt[t] = None

                t -= 1

    if t == T:

        sample_submission_df.loc[test_df.index[n]] = 1 * board[T].ravel()

    plot_3d(board, index)
sample_submission_df.to_csv("submission.csv", index=True)
submission_df = pd.read_csv('./submission.csv', index_col='id')