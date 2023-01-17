# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
N = 8

board_state_memory = {}

board = np.zeros((N,N), np.int8)
def create_board_string(board):

    board_string = ''

    for i in range(N):

        for j in range(N):

            board_string += str(board[i][j])

    return board_string

create_board_string(board)
board_copy = board.copy()

board_copy[0,1] =1
create_board_string(board_copy)
def is_board_safe(board):

    board_key = create_board_string(board)

    if board_key in board_state_memory:

        print("Using Cached Memory")

        return board_state_memory[board_key]

    row_sum = np.sum(board, axis=1)

    if (len(row_sum[np.where(row_sum>1)])>0):

        board_state_memory[board_key] = False

        return False

    

    col_sum = np.sum(board, axis=0)

    if (len(col_sum[np.where(col_sum>1)])>0):

        board_state_memory[board_key] = False

        return False

    

    diags = [board[::-1, :].diagonal(i) for i in range(-board.shape[0]+1, board.shape[1])]

    diags.extend(board.diagonal(i) for i in range(board.shape[1]-1, -board.shape[0], -1))

    

    for diag in diags:

        if np.sum(diag) > 1:

            board_state_memory[board_key] = False

            return False

        

    board_state_memory[board_key] = True

    return True
board_copy = board.copy()
board_copy[0][0] =1

board_copy[0][3] =1
print(board_copy)

is_board_safe(board_copy)
board_copy = board.copy()

board_copy[3][3] =1

board_copy[0][3] =1

print(board_copy)

is_board_safe(board_copy)
board_copy = board.copy()

board_copy[3][3] =1

board_copy[2][4] =1

print(board_copy)

is_board_safe(board_copy)
def place_queen(board, column):

    if column >= N :

        return True

    for row in range(N):

        board[row][column] = 1

        

        safe = False

        if is_board_safe(board):

            safe = place_queen(board, column+1)

        

        if not safe:

            board[row][column] = 0

            

        else:

            break

            

    return safe
board_state_memory
board = np.zeros((N,N ), np.int8)

placed = place_queen(board, 0)

print(placed)
board
board_state_memory
board = np.zeros((N,N ), np.int8)

placed = place_queen(board, 2)

print(placed)
board
test_np = np.arange(64).reshape(8,8)

test_np
test_np[::-1,:]
board = test_np

diags = [board[::-1, :].diagonal(i) for i in range(-board.shape[0]+1, board.shape[1])]

diags
diags1 = [board.diagonal(i) for i in range(board.shape[1]-1, -board.shape[0], -1)]

diags1
nums = 10%2 * list('018') or ['']

nums
l1 = [0,1,2,3,4]

l1.append(5)

l1
l1[0:0] = [-2, -1]

l1
l1[7:7] = [6 ,7]

l1
def op_dict(operator: str, x: float, y: float) -> float:

     return {

         '+': lambda: x + y,

         '-': lambda: x - y,

         '*': lambda: x * y,

         '/': lambda: x / y,

     }.get(operator, lambda : None)()



op_dict('%', 2, 3)
from functools import wraps



def memoization(func):

    cache = {}

    miss = object()

    

    @wraps(func)

    def wrapper(*args):

        result = cache.get(args, miss)

        if result is miss:

            result = func(*args)

            cache[args] = result

        return result

    

    return wrapper



@memoization

def fib(n):

    if n < 2:

        return n

    return fib(n-1) + fib(n-2)

    
from functools import partial

maxret = partial(max)

maxret.__doc__ = 'Returns Max in a list'

maxret([1,2,3,4])
def is_subsequence(s: str, t: str) -> bool:

    t = iter(t)

    return all(c in t for c in s)
is_subsequence('abc', 'abcdef')
from functools import wraps



def memoization(func):

    cache = {}

    miss = object()

    

    @wraps(func)

    def wrapper(*args):

        result = cache.get(args, miss)

        if result is miss:

            result = func(*args)

            cache[args] = result

        return result

    

    return wrapper



@memoization

def fact(n):

    if n < 2:

        return n

    return n* fact(n-1)



fact(5)
list(map(fib, range(5)))
a = np.random.randint(1, 10, 9).reshape(3,3)



b = np.random.randint(10, 20, 9).reshape(3,3)



c = np.random.randint(20, 30, 9).reshape(3,3)

a,b,c
from functools import reduce
reduce(lambda x, y: x if x> y else y,( list(map(fib, range(10)))))
a
list(map(list, zip(*a)))
keys = np.arange(10)

keys = [str(key) for key in keys]

values = np.arange(10)

kvs=dict(zip(keys, values))

kvs
a = np.arange(10)

a
def fibo(n):

    a,b,counter = 0,1,0

    while counter <=n :

        yield a

        a, b = b,a+b

        counter += 1

    

f = fibo(5)

while True:

    try:

        print(next(f), end=' ')

    except StopIteration:

        break
"""add a sentinel n at the end (which is the appropriate last insertion index then)"""

# L47: given a collection of numbers that might contain duplicates, return all possible unique permutations.

def permute_unique(nums):

    perms = [[]]

    for n in nums:

        perms = [p[:i] + [n] + p[i:]

                 for p in perms

                 for i in range((p + [n]).index(n) + 1)]

    return perms



permute_unique([1,1])
p = []

p = p[:0] + [1] + p[0:]

p
p = p[:0] + [1] + p[0:]

p
arr = np.arange(9).reshape(3,3)

arr

def get_element(matrix, i, j) :

    return matrix[i][j] if 0 <= i < matrix.shape[0] and 0 <= j < matrix.shape[1] else -1



get_element(arr, 0, 2)
origin_list = [1, 2, 3, 3, 4, 3, 5]

break_elment = 4



new_list = [a for end in [[]] for a in origin_list

         if not end and not (a == break_elment and end.append(-1))]

# output: [1, 2, 3]
new_list[:-1]
even = [n for end in [[]] for n in np.arange(1000) if (False if end or n!=412 else end.append(412))

       or not end and not n%2]

even