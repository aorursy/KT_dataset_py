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
# Prog-06: 8-Puzzle
# 6330011321 : กฤติพงศ์ มานะชำนิ
# ...
# ข้าพเจ้า นายกฤติพงศ์ มานะชำนิ เขียนโค้ดนี้ด้วยตัวเอง

def is_goal(node): 
    return node[:-1] == \
           list(range(1,len(node)-1))+[0]

def insert_all(node, fringe):
    n = len(node)
    children = gen_successors(node)
    for i in range(0,len(children),n):
        fringe.append(children[i:i+n])

def bfs(board):
    

    start_node = board + ['']
    fringe = [start_node]
    while True:
        if len(fringe) == 0:
            break
        front = fringe[0]
        fringe = fringe[1:]
        if is_goal(front):
            return front[-1]
        insert_all(front,fringe) 
    
    return ''

def print_successors(s):
    N = 1
    for e in s:
        if type(e) is str: break
        N += 1
    for i in range(0,len(s),N):
        print(s[i:i+N])
#---------------------------------------
def gen_successors(node):
    
    moves = ["U","L","R","D"]
    successors = []
    new_node = node[:-1]
    size = int(len(new_node)**0.5)
    previous_action = node[-1]
    position = new_node.index(0)
    alter_proto = []
    zero_position = [position//size, position%size]
    
    # Create 2d list
    for counter in range(size):
        alter_proto.append(new_node[counter*size:counter*size + size])
    
    # Upside
    if zero_position[0] == 0:
        moves.remove("U")
    
    # Downside
    if zero_position[0] == size - 1:
        moves.remove("D")
    
    #Leftside
    if zero_position[1] == 0:
        moves.remove("L")
    
    #Rightside
    if zero_position[1] == size - 1:
        moves.remove("R")
    

    # Action all moves
    for move in moves:
        
        # ข้างใน alter_proto มี list อยู่ ซึ่ง reference ด้วย list อีกตัว เมื่อ alter_proto.copy() จะทำให้ pointer ชี้ไปจุดเดียวกัน
        # สิ่งที่ทำคือดึงเฉพาะ element มา แล้วเอามาใส่ใน new_list แทน reference จึงจะเปลี่ยน
        proto_succssor = [x[:] for x in alter_proto]
        if move == "U":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]-1][zero_position[1]]
            proto_succssor[zero_position[0]-1][zero_position[1]] = 0
            
            
            
        elif move == "D":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]+1][zero_position[1]]
            proto_succssor[zero_position[0]+1][zero_position[1]] = 0
            
            
        elif move == "L":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]][zero_position[1]-1]
            proto_succssor[zero_position[0]][zero_position[1]-1] = 0
            
            
        elif move == "R":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]][zero_position[1]+1]
            proto_succssor[zero_position[0]][zero_position[1]+1] = 0
            
        
        final_element = []
        
        # Flatten 2D
        for element in proto_succssor:
            final_element += element
        final_action = previous_action + move
        successors += final_element
        successors.append(final_action)
    
            
    return successors 
#------------------------------------------
def print_moves(board, moves):
    
    
    
    # bonus function: optional
    dash = "-"
    size = int(len(board)**0.5)
    position = board.index(0)
    progress_board = ""
    for action in moves:
        for i in range(0,len(board),size):
            new_sequence = ""
            for index_of_size in range(size):
                new_sequence += f"{board[i + index_of_size]}  "
            progress_board += (" "+new_sequence+"\n")
            
        progress_board += (f"{dash*(size**2)} {action}\n")
        
        if action == "U":
            board[position] = board[position - size]
            board[position - size] = 0
            position = position - size
            
        elif action == "D":
            board[position] = board[position + size]
            board[position + size] = 0
            position = position + size
            
        elif action == "L":
            board[position] = board[position - 1]
            board[position - 1] = 0
            position = position - 1
            
        elif action == "R":
            board[position] = board[position + 1]
            board[position + 1] = 0
            position = position + 1
            
            
    for i in range(0,len(board),size):
        new_sequence = ""
        for index_of_size in range(size):
            new_sequence += f"{board[i + index_of_size]}  "
        progress_board += (" "+new_sequence+"\n")
        
    print(progress_board)
    

    return
#------------------------------------------
import time
start = time.time()
board = [4,1,3,2,5,6,7,8,0]
s = gen_successors(board + ['UDR'])
print_successors(s)
moves = bfs(board)
print(moves)
print_moves(board, moves) # optional bonus
end = time.time()
print("Time used",end - start)
# Prog-06: 8-Puzzle
# 6330011321 : กฤติพงศ์ มานะชำนิ
# ...
# ข้าพเจ้า นายกฤติพงศ์ มานะชำนิ เขียนโค้ดนี้ด้วยตัวเอง

def is_goal(node): 
    return node[:-1] == \
           list(range(1,len(node)-1))+[0]

def insert_all(node, fringe):
    n = len(node)
    children = gen_successors(node)
    for i in range(0,len(children),n):
        fringe.append(children[i:i+n])

def bfs(board):
    start_node = board + ['']
    fringe = [start_node]
    while True:
        if len(fringe) == 0:
            break
        front = fringe[0]
        fringe = fringe[1:]
        if is_goal(front):
            return front[-1]
        insert_all(front,fringe) 
    return ''

def print_successors(s):
    N = 1
    for e in s:
        if type(e) is str: break
        N += 1
    for i in range(0,len(s),N):
        print(s[i:i+N])
#---------------------------------------
def gen_successors(node):
    
    moves = ["U","L","R","D"]
    successors = []
    new_node = node[:-1]
    size = int(len(new_node)**0.5)
    previous_action = node[-1]
    position = new_node.index(0)
    alter_proto = []
    zero_position = [position//size, position%size]
    
    # Create 2d list
    for counter in range(size):
        alter_proto.append(new_node[counter*size:counter*size + size])
    
    # Upside
    if zero_position[0] == 0:
        moves.remove("U")
    
    # Downside
    if zero_position[0] == size - 1:
        moves.remove("D")
    
    #Leftside
    if zero_position[1] == 0:
        moves.remove("L")
    
    #Rightside
    if zero_position[1] == size - 1:
        moves.remove("R")
    

    # Action all moves
    for move in moves:
        
        # ข้างใน alter_proto มี list อยู่ ซึ่ง reference ด้วย list อีกตัว เมื่อ alter_proto.copy() จะทำให้ pointer ชี้ไปจุดเดียวกัน
        # สิ่งที่ทำคือดึงเฉพาะ element มา แล้วเอามาใส่ใน new_list แทน reference จึงจะเปลี่ยน
        proto_succssor = [x[:] for x in alter_proto]
        if move == "U":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]-1][zero_position[1]]
            proto_succssor[zero_position[0]-1][zero_position[1]] = 0
            
            
            
        elif move == "D":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]+1][zero_position[1]]
            proto_succssor[zero_position[0]+1][zero_position[1]] = 0
            
            
        elif move == "L":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]][zero_position[1]-1]
            proto_succssor[zero_position[0]][zero_position[1]-1] = 0
            
            
        elif move == "R":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]][zero_position[1]+1]
            proto_succssor[zero_position[0]][zero_position[1]+1] = 0
            
        
        final_element = []
        
        # Flatten 2D
        for element in proto_succssor:
            final_element += element
        final_action = previous_action + move
        successors += final_element
        successors.append(final_action)
    
            
    return successors 
#------------------------------------------
def print_moves(board, moves):
    # bonus function: optional
    dash = "-"
    size = int(len(board)**0.5)
    position = board.index(0)
    progress_board = ""
    for action in moves:
        for i in range(0,len(board),size):
            new_sequence = ""
            for index_of_size in range(size):
                new_sequence += f"{board[i + index_of_size]}  "
            progress_board += (" "+new_sequence+"\n")
            
        progress_board += (f"{dash*(size**2)} {action}\n")
        
        if action == "U":
            board[position] = board[position - size]
            board[position - size] = 0
            position = position - size
            
        elif action == "D":
            board[position] = board[position + size]
            board[position + size] = 0
            position = position + size
            
        elif action == "L":
            board[position] = board[position - 1]
            board[position - 1] = 0
            position = position - 1
            
        elif action == "R":
            board[position] = board[position + 1]
            board[position + 1] = 0
            position = position + 1
            
            
    for i in range(0,len(board),size):
        new_sequence = ""
        for index_of_size in range(size):
            new_sequence += f"{board[i + index_of_size]}  "
        progress_board += (" "+new_sequence+"\n")
        
    print(progress_board)

    return
#------------------------------------------
import time
start = time.time()
board = [ 1, 2, 3, 4, 5, 0, 7, 8, 9, 6, 10, 12, 13, 14, 11, 15]
s = gen_successors(board + ['UDR'])
print_successors(s)
moves = bfs(board)
print(moves)
print_moves(board, moves) # optional bonus
end = time.time()
print("Time used",end - start)
# Prog-06: 8-Puzzle
# 6330011321 : กฤติพงศ์ มานะชำนิ
# ...
# ข้าพเจ้า นายกฤติพงศ์ มานะชำนิ เขียนโค้ดนี้ด้วยตัวเอง

def is_goal(node): 
    return node[:-1] == \
           list(range(1,len(node)-1))+[0]

def insert_all(node, fringe):
    n = len(node)
    children = gen_successors(node)
    for i in range(0,len(children),n):
        fringe.append(children[i:i+n])

def bfs(board):
    start_node = board + ['']
    fringe = [start_node]
    while True:
        if len(fringe) == 0:
            break
        front = fringe[0]
        fringe = fringe[1:]
        if is_goal(front):
            return front[-1]
        insert_all(front,fringe) 
    return ''

def print_successors(s):
    N = 1
    for e in s:
        if type(e) is str: break
        N += 1
    for i in range(0,len(s),N):
        print(s[i:i+N])
#---------------------------------------
def gen_successors(node):
    
    moves = ["U","L","R","D"]
    successors = []
    new_node = node[:-1]
    size = int(len(new_node)**0.5)
    previous_action = node[-1]
    position = new_node.index(0)
    alter_proto = []
    zero_position = [position//size, position%size]
    
    # Create 2d list
    for counter in range(size):
        alter_proto.append(new_node[counter*size:counter*size + size])
    
    # Upside
    if zero_position[0] == 0:
        moves.remove("U")
    
    # Downside
    if zero_position[0] == size - 1:
        moves.remove("D")
    
    #Leftside
    if zero_position[1] == 0:
        moves.remove("L")
    
    #Rightside
    if zero_position[1] == size - 1:
        moves.remove("R")
    

    # Action all moves
    for move in moves:
        
        # ข้างใน alter_proto มี list อยู่ ซึ่ง reference ด้วย list อีกตัว เมื่อ alter_proto.copy() จะทำให้ pointer ชี้ไปจุดเดียวกัน
        # สิ่งที่ทำคือดึงเฉพาะ element มา แล้วเอามาใส่ใน new_list แทน reference จึงจะเปลี่ยน
        proto_succssor = [x[:] for x in alter_proto]
        if move == "U":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]-1][zero_position[1]]
            proto_succssor[zero_position[0]-1][zero_position[1]] = 0
            
            
            
        elif move == "D":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]+1][zero_position[1]]
            proto_succssor[zero_position[0]+1][zero_position[1]] = 0
            
            
        elif move == "L":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]][zero_position[1]-1]
            proto_succssor[zero_position[0]][zero_position[1]-1] = 0
            
            
        elif move == "R":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]][zero_position[1]+1]
            proto_succssor[zero_position[0]][zero_position[1]+1] = 0
            
        
        final_element = []
        
        # Flatten 2D
        for element in proto_succssor:
            final_element += element
        final_action = previous_action + move
        successors += final_element
        successors.append(final_action)
    
            
    return successors 
#------------------------------------------
def print_moves(board, moves):
    # bonus function: optional
    dash = "-"
    size = int(len(board)**0.5)
    position = board.index(0)
    progress_board = ""
    for action in moves:
        for i in range(0,len(board),size):
            new_sequence = ""
            for index_of_size in range(size):
                new_sequence += f"{board[i + index_of_size]}  "
            progress_board += (" "+new_sequence+"\n")
            
        progress_board += (f"{dash*(size**2)} {action}\n")
        
        if action == "U":
            board[position] = board[position - size]
            board[position - size] = 0
            position = position - size
            
        elif action == "D":
            board[position] = board[position + size]
            board[position + size] = 0
            position = position + size
            
        elif action == "L":
            board[position] = board[position - 1]
            board[position - 1] = 0
            position = position - 1
            
        elif action == "R":
            board[position] = board[position + 1]
            board[position + 1] = 0
            position = position + 1
            
            
    for i in range(0,len(board),size):
        new_sequence = ""
        for index_of_size in range(size):
            new_sequence += f"{board[i + index_of_size]}  "
        progress_board += (" "+new_sequence+"\n")
        
    print(progress_board)

    return
#------------------------------------------
import time
start = time.time()
board = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 16, 17, 18, 19, 15, 21, 22, 23, 24, 20]
s = gen_successors(board + ['UDR'])
print_successors(s)
moves = bfs(board)
print(moves)
print_moves(board, moves) # optional bonus
end = time.time()
print("Time used",end - start)
# Prog-06: 8-Puzzle
# 6330011321 : กฤติพงศ์ มานะชำนิ
# ...
# ข้าพเจ้า นายกฤติพงศ์ มานะชำนิ เขียนโค้ดนี้ด้วยตัวเอง

def is_goal(node): 
    return node[:-1] == \
           list(range(1,len(node)-1))+[0]

def insert_all(node, fringe):
    n = len(node)
    children = gen_successors(node)
    for i in range(0,len(children),n):
        fringe.append(children[i:i+n])

def bfs(board):
    start_node = board + ['']
    fringe = [start_node]
    while True:
        if len(fringe) == 0:
            break
        front = fringe[0]
        fringe = fringe[1:]
        if is_goal(front):
            return front[-1]
        insert_all(front,fringe) 
    return ''

def print_successors(s):
    N = 1
    for e in s:
        if type(e) is str: break
        N += 1
    for i in range(0,len(s),N):
        print(s[i:i+N])
#---------------------------------------
def gen_successors(node):
    
    moves = ["U","L","R","D"]
    successors = []
    new_node = node[:-1]
    size = int(len(new_node)**0.5)
    previous_action = node[-1]
    position = new_node.index(0)
    alter_proto = []
    zero_position = [position//size, position%size]
    
    # Create 2d list
    for counter in range(size):
        alter_proto.append(new_node[counter*size:counter*size + size])
    
    # Upside
    if zero_position[0] == 0:
        moves.remove("U")
    
    # Downside
    if zero_position[0] == size - 1:
        moves.remove("D")
    
    #Leftside
    if zero_position[1] == 0:
        moves.remove("L")
    
    #Rightside
    if zero_position[1] == size - 1:
        moves.remove("R")
    

    # Action all moves
    for move in moves:
        
        # ข้างใน alter_proto มี list อยู่ ซึ่ง reference ด้วย list อีกตัว เมื่อ alter_proto.copy() จะทำให้ pointer ชี้ไปจุดเดียวกัน
        # สิ่งที่ทำคือดึงเฉพาะ element มา แล้วเอามาใส่ใน new_list แทน reference จึงจะเปลี่ยน
        proto_succssor = [x[:] for x in alter_proto]
        if move == "U":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]-1][zero_position[1]]
            proto_succssor[zero_position[0]-1][zero_position[1]] = 0
            
            
            
        elif move == "D":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]+1][zero_position[1]]
            proto_succssor[zero_position[0]+1][zero_position[1]] = 0
            
            
        elif move == "L":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]][zero_position[1]-1]
            proto_succssor[zero_position[0]][zero_position[1]-1] = 0
            
            
        elif move == "R":
            proto_succssor[zero_position[0]][zero_position[1]] = proto_succssor[zero_position[0]][zero_position[1]+1]
            proto_succssor[zero_position[0]][zero_position[1]+1] = 0
            
        
        final_element = []
        
        # Flatten 2D
        for element in proto_succssor:
            final_element += element
        final_action = previous_action + move
        successors += final_element
        successors.append(final_action)
    
            
    return successors 
#------------------------------------------
def print_moves(board, moves):
    # bonus function: optional
    dash = "-"
    size = int(len(board)**0.5)
    position = board.index(0)
    progress_board = ""
    for action in moves:
        for i in range(0,len(board),size):
            new_sequence = ""
            for index_of_size in range(size):
                new_sequence += f"{board[i + index_of_size]}  "
            progress_board += (" "+new_sequence+"\n")
            
        progress_board += (f"{dash*(size**2)} {action}\n")
        
        if action == "U":
            board[position] = board[position - size]
            board[position - size] = 0
            position = position - size
            
        elif action == "D":
            board[position] = board[position + size]
            board[position + size] = 0
            position = position + size
            
        elif action == "L":
            board[position] = board[position - 1]
            board[position - 1] = 0
            position = position - 1
            
        elif action == "R":
            board[position] = board[position + 1]
            board[position + 1] = 0
            position = position + 1
            
            
    for i in range(0,len(board),size):
        new_sequence = ""
        for index_of_size in range(size):
            new_sequence += f"{board[i + index_of_size]}  "
        progress_board += (" "+new_sequence+"\n")
        
    print(progress_board)

    return
#------------------------------------------
board = [1,0,3,4,2,5,7,8,6]
s = gen_successors(board + ['UDR'])
print_successors(s)
moves = bfs(board)
print(moves)
print_moves(board, moves) # optional bonus
board = [[1,0,3,4,2,5,7,8,6]]
board = board[1:]
print(board)