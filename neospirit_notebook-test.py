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
import math


a = float(input())

print(round(math.log10(a),6))
a = input()

character = []
counter = []
i = 0
digit = 0

for number in range(len(a)):
    if number == 0:
        character.append(a[number])
        i+=1
        counter.append(i)
        
    elif a[number] != a[number - 1]:
        character.append(a[number])
        i = 1
        digit+=1
        counter.append(i)
        
    elif a[number] == a[number - 1]:
        i+=1
        counter[digit] = i
        

for i in range(len(character)):
    print(character[i],counter[i],end=" ")
loop = int(input())
zig = []
zag = []

for i in range(loop):
    
    X, Y = input().split()
    if i%2 == 0:
        zig.append(int(X))
        zag.append(int(Y))
    else:
        zig.append(int(Y))
        zag.append(int(X))
        
rule = input()

if rule == "Zig-Zag":
    print(min(zig),max(zag))

else:
    print(min(zag),max(zig))
floor = int(input())
star = "*"
space = " "

for stair in range(floor):
    if stair == 0:
        print(space*(floor - stair - 1) + star + space*(floor - stair - 1))
        
    elif stair == (floor - 1):
        print(star*(2*floor-1))
        
    else:
        print(space*(floor - stair - 1) + star + ((stair - 1)*2 + 1)*space + star + (space*(floor - stair -1)))
        

question = input()
answer = input()

counter = 0

if len(question) != len(answer):
    print("Incomplete answer")
    
else:
    for digit in range(len(question)):
        if question[digit] == answer[digit]:
            counter += 1
    print(counter)
data = input()

checking = ["(","[",")","]"]
changing = ["[","(","]",")"]

new_string = ""

for position in data:
    if position == checking[0]:
        new_string += changing[0]
        
    elif position == checking[1]:
        new_string += changing[1]
        
    elif position == checking[2]:
        new_string += changing[2]
        
    elif position == checking[3]:
        new_string += changing[3]
        
    else:
        new_string += position
        
print(new_string)
expect = input()

data = input().strip('",().')

data = data.split()

count = 0

for counter in data:
    if expect == counter.strip('",().'):
        count += 1
        
print(count)
data = []

while(True):
    value = input()
    if value == "q":
        break
    data.append(float(value))
    
if data == []:
    print("No Data")
    
else:
    print(round(sum(data)/len(data),2))
a = input()
list_big = [["+","0"],
            ["1"],
            ["A","B","C","2"],
            ["D","E","F","3"],
            ["G","H","I","4"],
            ["J","K","L","5"],
            ["M","N","O","6"],
            ["P","Q","R","S","7"],
            ["T","U","V","8"],
            ["W","X","Y","Z","9"]]

data = input()
new_num = ""

for digit in data:
    for num in list_big:
        if digit in num:
            new_num+=str(list_big.index(num))
            
print(new_num)
o2 = ["A","B","C",'2']
o3 = ["D","E","F",'3']
o4 = ["G","H","I",'4']
o5 = ["J","K","L",'5']
o6 = ["M","N","O",'6']
o7 = ["P","Q","R","S",'7']
o8 = ["T","U","V",'8']
o9 = ["W","X","Y","Z",'9']

o = [o2,o3,o4,o5,o6,o7,o8,o9]
k = ""
i = input()

for a in i:
    for t in o:
        if a in t:
            k += t[-1]
            break

print(k)

o2 = ["A","B","C",'2']
o3 = ["D","E","F",'3']
o4 = ["G","H","I",'4']
o5 = ["J","K","L",'5']
o6 = ["M","N","O",'6']
o7 = ["P","Q","R","S",'7']
o8 = ["T","U","V",'8']
o9 = ["W","X","Y","Z",'9']

o = [o2,o3,o4,o5,o6,o7,o8,o9]
k = ""
i = input()
k = i[:2]
i = i[2:]
for a in i:
    for t in o:
        if a in t:
            k += t[-1]
            

print(k)
sample_list = ['a','b','c','d','e','f']

sample_list.reverse()

print(sample_list)
name1, grade1, gpax1 = input().split()
name2, grade2, gpax2 = input().split()
name3, grade3, gpax3 = input().split()


if grade1 <= grade2:
    if grade1 <= grade3:
        if grade1 < grade3:
            print(name1)
        else:
            if float(gpax1)>float(gpax3):
                print(name1)
            else:
                print(name3)
        
    elif grade1==grade2:
        if float(gpax1)>float(gpax2):
            print(name1)
        else:
            print(name2) 
            
elif grade2 <= grade3:
    if grade2 < grade3:
        print(name2)
        
        
    elif grade2==grade3:
        if float(gpax2)>float(gpax3):
            print(name2)
        else:
            print(name3) 
            
else:
    print(name3)
if "A" < "B":
    print("t")
def sum(x):
    global result
    if x == 1:
        print(result+1)
        
    else:
        result+=x
        sum(x-1)
        
result = 0
sum(5)
class cat:
    def __init__(self):
        self.leg = 0
        

print(cat.leg)
class DynaParams:
    def __init__(self):
        # discount
        self.gamma = 0.95

        # probability for exploration
        self.epsilon = 0.1

        # step size
        self.alpha = 0.1

        # weight for elapsed time
        self.time_weight = 0

        # n-step planning
        self.planning_steps = 5

        # average over several independent runs
        self.runs = 10

        # algorithm names
        self.methods = ['Dyna-Q', 'Dyna-Q+']

        # threshold for priority queue
        self.theta = 0
        
print(DynaParams.theta[1])
# Prog-06: 8-Puzzle
# Fill in your ID & Name
# ...
# Declare that you do this by yourself

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
        proto_succssor = alter_proto.copy()
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
        previous_action = previous_action + move
        successors += final_element
        successors.append(previous_action)
    
            
    return successors 
#------------------------------------------
def print_moves(board, moves):
    # bonus function: optional


    return
#------------------------------------------
board = [4,1,3,2,5,6,7,8,0]
s = gen_successors(board + ['UDR'])
print_successors(s)
moves = bfs(board)
print(moves)
print_moves(board, moves) # optional bonus
a = [0,1,2]

b= []

b.append(a[0:2])

print(b)