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
#อย่าหาทำ
'''
sample = [1,2,3]
print(max(sample))
'''

sample = [1,2,3]
starter = sample[0]

for i in sample:
    if i > starter:
        starter = i
        
print(starter)
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
# ถ้าอยาก input ลบตรงนี้ออกได้
'''
board = []

for i in range(int(input("Enter size of puzzle: "))):
    row = input('Enter value in row with spacing(Ex.: "1 0 4" ): ').split()
    row = list(map(int, row))
    board += row
    
'''
    
process = time.time()    
moves = bfs(board)
print(moves)
print_moves(board, moves) # optional bonus
end = time.time()
print("Time used (all progress)",end - start)
print("Time used (processing)",end - process)
import matplotlib.pyplot as plt
import networkx as nx

node = 9
edge = 6

G = nx.gnm_random_graph(node, edge, seed=None, directed=False)

G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9])
G.add_edges_from([(1,2), (3,4), (2,5), (4,5), (6,7), (8,9), (4,7), (1,7), (3,5), (2,7), (5,8), (2,9), (5,7)])
G.add_nodes_from([
    (4, {"color": "red"}),
    (5, {"color": "green"}),
])

nx.draw(G)

fig = plt.gcf()

axes = plt.gca()

axes.get_children()

text_object = axes.get_children()[2]
nodes = axes.get_children()[1] 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("../input/co2-ghg-emissionsdata/co2_emission.csv",delimiter=',')

df
df["Year"].unique()
df.info()
df.describe()
df.hist()
fig = px.scatter(df, x="Year", y="Entity", 
                 color="Year",
                 size='Year', 
                 hover_data=['Annual CO₂ emissions (tonnes )', 'Code'], 
                 title = "CO2 Emissions")
fig.show()
# Selecting latest year
emission2017 = df.loc[df['Year'] == 2017]
import altair as alt

select_year = alt.selection_single(
    name='select', fields=['year'], init={'year': 1949},
    bind=alt.binding_range(min=1949, max=2017, step=10)
)

alt.Chart(emission2017).mark_point().encode(
    alt.X('Code'),
    alt.Y('Annual CO₂ emissions (tonnes )')
).add_selection(select_year).transform_filter(select_year)