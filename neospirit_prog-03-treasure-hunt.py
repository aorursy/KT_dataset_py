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
# Prog-03: Treasure Hunt
# 6330011321 นายกฤติพงศ์ มานะชำนิ
# โปรแกรมนี้ดผมคิดวิธีการเองจากความรู้ของผม
# Special task ผมเป็นคนตั้งโจทย์ขึ้นและคิดวิธีของผมเอง โจทย์คืออยากรู้ว่ามีช่องที่ถูกเหยียบใน gridworld แล้วกี่ช่อง
# คุณสามารถสำรวจโค้ดของผมเพิ่มเติมได้ที่นี่ https://www.kaggle.com/neospirit/prog-03-treasure-hunt/

N = 10    # size of world N * N
world = [[0]*N for i in range(N)]

# bottomleft is (0,0), x goes right, y goes up
def printworld():
    for i in range(N-1,-1,-1):
        for j in range(N):
            print(world[j][i], end = " ")
        print()
     

hd = 0   # heading 0-N, 1-E, 2-S, 3-W
px = 0   # current position x
py = 0   # current position y

# move forward one step
def forward():
    global hd, px, py
    # move one step
    if(hd == 0):
        py += 1
    elif(hd == 1):
        px += 1
    elif(hd == 2):
        py -= 1
    elif(hd == 3):
        px -= 1
    # constrain x,y in bound 0..N-1
    if(px > N-1):
        px = N-1
    if(px < 0):
        px = 0
    if(py > N-1):
        py = N-1
    if(py < 0):
        py = 0
    world[px][py] = 1

# turn head left 90 degree
def turnleft():
    global hd
    hd -= 1
    if(hd < 0):
        hd = 3

# turn head right 90 degree
def turnright():
    global hd
    hd = (hd + 1) % 4

# make move according to m (map)
def makemove(m):
    for c in m:
        if(c == "F"):
            forward()
        elif(c == "L"):
            turnleft()
        elif(c == "R"):
            turnright()

def origin(x,y):
    global px,py
    px = x
    py = y
    world[px][py] = 1
    
def test():
    origin(1,1)
    mymap = "FFRFFLFFF"
    makemove(mymap)
    printworld()
    
def joinmap1(s1,s2):
    s = s1 + s2
    return s

def joinmap2(s1,s2):
    divide_s1 = (int((len(s1)/2)) + len(s1)%2)
    divide_s2 = (int((len(s2)/2)) + len(s2)%2)

    s = s1[0:divide_s1] + s2[0:divide_s2] + s1[divide_s1:len(s1)] +s2[divide_s2:len(s2)]
    return s

def joinmap3(s1,s2):
    s = ''
    if len(s1) < len(s2):
        for i in range(len(s1)):
            s += (s1[i] + s2[i])
        s += s2[len(s1):len(s2)]
        
    else:
        for i in range(len(s2)):
            s += (s1[i] + s2[i])
        s += s2[len(s1):len(s2)]
        
    return s

# This function will show how many position you have step in the world.
def print_sum_footprint(): 
    result = 0
    for column in world:
        result += sum(column)
    print(result)
    
    return result               #I return value to if you want to use or save in other variable

def main():
    # test()
    origin(1,1)
    s1 = "FFFR"
    s2 = "FFLFLF"
    print("Task 1")
    s = joinmap1(s1,s2)
    makemove(s)
    printworld()
    
    print("Task 2")
    s = joinmap2(s1,s2)
    makemove(s)
    printworld()
    
    print("Task 3")
    s = joinmap3(s1,s2)
    makemove(s)
    printworld()
    
    print("Special Task (created by myself)")
    s = joinmap3(s1,s2)                     # Using method in condition 3 for example
    makemove(s)
    printworld()
    print_sum_footprint()
    
    
main()


