import numpy as np

import pandas as pd



import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np

from pathlib import Path



for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'

training_tasks = sorted(os.listdir(training_path))

eval_tasks = sorted(os.listdir(evaluation_path))
T = training_tasks

Trains = []

for i in range(400):

    task_file = str(training_path / T[i])

    task = json.load(open(task_file, 'r'))

    Trains.append(task)

    

E = eval_tasks

Evals= []

for i in range(400):

    task_file = str(evaluation_path / E[i])

    task = json.load(open(task_file, 'r'))

    Evals.append(task)
cmap = colors.ListedColormap(

    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)

# 0:black, 1:blue, 2:red, 3:greed, 4:yellow,

# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown

plt.figure(figsize=(5, 2), dpi=200)

plt.imshow([list(range(10))], cmap=cmap, norm=norm)

plt.xticks(list(range(10)))

plt.yticks([])

plt.show()



def plot_task(task):

    n = len(task["train"]) + len(task["test"])

    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig_num = 0

    for i, t in enumerate(task["train"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Train-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Train-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    for i, t in enumerate(task["test"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Test-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Test-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    

    plt.tight_layout()

    plt.show()
def plot_picture(x):

    plt.imshow(np.array(x), cmap = cmap, norm = norm)

    plt.show()
#Transformations

def Vert(M):

    n = len(M)

    k = len(M[0])

    ans = np.zeros((n,k), dtype = int)

    for i in range(n):

        for j in range(k):

            ans[i][j] = 0+M[n-1-i][j]

    return ans.tolist()



def Hor(M):

    n = len(M)

    k = len(M[0])

    ans = np.zeros((n,k), dtype = int)

    for i in range(n):

        for j in range(k):

            ans[i][j] = 0+M[i][k-1-j]

    return ans.tolist()



def Rot1(M):

    n = len(M)

    k = len(M[0])

    ans = np.zeros((k,n), dtype = int)

    for i in range(n):

        for j in range(k):

            ans[j][i] = 0 + M[i][k-1-j]

    return ans.tolist()

            

def Rot2(M):

    n = len(M)

    k = len(M[0])

    ans = np.zeros((k,n), dtype = int)

    for i in range(n):

        for j in range(k):

            ans[j][i] = 0 + M[n-1-i][j]

    return ans.tolist()

Geometric = [[Hor, Hor], [Rot2], [Rot1, Rot1], [Rot1], [Vert], [Hor, Rot2], [Hor], [Vert, Rot2]]
def Defensive_Copy(A): 

    n = len(A)

    k = len(A[0])

    L = np.zeros((n,k), dtype = int)

    for i in range(n):

        for j in range(k):

            L[i,j] = 0 + A[i][j]

    return L.tolist()
def Apply(S, x):

    if S in Geometric:

        x1 = Defensive_Copy(x)

        for t in S:

            x1 = t(x1)

    return x1
def Cut(M, r1, r2): #Cut a region into tiles

    List = []

    n = len(M)

    n1 = n//r1

    k = len(M[0])

    k1 = k//r2

    for i in range(r1):

        for j in range(r2):

            R = np.zeros((n1,k1), dtype = int)

            for t1 in range(n1):

                for t2 in range(k1):

                    R[t1,t2] = 0+M[i*n1+t1][j*k1+t2]

            List.append(R.tolist())

    return List
def Glue(List, r1, r2): #Combine tiles to one picture

    n1 = len(List[0])

    k1 = len(List[0][0])

    ans = np.zeros((n1*r1, k1*r2), dtype = int)

    counter = 0

    for i in range(r1):

        for j in range(r2):

            R = List[counter]

            counter +=1

            for t1 in range(n1):

                for t2 in range(k1):

                    ans[i*n1+t1, j*k1+t2] = 0 + R[t1][t2]

    return ans.tolist()
A = np.array([[0,0,0,0,0], [0,1,2,3,0], [0,4,5,6,0], [0,7,8,9,0], [0,0,0,0,0]])

plot_picture(A)
List = [Apply(S,A) for S in Geometric]

B = Glue(List, 2, 4)

plot_picture(B)
List2 = Cut(B, 2, 1)

for x in List2:

    plot_picture(x)
def Match(basic_task): 

    #returns -1 if no match is found

    #returns  Transformed_Test_Case  if the mathching rule is found

    Input = [Defensive_Copy(x) for x in basic_task[0]]

    Output = [Defensive_Copy(y) for y in basic_task[1]]

    Test_Case = Input[-1]

    Input = Input[:-1]

    for i in range(len(Geometric)):

        S = Geometric[i]

        solved = True

        for x, y in zip(Input,Output):

            transformed_x = Apply(S,x)

            if transformed_x != y:

                solved = False

                break

        if solved == True:

            Transformed_Test_Case = Apply(S, Test_Case)

            return Transformed_Test_Case

    return -1
def Solve(basic_task): 

    # returns -1 if no match is found

    # returns Transformed_Test_Case  if the mathching rule is found

    # for this notebook we only look at mosaics

    Input = [Defensive_Copy(x) for x in basic_task[0]]

    Output = [Defensive_Copy(y) for y in basic_task[1]]

    same_ratio = True

    R_x = []

    R_y = []

    for x, y in zip(Input[:-1],Output):

        n1 = len(x)

        n2 = len(y)

        k1 = len(x[0])

        k2 = len(y[0])

        if n2%n1 != 0 or k2%k1 != 0:

            same_ratio = False

            break

        else :

            R_y.append(n2//n1)

            R_x.append(k2//k1)

       

    if same_ratio and min(R_x) == max(R_x) and min(R_y) == max(R_y): 

        r1 = min(R_y)

        r2 = min(R_x)

        Fractured_Output = [Cut(x, r1, r2) for x in Output]

        

        Partial_Solutions = []

        for panel in range(r1*r2):

            List = [Fractured_Output[i][panel] for i in range(len(Output))]

            

            proposed_solution = Match([Input, List])

            

            if proposed_solution == -1:

                return -1

            else: 

                Partial_Solutions.append(proposed_solution)

        Transformed_Test_Case = Glue(Partial_Solutions, r1, r2)

        return Transformed_Test_Case    

        

    return -1
def Create(task, task_id = 0):

    n = len(task['train'])

    Input = [Defensive_Copy(task['train'][i]['input']) for i in range(n)]

    Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]

    Input.append(Defensive_Copy(task['test'][task_id]['input']))

    return Input, Output

    
solved_train = []

for i in range(400):

    task = Trains[i]

    k = len(task['test'])

    for j in range(k):

        basic_task = Create(task, j)

        proposed_solution = Solve(basic_task)

        if proposed_solution != -1:

            print(i,j)

            solved_train.append((i,j))

            plot_task(task)

            plot_picture(proposed_solution)
print(len(solved_train))

for x in solved_train:

    print(x)
solved_eval = []

for i in range(400):

    task = Evals[i]

    k = len(task['test'])

    for j in range(k):

        basic_task = Create(task, j)

        proposed_solution = Solve(basic_task)

        if proposed_solution != -1:

            print(i,j)

            solved_eval.append((i,j))

            plot_task(task)

            plot_picture(proposed_solution)
print(len(solved_eval))

for x in solved_eval:

    print(x)