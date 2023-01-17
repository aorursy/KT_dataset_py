"""
The program runs in 4x4 , 9x9 , 16x16 , 25x25 , etc Sudoku. 
It cannot works in 12x12 sudoku. 
Because it controls square not rectangle 
@aliosmanbilgin
"""

from ortools.constraint_solver import pywrapcp
import pandas as pd
import math
import numpy as np

#reading csv file in python
Sudoku = pd.read_csv("../input/sudoku/sudokuempty.csv", delimiter=";", header=None)
Sudoku = Sudoku.fillna(0)
Sudoku = Sudoku.astype(int)

for a in range(0,1000):
    solver= pywrapcp.Solver('Sudoku');
    
    #Calculating Sudoku's length and its sqrt
    n=len(Sudoku)
    m=int(math.sqrt(n))

    Sudoku_orj =Sudoku.copy()
    Sudoku.iloc[np.random.randint(0,n),np.random.randint(0,n)]=np.random.randint(0,n)
    
    #creating solution matrix
    SolutionValues =[] #it can be create a matrix but i prefer an array

    #creating Solution Values...
    for i in range(0,n):
        for j in range(0,n):
            x=int(Sudoku.iloc[i,j])
            if(x==0):
                si=solver.IntVar(1,n,'S{}{}'.format(i,j))
            else:
                si=solver.IntVar(x,x,'S{}{}'.format(i,j))
            SolutionValues.append(si)

    #calculating expressions
    for i in range(0,n):    

        #adding to solver row & call expressions 
        row_exp=solver.AllDifferent(SolutionValues[i*n:(i+1)*n])
        coll_exp=solver.AllDifferent(SolutionValues[slice(i,n*n,n)])

        #adding square expressions
        #            row s.p    call s.p
        start_point= i//m*n*m + i%m*m
        SquareExpression=[]
        for j in range(0,m):
            SquareExpression+=SolutionValues[start_point+j*n:start_point+j*n+m]
        sqr_exp=solver.AllDifferent(SquareExpression)

        solver.Add(row_exp)
        solver.Add(coll_exp)
        solver.Add(sqr_exp)
        if not solver.CheckConstraint(row_exp) & solver.CheckConstraint(coll_exp) & solver.CheckConstraint(sqr_exp):
            Sudoku =Sudoku_orj
            #print("hata")

    #print("\n")
    #print(Sudoku)
    
    db = solver.Phase(SolutionValues,
                         solver.CHOOSE_FIRST_UNBOUND,
                         solver.ASSIGN_MIN_VALUE)
    solver =None
    """
    solver.NewSearch(db)
    while solver.NextSolution():

        #printing Sudoku
        for i in range(0,n):
            satir=""
            for j in range(0,n):
                satir=satir+str(SolutionValues[j+(i*n)])[-3:]
                satir=satir.replace("("," ")
                satir=satir.replace(")"," ")
            print(satir)
        print("\n")
    solver.EndSearch()
    """
print(Sudoku)
"""
The program runs in 4x4 , 9x9 , 16x16 , 25x25 , etc Sudoku. 
It cannot works in 12x12 sudoku. 
Because it controls square not rectangle 
@aliosmanbilgin
"""

from ortools.constraint_solver import pywrapcp
import pandas as pd
import math
solver= pywrapcp.Solver('Sudoku');

#reading csv file in python
Sudoku = pd.read_csv("../input/sudoku/sudoku.csv", delimiter=";", header=None)
Sudoku = Sudoku.fillna(0)
Sudoku = Sudoku.astype(int)

#creating solution matrix
SolutionValues =[] #it can be create a matrix but i prefer an array

#Calculating Sudoku's length and its sqrt
n=len(Sudoku)
m=int(math.sqrt(n))

#creating Solution Values...
for i in range(0,n):
    for j in range(0,n):
        x=int(Sudoku.iloc[i,j])
        if(x==0):
            si=solver.IntVar(1,n,'S{}{}'.format(i,j))
        else:
            si=solver.IntVar(x,x,'S{}{}'.format(i,j))
        SolutionValues.append(si)

#calculating expressions
for i in range(0,n):    
    
    #adding to solver row & call expressions 
    solver.Add(solver.AllDifferent(SolutionValues[i*n:(i+1)*n]))
    solver.Add(solver.AllDifferent(SolutionValues[slice(i,n*n,n)]))
    
    #adding square expressions
    #            row s.p    call s.p
    start_point= i//m*n*m + i%m*m
    SquareExpression=[]
    for j in range(0,m):
        SquareExpression+=SolutionValues[start_point+j*n:start_point+j*n+m]
    solver.Add(solver.AllDifferent(SquareExpression))
    
db = solver.Phase(SolutionValues,
                     solver.CHOOSE_FIRST_UNBOUND,
                     solver.ASSIGN_MIN_VALUE)

solver.NewSearch(db)
while solver.NextSolution():
    
    #printing Sudoku
    for i in range(0,n):
        satir=""
        for j in range(0,n):
            satir=satir+str(SolutionValues[j+(i*n)])[-3:]
            satir=satir.replace("("," ")
            satir=satir.replace(")"," ")
        print(satir)
    print("\n")
solver.EndSearch()