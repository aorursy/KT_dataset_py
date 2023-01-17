# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
class LifeGrid:

    

    #initial constant DEAD CELL and LIVE CELL

    DEAD_CELL=0

    LIVE_CELL=1

    

    def __init__(self,rows,cols):

        #initial nrow and ncol

        self._rows=rows

        self._cols=cols

        #initial grid size nrow*ncol as None

        self._grid=[[0]*cols for i in range(rows)]

        #set all element in grid as DEAD CELL

        #self.configure(list())

        

    #method that return nrow and ncol

    def numRows(self):

        return self._rows

    def numCols(self):

        return self._cols

    

    #configure status of cell from live cell in coordList

    def configure(self,coordList):

        #set status of all cell to dead

        for i in range(self.numRows()):

            for j in range(self.numCols()):

                self.clearCell(i,j) 

        #set status of specific cell as LIVE CELL

        for coord in coordList:

            self.setCell(coord[0],coord[1])

        

    #method that check whether it's live cell

    def isLiveCell(self,row,col):

        return self[row][col]==self.LIVE_CELL

    

    #method that clear that cell to DEAD CELL

    def clearCell(self,row,col):

        self[row][col]=self.DEAD_CELL

        

    #method that set that cell to LIVE CELL

    def setCell(self,row,col):

        self[row][col]=self.LIVE_CELL

        

    #method that check live cell around it

    def numLiveNeighbors(self,row,col):

        #this line is same as

        #   ans=0

        #   for i in range(max(0,row-1),min(row+1,self.numRows()-1)+1):

        #       for j in range(max(0,col-1),min(col+1,self.numCols()-1)+1):

        #           if self.isLiveCell(i,j):

        #               ans+=1

        #why must use range(max(0,row-1),min(row+1,self.numRows()-1)+1)?

        #   ans -> in normal state will check as

        #           i-1 i i+1

        #           ?   ?   ?

        #           ?   *   ?

        #           ?   ?   ?

        #           if row want to find is 0

        #           -1  0 +1

        #            ?  ?  ?

        #            ?  *  ?

        #            ?  ?  ?

        #           so should start from 0

        #           if row want to find is nrow-1

        #           nrow-2  nrow-1  norw

        #           ?       ?       ?

        #           ?       *       ?

        #           ?       ?       ?

        #           so shouhld end at nrow-1

        #       col is same as row

        ans = sum([1 for j in range(max(0,col-1),min(col+1,self.numCols()-1)+1) \

                for i in range(max(0,row-1),min(row+1,self.numRows()-1)+1) if \

                self.isLiveCell(i,j)])

        #return ans-1 if itself is LIVE CELL else return ans

        return (ans-1 if self.isLiveCell(row,col) else ans)

        

    #special method that can access row directly, dont need self._grid anymore

    def __getitem__(self,row):

        return self._grid[row]
from time import time

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



#initialize start live cell as tuple

INIT_CONFIG=[]



#initialize width and height of game ( can change to input later )

GRID_WIDTH=4

GRID_HEIGHT=4



#initialize generation of game ( can change to input later )

NUM_GEN=10



#function that evolve grid for one turn

def evolve(grid):

    #initialize liveCells as empty list

    liveCells=list()

    #access every cells in grid

    for i in range(grid.numRows()):

        for j in range(grid.numCols()):

            #count neighbors that is LIVE CELL around (i,j)

            neighbors=grid.numLiveNeighbors(i,j)

            #if neighbors is 2 and (i,j) has live or neighbors is 3

            if (neighbors==2 and grid.isLiveCell(i,j)) or neighbors==3:

                #this cell will live in next turn

                liveCells.append((i,j))

    #configure grid with new live Cell

    grid.configure(liveCells)

    

#run the following lines if this is main file only

if __name__=="__main__":

    n=[10,100,1000,10000]

    ans=[]

    for i in n:

        print("FOR",i,"CELLS , 10%")

        INIT_CONFIG=[(np.random.randint(0,i),np.random.randint(0,i)) for j in range(int(i*i*0.1))]

        #initialize grid with HEIGHT AND WIDTH

        grid=LifeGrid(i,i)

        #configure grid with initial live cell

        grid.configure(INIT_CONFIG)

        #draw grid before evolve

        #draw(grid)

        start=time()

        #describe how to do in each gen

        for i in range(NUM_GEN):

            #evolve grid

            evolve(grid)

            #draw grid

            #draw(grid)

            print(f"GEN:{i}")

        endd=time()

        ans.append(endd-start)

        print(endd-start,"Second")

    sns_plot=sns.lineplot(n,ans)

    sns_plot.set_title("Algo1 with 10%")

    sns_plot.set(xlabel="number of height and width",ylabel="time(sec)")

    sns_plot.figure.savefig("10percent"+".png")