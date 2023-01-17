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
#import library name numpy and rename it to np

import numpy as np



class LifeGrid2:

    

    def __init__(self,rows,cols):

        #initial nrow and ncol

        self._rows=rows

        self._cols=cols

        #initial dictionary that collect only live cell

        self._liveDict=dict()

        #initial grid rows*cols

        self._grid=np.zeros([rows,cols])

        #set all element in grid as DEAD CELL

        self.configure(list())

        

    #method that return nrow and ncol

    def numRows(self):

        return self._rows

    def numCols(self):

        return self._cols

    

    #configure status of cell from live cell in coordList

    def configure(self,coordList):

        #set status of all cell to dead

        self._liveDict.clear()

        #set all elements in grid as 0

        self._grid=np.zeros([self.numRows(),self.numCols()])

        #set status of specific cell as LIVE CELL

        for coord in coordList:

            self.setCell(coord[0],coord[1])

            self[coord[0]][coord[1]]=1

        

    #method that check whether it's live cell

    def isLiveCell(self,row,col):

        return self[row][col]==1

    

    #method that clear that cell to DEAD CELL

    def clearCell(self,row,col):

        #set value of that cell as 0

        self._grid[row][col]=0

        #these lines below has meaning as

        #   if (row,col) in self._liveDict.keys():

        #      del self._liveDict[(row,col)]

        #but I love try-except 55555

        try:

            del self._liveDict[(row,col)]

        except:

            return ;

        

    #method that set that cell to LIVE CELL

    def setCell(self,row,col):

        self._liveDict[(row,col)]=1

        self._grid[row][col]=1

        

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

    

    #function that abbreviate self._grid[row][col] to self[row][col]

    def __getitem__(self,row):

        return self._grid[row]



class LifeGrid4:

    

    DEAD_CELL=0

    LIVE_CELL=0

    

    def __init__(self,rows,cols):

        #initial nrow and ncol

        self._rows=rows

        self._cols=cols

        #initial grid rows*cols

        self._grid=np.zeros([rows,cols])

        

    #method that return nrow and ncol

    def numRows(self):

        return self._rows

    def numCols(self):

        return self._cols

    

    #configure status of cell from live cell in coordList

    def configure(self,coordList):

        #set all elements in grid as 0

        self._grid=np.zeros([self.numRows(),self.numCols()])

        #set status of specific cell as LIVE CELL

        for coord in coordList:

            self.setCell(coord[0],coord[1])

            self[coord[0]][coord[1]]=1

        

    #method that check whether it's live cell

    def isLiveCell(self,row,col):

        return self[row][col]==1

    

    #method that clear that cell to DEAD CELL

    def clearCell(self,row,col):

        #set value of that cell as 0

        self._grid[row][col]=0

        

    #method that set that cell to LIVE CELL

    def setCell(self,row,col):

        self._grid[row][col]=1

        

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

    

    #function that abbreviate self._grid[row][col] to self[row][col]

    def __getitem__(self,row):

        return self._grid[row]
from time import time

import numpy as np

import matplotlib.pyplot as plt



#initialize generation of game ( can change to input later )

NUM_GEN=10



#function that evolve grid for one turn

def evolve2(grid):

    #initialize liveCells as empty set

    liveCells=set()

    #initial unknown cell as empty set

    unknownCells=set()

    #iteration in every item in liveDict

    for key,_ in grid._liveDict.items():

        #add all dead cell around this live cell to unknownCells

        unknownCells.update([(i,j) for j in range(max(0,key[1]-1),min(key[1]+1\

                              ,grid.numCols()-1)+1) for i in range(max(0,key[0]\

                                           -1),min(key[0]+1,grid.numRows()-1)+1) if not grid.isLiveCell(i,j)])

        #find number of neighbors around this cell

        neighbors=grid.numLiveNeighbors(key[0],key[1])

        #check neighbors from condition , if true then add to liveCells

        if (neighbors==2 and grid.isLiveCell(key[0],key[1])) or neighbors==3:

            liveCells.add(key)

    #iteration in every unknownCells

    for i,j in unknownCells:

        #find neighbors of this cell

        neighbors=grid.numLiveNeighbors(i,j)

        #check condition with neighbors , if true then add to liveCells

        if (neighbors==2 and grid.isLiveCell(i,j)) or neighbors==3:

            liveCells.add((i,j))

    #configure grid with liveCells

    grid.configure(list(liveCells))



#function that evolve grid for one turn

def evolve4(grid):

    #shift to bottom

    grid1=np.append(np.zeros([1,grid.numCols()]),grid[:grid.numRows()-1],axis=0)

    #shift to above

    grid2=np.append(grid[1:],np.zeros([1,grid.numCols()]),axis=0)

    #shift to left

    grid3=np.append(grid[:,1:],np.zeros([grid.numRows(),1]),axis=-1)

    #shift to right

    grid4=np.append(np.zeros([grid.numRows(),1]),grid[:,:grid.numCols()-1],axis=-1)

    #shift to bottom and left

    grid5=np.append(grid1[:,1:],np.zeros([grid.numRows(),1]),axis=-1)

    #shift to bottom and right

    grid6=np.append(np.zeros([grid.numRows(),1]),grid1[:,:grid.numCols()-1],axis=-1)

    #shift to above and left

    grid7=np.append(grid2[:,1:],np.zeros([grid.numRows(),1]),axis=-1)

    #shift to above and right

    grid8=np.append(np.zeros([grid.numRows(),1]),grid2[:,:grid.numCols()-1],axis=-1)

    #sum of all 8 girds

    temp_grid=grid1+grid2+grid3+grid4+grid5+grid6+grid7+grid8

    

    grid_new=grid._grid

    grid_new[(grid._grid==0) & (temp_grid==3)]=1

    grid_new[(grid._grid==1) & (temp_grid<2)]=0

    grid_new[(grid._grid==1) & (temp_grid>3)]=0

    grid._grid=grid_new



temp_plot={}

for i in [10,20,30,40,50,60,70,80,90]:

    temp_plot[i]={}

 

print("Start Algorithm2")

for p in [10,20,30,40,50,60,70,80,90]:

    print("For n = 1000,",p,"%")

    INIT_CONFIG=[(np.random.randint(0,1000),np.random.randint(0,1000)) for k in range(int(1000*1000*p/100))]

     #initialize grid with HEIGHT AND WIDTH

    grid=LifeGrid2(1000,1000)

    #configure grid with initial live cell

    grid.configure(INIT_CONFIG)

    start=time()

    #describe how to do in each gen

    for i in range(NUM_GEN):

        #evolve grid

        evolve2(grid)

        #draw grid

        #draw(grid)

    endd=time()

    print(endd-start,"Second")

    temp_plot[p]['algo2']=endd-start

    

print("Start Algorithm4")

for p in [10,20,30,40,50,60,70,80,90]:

    print("For n = 1000,",p,"%")

    INIT_CONFIG=[(np.random.randint(0,1000),np.random.randint(0,1000)) for k in range(int(1000*1000*p/100))]

     #initialize grid with HEIGHT AND WIDTH

    grid=LifeGrid4(1000,1000)

    #configure grid with initial live cell

    grid.configure(INIT_CONFIG)

    start=time()

    #describe how to do in each gen

    for i in range(NUM_GEN):

        #evolve grid

        evolve4(grid)

        #draw grid

        #draw(grid)

    endd=time()

    print(endd-start,"Second")

    temp_plot[p]['algo4']=endd-start

df=pd.DataFrame(temp_plot)

df.T.plot(kind='bar',logy=True,grid=True,title="Time comparison")

#plt.title("Time Comparison")

plt.xlabel("Ratio of initial input")

plt.ylabel("time(sec)")

plt.show()