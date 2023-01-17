from scipy.optimize import linprog



c = [-1, 4]

A = [[-3, 1], [1, 2]]

b = [6, 4]

x0_bounds = (None, None)

x1_bounds = (-3, None)







res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds),

              options={"disp": True})

print(res)
import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



# For Density plots

from plotly.tools import FigureFactory as FF







data_matrix = [['Train Wagon', 'Item Capacity', 'Space Capacity'],

               ['w1', 10, 5000],

               ['w2', 8, 4000],

               ['w3', 12, 8000],]

               

table = FF.create_table(data_matrix)

iplot(table)



data_matrix = [['Cargo<br>Type', '#Items Available', 'Volume','Profit'],

               ['c1', 18, 400,2000],

               ['c2', 10, 300,2500],

               ['c3', 5, 200,5000],

               ['c4', 20, 500,3500]]

               

table = FF.create_table(data_matrix)

iplot(table)

from scipy.optimize import linprog





"""

/* Objective function */

max: +2000 C1 +2500 C2 +5000 C3 +3500 C4 +2000 C5 +2500 C6 +5000 C7 +3500 C8 +2000 C9 +2500 C10 +5000 C11

 +3500 C12;



Above flip sign to get min problem.





/* Constraints */

+C1 +C2 +C3 +C4 <= 10;

+C5 +C6 +C7 +C8 <= 8;

+C9 +C10 +C11 +C12 <= 12;

+400 C1 +300 C2 +200 C3 +500 C4 <= 5000;

+400 C5 +300 C6 +200 C7 +500 C8 <= 4000;

+400 C9 +300 C10 +200 C11 +500 C12 <= 8000;

+C1 +C5 +C9 <= 18;

+C2 +C6 +C10 <= 10;

+C3 +C7 +C11 <= 5;

+C4 +C8 +C12 <= 20;



"""



# Change min to max

c = [-2000,-2500,-5000,-3500,-2000,-2500,-5000,-3500,-2000,-2500,-5000,-3500]

xb=[]

for i in range(0,12):

    xb.append((0, None))



A = [[1,1,1,1,0,0,0,0,0,0,0,0], 

     [0,0,0,0,1,1,1,1,0,0,0,0],

     [0,0,0,0,0,0,0,0,1,1,1,1],

     [400,300,200,500,0,0,0,0,0,0,0,0,],

     [0,0,0,0,400,300,200,500,0,0,0,0,],

     [0,0,0,0,0,0,0,0,400,300,200,500],

     [1,0,0,0,1,0,0,0,1,0,0,0],

     [0,1,0,0,0,1,0,0,0,1,0,0],

     [0,0,1,0,0,0,1,0,0,0,1,0],

     [0,0,0,1,0,0,0,1,0,0,0,1],

    ]    



b = [10,8,12,5000,4000,8000,18,10,5,20]



res = linprog(c, A_ub=A, b_ub=b, bounds=xb,

              options={"disp": True})

print(res)
import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



# For Density plots

from plotly.tools import FigureFactory as FF







data_matrix = [['Train Wagon', 'Item Capacity', 'Space Capacity'],

               ['w1', 'inf', 5000],

               ['w2', 'inf', 4000],

               ['w3', 'inf', 8000],]

               

table = FF.create_table(data_matrix)

iplot(table)



data_matrix = [['Cargo<br>Type', '#Items Available', 'Volume','Profit'],

               ['c1', 18, 400,2000],

               ['c2', 10, 300,2500],

               ['c3', 5, 200,5000],

               ['c4', 20, 500,3500]]

               

table = FF.create_table(data_matrix)

iplot(table)
from scipy.optimize import linprog





# What if we get rid of item constraint?

# Change min to max

c = [-2000,-2500,-5000,-3500,-2000,-2500,-5000,-3500,-2000,-2500,-5000,-3500]

xb=[]

for i in range(0,12):

    xb.append((0, None))



A = [

     [400,300,200,500,0,0,0,0,0,0,0,0,],

     [0,0,0,0,400,300,200,500,0,0,0,0,],

     [0,0,0,0,0,0,0,0,400,300,200,500],

     [1,0,0,0,1,0,0,0,1,0,0,0],

     [0,1,0,0,0,1,0,0,0,1,0,0],

     [0,0,1,0,0,0,1,0,0,0,1,0],

     [0,0,0,1,0,0,0,1,0,0,0,1],

    ]    



b = [5000,4000,8000,18,10,5,20]



res = linprog(c, A_ub=A, b_ub=b, bounds=xb,

              options={"disp": True})

print(res)