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
#Import required packages

import numpy as np

from scipy.optimize import linprog
LABOR_HOURS_TYPE1 = 9

LABOR_HOURS_TYPE2 = 6

PROFIT_UNIT_TYPE1 = 350

PROFIT_UNIT_TYPE2 = 300

TUBING_FEET_TYPE1 = 12

TUBING_FEET_TYPE2 = 16



MAX_TOTAL_TUBES = 200

MAX_LABOR_HOURS = 1566

MAX_TUBING = 2880 
#Solve the resulting Linear Program

f = np.array([-PROFIT_UNIT_TYPE1,-PROFIT_UNIT_TYPE2]); 



#Objective function



#maximize PROFIT_UNIT_TYPE1*x1 + PROFIT_UNIT_TYPE2*x2

#equivalent to minimize -PROFIT_UNIT_TYPE1*x1 - PROFIT_UNIT_TYPE2*x2



A_ineq = np.array([[1,1],[LABOR_HOURS_TYPE1,LABOR_HOURS_TYPE2],[TUBING_FEET_TYPE1,TUBING_FEET_TYPE2]]);

b_ineq = np.array([MAX_TOTAL_TUBES,MAX_LABOR_HOURS,MAX_TUBING]);

lb_ub = (0,None);

res = linprog(f, A_ub=A_ineq, b_ub=b_ineq,bounds=lb_ub);



maxProfit = -res.fun; #Since we minimized the negative 



print("Max value of objective = ",maxProfit);

print(' Solution x = ', res.x);

res
LABOR_HOURS_TYPE1 = 9

LABOR_HOURS_TYPE2 = 6

PROFIT_UNIT_TYPE1 = 350

PROFIT_UNIT_TYPE2 = 300

TUBING_FEET_TYPE1 = 16

TUBING_FEET_TYPE2 = 12



MAX_TOTAL_TUBES = 200

MAX_LABOR_HOURS = 1566

MAX_TUBING = 2880
#Solve the resulting Linear Program

f = np.array([-PROFIT_UNIT_TYPE1,-PROFIT_UNIT_TYPE2]); 



#Objective function



#maximize PROFIT_UNIT_TYPE1*x1 + PROFIT_UNIT_TYPE2*x2

#equivalent to minimize -PROFIT_UNIT_TYPE1*x1 - PROFIT_UNIT_TYPE2*x2



A_ineq = np.array([[1,1],[LABOR_HOURS_TYPE1,LABOR_HOURS_TYPE2],[TUBING_FEET_TYPE1,TUBING_FEET_TYPE2]]);

b_ineq = np.array([MAX_TOTAL_TUBES,MAX_LABOR_HOURS,MAX_TUBING]);

lb_ub = (0,None);

res = linprog(f, A_ub=A_ineq, b_ub=b_ineq,bounds=lb_ub);



maxProfit = -res.fun; #Since we minimized the negative 



print("Max value of objective = ",maxProfit);

print(' Solution x = ', res.x);
LABOR_HOURS_TYPE1 = 6

LABOR_HOURS_TYPE2 = 9

PROFIT_UNIT_TYPE1 = 350

PROFIT_UNIT_TYPE2 = 300

TUBING_FEET_TYPE1 = 12

TUBING_FEET_TYPE2 = 16



MAX_TOTAL_TUBES = 200

MAX_LABOR_HOURS = 1566

MAX_TUBING = 2880
#Solve the resulting Linear Program

f = np.array([-PROFIT_UNIT_TYPE1,-PROFIT_UNIT_TYPE2]); 



#Objective function



#maximize PROFIT_UNIT_TYPE1*x1 + PROFIT_UNIT_TYPE2*x2

#equivalent to minimize -PROFIT_UNIT_TYPE1*x1 - PROFIT_UNIT_TYPE2*x2



A_ineq = np.array([[1,1],[LABOR_HOURS_TYPE1,LABOR_HOURS_TYPE2],[TUBING_FEET_TYPE1,TUBING_FEET_TYPE2]]);

b_ineq = np.array([MAX_TOTAL_TUBES,MAX_LABOR_HOURS,MAX_TUBING]);

lb_ub = (0,None);

res = linprog(f, A_ub=A_ineq, b_ub=b_ineq,bounds=lb_ub);



maxProfit = -res.fun; #Since we minimized the negative 



print("Max value of objective = ",maxProfit);

print(' Solution x = ', res.x);
LABOR_HOURS_TYPE1 = 9

LABOR_HOURS_TYPE2 = 6

PROFIT_UNIT_TYPE1 = 350

PROFIT_UNIT_TYPE2 = 300

TUBING_FEET_TYPE1 = 36

TUBING_FEET_TYPE2 = 16



MAX_TOTAL_TUBES = 200

MAX_LABOR_HOURS = 1700

MAX_TUBING = 2880
#Solve the resulting Linear Program

f = np.array([-PROFIT_UNIT_TYPE1,-PROFIT_UNIT_TYPE2]); 



#Objective function



#maximize PROFIT_UNIT_TYPE1*x1 + PROFIT_UNIT_TYPE2*x2

#equivalent to minimize -PROFIT_UNIT_TYPE1*x1 - PROFIT_UNIT_TYPE2*x2



A_ineq = np.array([[1,1],[LABOR_HOURS_TYPE1,LABOR_HOURS_TYPE2],[TUBING_FEET_TYPE1,TUBING_FEET_TYPE2]]);

b_ineq = np.array([MAX_TOTAL_TUBES,MAX_LABOR_HOURS,MAX_TUBING]);

lb_ub = (0,None);

res = linprog(f, A_ub=A_ineq, b_ub=b_ineq,bounds=lb_ub);



maxProfit = -res.fun; #Since we minimized the negative 



print("Max value of objective = ",maxProfit);

print(' Solution x = ', res.x);
LABOR_HOURS_TYPE1 = 6

LABOR_HOURS_TYPE2 = 9

PROFIT_UNIT_TYPE1 = 350

PROFIT_UNIT_TYPE2 = 300

TUBING_FEET_TYPE1 = 16

TUBING_FEET_TYPE2 = 12



MAX_TOTAL_TUBES = 200

MAX_LABOR_HOURS = 1700

MAX_TUBING = 2880
#Solve the resulting Linear Program

f = np.array([-PROFIT_UNIT_TYPE1,-PROFIT_UNIT_TYPE2]); 



#Objective function



#maximize PROFIT_UNIT_TYPE1*x1 + PROFIT_UNIT_TYPE2*x2

#equivalent to minimize -PROFIT_UNIT_TYPE1*x1 - PROFIT_UNIT_TYPE2*x2



A_ineq = np.array([[1,1],[LABOR_HOURS_TYPE1,LABOR_HOURS_TYPE2],[TUBING_FEET_TYPE1,TUBING_FEET_TYPE2]]);

b_ineq = np.array([MAX_TOTAL_TUBES,MAX_LABOR_HOURS,MAX_TUBING]);

lb_ub = (0,None);

res = linprog(f, A_ub=A_ineq, b_ub=b_ineq,bounds=lb_ub);



maxProfit = -res.fun; #Since we minimized the negative 



print("Max value of objective = ",maxProfit);

print(' Solution x = ', res.x);
LABOR_HOURS_TYPE1 = 9

LABOR_HOURS_TYPE2 = 6

PROFIT_UNIT_TYPE1 = 350

PROFIT_UNIT_TYPE2 = 300

TUBING_FEET_TYPE1 = 16

TUBING_FEET_TYPE2 = 12



MAX_TOTAL_TUBES = 200

MAX_LABOR_HOURS = 1700

MAX_TUBING = 2880
#Solve the resulting Linear Program

f = np.array([-PROFIT_UNIT_TYPE1,-PROFIT_UNIT_TYPE2]); 



#Objective function



#maximize PROFIT_UNIT_TYPE1*x1 + PROFIT_UNIT_TYPE2*x2

#equivalent to minimize -PROFIT_UNIT_TYPE1*x1 - PROFIT_UNIT_TYPE2*x2



A_ineq = np.array([[1,1],[LABOR_HOURS_TYPE1,LABOR_HOURS_TYPE2],[TUBING_FEET_TYPE1,TUBING_FEET_TYPE2]]);

b_ineq = np.array([MAX_TOTAL_TUBES,MAX_LABOR_HOURS,MAX_TUBING]);

lb_ub = (0,None);

res = linprog(f, A_ub=A_ineq, b_ub=b_ineq,bounds=lb_ub);



maxProfit = -res.fun; #Since we minimized the negative 



print("Max value of objective = ",maxProfit);

print(' Solution x = ', res.x);
LABOR_HOURS_TYPE1 = 9

LABOR_HOURS_TYPE2 = 6

PROFIT_UNIT_TYPE1 = 350

PROFIT_UNIT_TYPE2 = 300

TUBING_FEET_TYPE1 = 12

TUBING_FEET_TYPE2 = 16

##

MAX_TOTAL_TUBES = 200

MAX_LABOR_HOURS = 1700

MAX_TUBING = 2880
#Solve the resulting Linear Program

f = np.array([-PROFIT_UNIT_TYPE1,-PROFIT_UNIT_TYPE2]); 



#Objective function



#maximize PROFIT_UNIT_TYPE1*x1 + PROFIT_UNIT_TYPE2*x2

#equivalent to minimize -PROFIT_UNIT_TYPE1*x1 - PROFIT_UNIT_TYPE2*x2



A_ineq = np.array([[1,1],[LABOR_HOURS_TYPE1,LABOR_HOURS_TYPE2],[TUBING_FEET_TYPE1,TUBING_FEET_TYPE2]]);

b_ineq = np.array([MAX_TOTAL_TUBES,MAX_LABOR_HOURS,MAX_TUBING]);

lb_ub = (0,None);

res = linprog(f, A_ub=A_ineq, b_ub=b_ineq,bounds=lb_ub);



maxProfit = -res.fun; #Since we minimized the negative 



print("Max value of objective = ",maxProfit);

print(' Solution x = ', res.x);
# !pip install pulp
import pulp

from pulp import *

import sys

import StringIO
# looking for an optimal maximum so we use LpMaximize (default = LpMinimize)

manufacturing_products_lp = pulp.LpProblem("Manufacturing 2 types of hot tubs", pulp.LpMaximize)



# Setting the lower and upper bounds for the decision variables

# Default lowBound is negative infinity

# Default upBound is positive infinity

# Default cat is 'Continuous'

X1 = pulp.LpVariable('X1', lowBound=0, cat='Integer')

X2 = pulp.LpVariable('X2', lowBound=0, cat='Integer')



# The objective function and constraints are added using the += operator to our model.



# Objective function

#manufacturing_products_lp += PROFIT_UNIT_TYPE1 * X1 + PROFIT_UNIT_TYPE2 * X2, "Z"

manufacturing_products_lp += 350 * X1 + 300 * X2, "Z"

#strObjective = "manufacturing_products_lp += " + str(PROFIT_UNIT_TYPE1) + " * X1 " + " + " + str(PROFIT_UNIT_TYPE2) + " * X2" + "," + " " + "\"Z\"";

#eval(strObjective);



# Constraints

#manufacturing_products_lp += X1 + X2 <= MAX_TOTAL_TUBES, "C1 : Constraint on availability of pumps"

manufacturing_products_lp += X1 + X2 <= 200, "C1 : Constraint on availability of pumps"



#manufacturing_products_lp += LABOR_HOURS_TYPE1 * X1 + LABOR_HOURS_TYPE2 * X2 <= MAX_LABOR_HOURS, "C2 : Constraint on available labour hours"

manufacturing_products_lp += 9 * X1 + 6 * X2 <= 1700, "C2 : Constraint on available labour hours"



#manufacturing_products_lp += TUBING_FEET_TYPE1 * X1 + TUBING_FEET_TYPE2 * X2 <= MAX_TUBING, "C2 : Constraint on available tubing"

manufacturing_products_lp += 12 * X1 + 16 * X2 <= 2880, "C2 : Constraint on available tubing"
strObjective = "manufacturing_products_lp += " + str(PROFIT_UNIT_TYPE1) + " * X1 " + " + " + str(PROFIT_UNIT_TYPE2) + " * X2" + ","

print(strObjective)

eval(strObjective)
manufacturing_products_lp
manufacturing_products_lp.solve()



# There are 5 status codes: 'Not Solved', 'Optimal', 'Infeasible', 'Unbounded', 'Undefined

pulp.LpStatus[manufacturing_products_lp.status]
print("The optimized solution :", pulp.value(manufacturing_products_lp.objective) )

for variable in manufacturing_products_lp.variables():

    print(variable.name, variable.varValue)