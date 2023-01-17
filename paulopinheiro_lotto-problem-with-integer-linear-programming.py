#install packages before anything else

!pip install cvxopt

!pip install cvxpy

!pip install mip

!pip install pulp
#import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import notebook

import numpy as np

import cvxpy as cp

from itertools import combinations 

from ortools.linear_solver import pywraplp

from ortools.sat.python import cp_model

import cvxopt

import cvxpy

from mip import Model, xsum, maximize, minimize, BINARY

import pulp as plp

import random

import copy
#For referecne, display available solvers for CVXPY

print(cvxpy.installed_solvers())
# generate alll tickets and possible draws given the problem parameters

def generate(n,k,p,t):

    comb_tickets = combinations([i+1 for i in range(n)], k) 

    tickets = np.array([set(x) for x in list(comb_tickets)])



    comb_draws = combinations([i+1 for i in range(n)], p) 

    draws = np.array([set(x) for x in list(comb_draws)])

    

    coef_matrix = np.zeros((len(draws), len(tickets)), dtype=int)

    

    for i in notebook.tqdm(range(coef_matrix.shape[0])):

        for j in range(coef_matrix.shape[1]):

            intersection = len(draws[i] & tickets[j])

            coef_matrix[i,j] = 1 if intersection >= t else 0



    return draws, tickets, coef_matrix
def MIP_Optmise(coef_matrix):

    

    w = coef_matrix

    T = range(w.shape[1])

    D = range(w.shape[0])

    m = Model('lotto')



    x = [m.add_var(var_type=BINARY) for i in T]



    m.objective = minimize(xsum(x[i] for i in T))



    for i in D:

        m += (xsum(w[i][j]* x[j] for j in T) >= 1)



    #print(m.optimize(max_seconds=60))

    print(m.optimize())

    

    selected = [i for i in T if x[i].x >= 0.99]

    #print('selected items: {}'.format(selected))

    

    deselected = [i for i in T if x[i].x <= 0.99]

    

    return selected, deselected  
def CVXPY_Optmise(coef_matrix):

    

    M = coef_matrix



    selection = cp.Variable(M.shape[1], boolean = True)

    ones_vec = np.ones(M.shape[1], dtype=int)

    

    constraints = []

    

    for i in range(M.shape[0]):

        constraints.append(M[i] * selection >= 1)



    cost = ones_vec * selection



    problem = cp.Problem(cp.Minimize(cost), constraints)



    print(problem.solve(solver=cp.GLPK_MI))



    selected = np.nonzero(selection.value > 0.75)

    #print('selected items: {}'.format(list(selected[0])))

    

    deselected = np.nonzero(selection.value < 0.75)

    

    return list(selected[0]), list(deselected[0])  
def OR_Optmise(coef_matrix):    

    w = coef_matrix

    T = range(w.shape[1])

    D = range(w.shape[0])

    

    # Create the mip solver with the CBC backend.

    solver = pywraplp.Solver('simple_mip_program', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    

    # Variables

    x = {}

    for i in T:

        x[i] = solver.BoolVar('x_%i' % (i))

    

    # The amount packed

    for i in D:

        solver.Add(solver.Sum(w[i][j] * x[j] for j in T) >= 1)  

        

    solver.Minimize(solver.Sum([x[j] for j in T]))

      

    status = solver.Solve()

    

    if status == pywraplp.Solver.OPTIMAL:

        print('An optimal feasible solution was found.')

    elif status == pywraplp.Solver.FEASIBLE:

        print('A feasible solution was found, but we dont know if its optimal.')

    elif status == pywraplp.Solver.INFEASIBLE:

        print('The problem was proven infeasible.')

        return

    elif status == pywraplp.Solver.MODEL_INVALID:

        print('The given CpModelProto didnt pass the validation step.')

        return

    else:

        print('The status of the model is unknown because a search limit was reached.')

        return

         

    # The objective value of the solution.

    print('WallTime = %f mSecs' % solver.WallTime())

    

    selected = [i for i in T if x[i].solution_value() == 1]

    #print('selected items: {}'.format(selected))

    

    deselected = [i for i in T if x[i].solution_value() < 1]

    

    return selected, deselected  
def OR_ContraintProgramming(coef_matrix, target, greedy_initialisation = False):    

    

    w = coef_matrix

    T = range(w.shape[1])

    D = range(w.shape[0])

    

    # Create the mip solver with the CBC backend.

    model = cp_model.CpModel()

    

    # Variables

    x = {}

    for i in T:

        x[i] = model.NewBoolVar('x_%i' % (i))

    

    if greedy_initialisation == True:

        print("Greedy Init")

        selected, deselected = Greedy_Optmise(coef_matrix)

        for i in selected:

            model.AddHint(x[i], True)

        

    #break symmetry when it exists to speed up solution finding

    model.Add(x[0] == 1) 

    

    model.Add(sum([x[j] for j in T])==target)

    

    # The amount packed

    for i in D:

        model.Add(sum(w[i][j] * x[j] for j in T) >= 1)  

              

    solver = cp_model.CpSolver()



    status = solver.Solve(model)

    

    print(solver.ResponseStats())

    

    if status == cp_model.OPTIMAL:

        print('An optimal feasible solution was found.')

    elif status == cp_model.FEASIBLE:

        print('A feasible solution was found, but we dont know if its optimal.')

    elif status == cp_model.INFEASIBLE:

        print('The problem was proven infeasible.')

        return [], []

    elif status == cp_model.MODEL_INVALID:

        print('The given CpModelProto didnt pass the validation step.')

        return [], []

    else:

        print('The status of the model is unknown because a search limit was reached.')

        return [], []

    

    selected = [i for i in T if solver.Value(x[i]) == 1]

    print('selected items: {}'.format(selected))

    

    deselected = [i for i in T if solver.Value(x[i]) < 1]

    

    return selected, deselected  
def PULP_Optmise(coef_matrix):    

    w = coef_matrix

    T = range(w.shape[1])

    D = range(w.shape[0])

       

    opt_model = plp.LpProblem(name="MIP_Model")

   

    # if x is Binary

    x_vars  = {(i): plp.LpVariable(cat=plp.LpBinary, name="x_{0}".format(i)) for i in T}



    # >= constraints

    constraints = {i : opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(w[i,j] * x_vars[j] for j in T),

                         sense=plp.LpConstraintGE,

                         rhs=1,

                         name="constraint_{0}".format(i))) for i in D}



    objective = plp.lpSum(x_vars[i] for i in T)



    # for minimization

    opt_model.sense = plp.LpMinimize

    opt_model.setObjective(objective)



    # solving with CBC

    opt_model.solve()

    

    selected = [i for i in T if x_vars[i].varValue == 1]

    #print('selected items: {}'.format(selected))

    

    deselected = [i for i in T if x_vars[i].varValue < 1]

    

    return selected, deselected  
def RemoveRedundant(coef_matrix, selected, deselected):

    

    coef_matrix_red = np.zeros(coef_matrix.shape, dtype=int)

    

    for i in selected:

        coef_matrix_red[:,i] = coef_matrix[:,i]

        

    v = np.count_nonzero(coef_matrix_red, axis=1)

    

    selected_copy = copy.deepcopy(selected)

    for i in selected_copy:

        if(min(v[np.nonzero(coef_matrix_red[:,i])])>1):

            selected.remove(i)

            deselected.append(i)

            v = np.subtract(v, coef_matrix_red[:,i])

            

    return selected, deselected
def Greedy_Optmise(coef_matrix):

    # let solution be empty

    X = set()

    # for each ticket compute how many draws it covers

    d = np.count_nonzero(coef_matrix, axis=0)

    # initialise set of uncovered draws i.e. all uncovered

    I = set(range(coef_matrix.shape[0]))

    # initialise set of unselected tickets i.e. none selected

    J = set(range(coef_matrix.shape[1]))

    # select first ticket

    w = 0

    # add to the solution

    X.add(w)

    # remove the ticket from the unselected list

    J.remove(w)

    # get draws it covers

    Dw = set(np.nonzero(coef_matrix[:,w])[0])

    # remove covered draws from the uncovered set

    I = I - Dw

    for i in Dw:

        d = np.subtract(d, coef_matrix[i,:])

    while I != set():

        print("-", end="")

        # select random ticket that maximizes uncovered draws

        f = (np.where(d == np.amax(d))[0]) 

        w = random.choice(f)

        # add to the solution

        X.add(w)

        # remove the ticket from the unselected list

        J.remove(w)

        # get draws it covers

        Dw = set(np.nonzero(coef_matrix[:,w])[0])

        Dw = Dw.intersection(I)

        # remove covered draws from the uncovered set

        I = I - Dw 

        for i in Dw:

            d = np.subtract(d, coef_matrix[i,:]) 

    print("")

    

    return list(X), list(J)
def Improve(coef_matrix, selected, deselected, factor=0.5, algo='or'):

    #select % of solution

    sampling = random.sample(selected, k=int(len(selected)*factor))

    print(" Keeping ", len(sampling), "tickets ouf of" , len(selected))

    reduced_coef_matrix = coef_matrix[:,sampling]

    #check which draws are no longer covered

    v = np.count_nonzero(reduced_coef_matrix, axis=1)

    not_coverd_draws = np.where(v == 0)[0]

    

    others_tickets = list(set(range(coef_matrix.shape[1]))-set(sampling))

    

    reduced_coef_matrix = coef_matrix[:, others_tickets]

    reduced_coef_matrix = reduced_coef_matrix[not_coverd_draws, :]

    

    print(" OR Size = ", reduced_coef_matrix.shape)

    

    

    if algo == 'cvx':

        print('CVXPY Optimise')

        x, y = CVXPY_Optmise(reduced_coef_matrix)

        

    if algo == 'mip':

        print('MIP Optimise')

        x, y = MIP_Optmise(reduced_coef_matrix) 

        

    if algo == 'or':

        print('OR Optimise')

        x, y = OR_Optmise(reduced_coef_matrix)  



    if algo == 'pulp':

        print('PULP Optimise')

        x, y = PULP_Optmise(reduced_coef_matrix)      

    

    

    for i in x:

        sampling.append(others_tickets[i])

       

    others_tickets = list(set(range(coef_matrix.shape[1]))-set(sampling))

        

    return RemoveRedundant(coef_matrix, sampling, others_tickets)
def Greedy_Optmise_Tries(coef_matrix, tries=1, factor=0.5, algo = 'or'):

    selected, deselected = Greedy_Optmise(coef_matrix)

    print("Original Solution Tickets =", len(selected))



    selected, deselect = RemoveRedundant(coef_matrix, selected, deselected)

    print("Remove Redundant Tickets =", len(selected))



    for i in range(tries):

        print("Iteration = ", i, end="")

        selected, deselect = Improve(coef_matrix, selected, deselected, factor, algo)

        print("Improved Tickets =", len(selected))



    return selected, deselected      
#get tickets that provides the solution to our problem

def get_tickets(draws, tickets, coef_matrix, deselected):

    tickets_str = [str(x) for x in tickets]

    draws_str = [str(x) for x in draws]



    coefficient_matrix = pd.DataFrame(coef_matrix, index = draws_str, columns = tickets_str)

    coefficient_matrix = coefficient_matrix.drop(coefficient_matrix.columns[deselected], axis=1)

    

    return coefficient_matrix
# perform the optmisation using the library of our choice

def get_reduced_wheel(n,k,p,t,algo, target=0, greedy_initialisation = False, factor=0.5, tries=1):

    

    draws, tickets, coef_matrix = generate(n,k,p,t)



    number_tickets = len(tickets)

    print('Tickets =', number_tickets)



    number_draws = len(draws)

    print('Draws =', number_draws)

    

    selected, deselected =  Greedy_Optmise_Tries(coef_matrix, tries, factor, algo)

        

    optimal_tickets = get_tickets(draws, tickets, coef_matrix, deselected)

    

    return optimal_tickets
n = 12  # numbers in the set

k = 5  # numbers in ticket

p = 4  # numbers in draw

t = 3 # at least match
%%time

Greedy_tickets = get_reduced_wheel(n,k,p,t,algo = 'cvx', factor=0.75, tries=10)
Greedy_tickets