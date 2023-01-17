
#Edited version for Data Analytics Cia 1 Component 1 
############################ By #################################
#                      Kumar Navin Barnwal                      #
#                         1847267                               #
#################################################################
import random
import csv
import time
import scipy.optimize
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numba import njit
from ortools.linear_solver import pywraplp

NMB_DAYS = 100
NMB_FAMILIES = 5000
MAX_PEOPLE_PERDAY = 300
MIN_PEOPLE_PERDAY = 125

# only top 5 (0-4) preferences are allowed for assignment
# this decreases size of the model
MAX_PREF_ALLOWED = 4

#**************** solution representation *********************
#current solution cost
cost_ = 0

#[I] is the day family I is assigned to
assignment_ = [-1 for i in range(NMB_FAMILIES)]

#[d] is number of people assigned to day d
nmb_people_assigned_to_day_ = [0 for i in range(NMB_DAYS)]

#[d] is the list of families assigned to day d
families_assigned_to_day_ = []
for i in range(NMB_DAYS):
  families_assigned_to_day_.append([])   
#**************************************************************

def get_penalty(n, choice):
    penalty = None
    if choice == 0:
        penalty = 0
    elif choice == 1:
        penalty = 50
    elif choice == 2:
        penalty = 50 + 9 * n
    elif choice == 3:
        penalty = 100 + 9 * n
    elif choice == 4:
        penalty = 200 + 9 * n
    elif choice == 5:
        penalty = 200 + 18 * n
    elif choice == 6:
        penalty = 300 + 18 * n
    elif choice == 7:
        penalty = 300 + 36 * n
    elif choice == 8:
        penalty = 400 + 36 * n
    elif choice == 9:
        penalty = 500 + 36 * n + 199 * n
    else:
        penalty = 500 + 36 * n + 398 * n
    return penalty

def GetAssignmentCostMatrix(data):
    cost_matrix = np.zeros((NMB_FAMILIES, NMB_DAYS), dtype=np.int64)
    for i in range(NMB_FAMILIES):
        desired = data.values[i, :-1]
        cost_matrix[i, :] = get_penalty(FAMILY_SIZE[i], 10)
        for j, day in enumerate(desired):
            cost_matrix[i, day-1] = get_penalty(FAMILY_SIZE[i], j)
    return cost_matrix

def GetAccountingCostMatrix():
    ac = np.zeros((1000, 1000), dtype=np.float64)
    for n in range(ac.shape[0]):
        for n_p1 in range(ac.shape[1]):
            diff = abs(n - n_p1)
            ac[n, n_p1] = max(0, (n - 125) / 400.0 * n**(0.5 + diff / 50.0))
    return ac

def GetPreferenceMatrix(data):
    pref_matrix = np.zeros((NMB_FAMILIES, NMB_DAYS), dtype=np.int64)
    for i in range(NMB_FAMILIES):
        desired = data.values[i, :-1]
        pref_matrix[i, :] = 10
        for j, day in enumerate(desired):
            pref_matrix[i, day-1] = j
    return pref_matrix

def GetPreferenceForFamiliesMatrix(data):
    pref_matrix = np.zeros((NMB_FAMILIES, NMB_DAYS), dtype=np.int64)
    pref_matrix2 = np.zeros((NMB_FAMILIES, 10), dtype=np.int64)
    for i in range(NMB_FAMILIES):
        desired = data.values[i, :-1]
        pref_matrix[i, :] = 10
        for j, day in enumerate(desired):
            pref_matrix[i, day-1] = j
    for i in range(NMB_FAMILIES):
      for j in range(NMB_DAYS):
        if(pref_matrix[i][j] < 10):
           pref_matrix2[i][pref_matrix[i][j]] = j         
    return pref_matrix2

# to speed up the local search
# [D1][D2] is the list of families currently assigned to 
# D1 that can be swapped with something on day D2
swap_candidates_ = []   
for i in range(NMB_DAYS):
    swap_candidates_.append([])
    for j in range(NMB_DAYS):
        swap_candidates_[i].append([])

        
# drop some assignment possibilities in order to reduce problem size
def preprocessing():

    # I tried to understand the preprocessing but the data was just numerical. They did reduced some data for usage but I was confused with the code sir.
    # I'll try to solve the question myself once and then reupload this comment sir. Soon.

    nbTopForBinN = {}
    for i in range(10):
        for j in range(NMB_DAYS):
           nbTopForBinN[i, j] = 0

    for F in range(NMB_FAMILIES):
        for i in range(MAX_PREF_ALLOWED + 1):
             D = PREFERENCES_FOR_FAMILY[F][i]
             for p in range(i, MAX_PREF_ALLOWED + 1):
                 nbTopForBinN[p, D] += FAMILY_SIZE[F]
    
    nmbVarsDROPPED = 0

    for i in range(NMB_DAYS):
       if nbTopForBinN[0, i] < 100:
           for F in range(NMB_FAMILIES):
               if PREFERENCES_FOR_FAMILY[F][0] == i:
                  for P in range(1, MAX_PREF_ALLOWED + 1):
                    if PREFERENCE_MATRIX[F, PREFERENCES_FOR_FAMILY[F][P]] < 10:
                       nmbVarsDROPPED += 1
                       PREFERENCE_MATRIX[F, PREFERENCES_FOR_FAMILY[F][P]] = 10

    for D in range(NMB_DAYS):
        if nbTopForBinN[0, D] >= 300:
           for F in range(NMB_FAMILIES):
               for P in range(2, MAX_PREF_ALLOWED + 1):
                   if PREFERENCES_FOR_FAMILY[F][P] == D:
                      if PREFERENCE_MATRIX[F, PREFERENCES_FOR_FAMILY[F][P]] < 10:
                         nmbVarsDROPPED += 1
                         PREFERENCE_MATRIX[F, PREFERENCES_FOR_FAMILY[F][P]] = 10

    for D in range(NMB_DAYS):
        if nbTopForBinN[1, D] >= 300 and nbTopForBinN[0, D] < 300:
           for F in range(NMB_FAMILIES):
               for P in range(3, MAX_PREF_ALLOWED + 1):
                   if PREFERENCES_FOR_FAMILY[F][P] == D:
                      if PREFERENCE_MATRIX[F, PREFERENCES_FOR_FAMILY[F][P]] < 10:
                         nmbVarsDROPPED += 1
                         PREFERENCE_MATRIX[F, PREFERENCES_FOR_FAMILY[F][P]] = 10


    for D in range(NMB_DAYS):
        if nbTopForBinN[2, D] >= 300 and nbTopForBinN[0, D] < 300 and nbTopForBinN[1, D] < 300:
           for F in range(NMB_FAMILIES):
               for P in range(4, MAX_PREF_ALLOWED + 1):
                   if PREFERENCES_FOR_FAMILY[F][P] == D:
                      if PREFERENCE_MATRIX[F, PREFERENCES_FOR_FAMILY[F][P]] < 10:
                         nmbVarsDROPPED += 1
                         PREFERENCE_MATRIX[F, PREFERENCES_FOR_FAMILY[F][P]] = 10

    
# calculate current solution cost
def calculate_solution_cost():
    global cost_
    cost_ = 0
    assign_cost = 0
    account_cost = 0
    for i in range(NMB_FAMILIES):
       cost_ += ASSIGNMENT_COST_MATRIX[i][assignment_[i]]
    assign_cost = cost_
    for d in range(NMB_DAYS):
        today_count = nmb_people_assigned_to_day_[d]
        yesterday_count = today_count
        if(d < NMB_DAYS - 1):
           yesterday_count = nmb_people_assigned_to_day_[d + 1]
        cost_ += ACCOUNTING_COST_MATRIX[today_count][yesterday_count]
        account_cost += ACCOUNTING_COST_MATRIX[today_count][yesterday_count]
    return cost_, assign_cost, account_cost

# accept a given assignment as a new one
def update_solution(assign):
    global cost_ 
    global nmb_people_assigned_to_day_ 
    global assignment_    
    for i in range(NMB_DAYS):
       nmb_people_assigned_to_day_[i] = 0
       families_assigned_to_day_[i] = []
       for j in range(NMB_DAYS):
           swap_candidates_[i][j] = []
            
    for F in range(NMB_FAMILIES):
       day = assign[F] 
       assignment_[F] = day
       nmb_people_assigned_to_day_[day] += FAMILY_SIZE[F]
       families_assigned_to_day_[day].append(F)
    
       for k in range(MAX_PREF_ALLOWED + 1):
            DD = PREFERENCES_FOR_FAMILY[F][k]
            if(PREFERENCE_MATRIX[F][DD] <= MAX_PREF_ALLOWED):
               swap_candidates_[day][DD].append(F);
    
    calculate_solution_cost()  

# calculate accounting cost for a given occupacy list and given days
# used when evaluating moves
def calculate_accounting_cost(nmb_people_assigned_to_day_, days_to_calc):
    accounting_cost = 0
    for d in days_to_calc:
        today_count = nmb_people_assigned_to_day_[d]
        yesterday_count = nmb_people_assigned_to_day_[d]
        if(d < NMB_DAYS - 1):
           yesterday_count = nmb_people_assigned_to_day_[d + 1]
        accounting_cost += ACCOUNTING_COST_MATRIX[today_count][yesterday_count]    
    return accounting_cost

# check if move is feasible
def check_move(F1, F2, D2):
    global cost_ 
    global nmb_people_assigned_to_day_ 
    global assignment_  
    D1 = assignment_[F1]
    N1 = FAMILY_SIZE[F1]
    N2 = 0
    if(F2 >= 0):
        N2 = FAMILY_SIZE[F2]
    if(nmb_people_assigned_to_day_[D1] - N1 + N2 < MIN_PEOPLE_PERDAY):
        return False
    if(nmb_people_assigned_to_day_[D1] - N1 + N2 > MAX_PEOPLE_PERDAY):
        return False
    if(nmb_people_assigned_to_day_[D2] - N2 + N1 < MIN_PEOPLE_PERDAY):
        return False
    if(nmb_people_assigned_to_day_[D2] - N2 + N1 > MAX_PEOPLE_PERDAY):
        return False
    return True

def calculate_assignment_cost_diff_with_move(F1, F2, D2):
    D1 = assignment_[F1]
    cost_diff = 0
    cost_diff += (ASSIGNMENT_COST_MATRIX[F1][D2] - ASSIGNMENT_COST_MATRIX[F1][D1])
    if(F2 >= 0):
        cost_diff += (ASSIGNMENT_COST_MATRIX[F2][D1] - ASSIGNMENT_COST_MATRIX[F2][D2])
    return cost_diff

def calculate_accounting_cost_diff_with_move(F1, F2, D2):
    D1 = assignment_[F1]
    N1 = FAMILY_SIZE[F1]
    N2 = 0
    if(F2 >= 0):
        N2 = FAMILY_SIZE[F2]
    accounting_cost_diff = 0
    accounting_cost_old = 0
    accounting_cost_new = 0
    days_to_calc = {D1, D2}
    if(D1 > 0):
       days_to_calc.add(D1 - 1)
    if(D2 > 0):
       days_to_calc.add(D2 - 1)
    accounting_cost_old = calculate_accounting_cost(nmb_people_assigned_to_day_, days_to_calc)
    nmb_people_assigned_to_day_new_ = list(nmb_people_assigned_to_day_)
    nmb_people_assigned_to_day_new_[D1] += (N2 - N1)
    nmb_people_assigned_to_day_new_[D2] += (N1 - N2)
    accounting_cost_new = calculate_accounting_cost(nmb_people_assigned_to_day_new_, days_to_calc)
    accounting_cost_diff = accounting_cost_new - accounting_cost_old
    return accounting_cost_diff


def perform_move(F1, F2, D2):
    global cost_ 
    global nmb_people_assigned_to_day_ 
    global assignment_  
    D1 = assignment_[F1]
    N1 = FAMILY_SIZE[F1]
    N2 = 0
    if(F2 >= 0):
        N2 = FAMILY_SIZE[F2]
        
    assignment_[F1] = D2
    families_assigned_to_day_[D2].append(F1)
    families_assigned_to_day_[D1].remove(F1)
    
    for k in range(MAX_PREF_ALLOWED + 1):
        DD = PREFERENCES_FOR_FAMILY[F1][k]
        if(PREFERENCE_MATRIX[F1][DD] <= MAX_PREF_ALLOWED):
            swap_candidates_[D2][DD].append(F1);
            swap_candidates_[D1][DD].remove(F1);
                                        
    if(F2 >= 0):
       assignment_[F2] = D1
       families_assigned_to_day_[D1].append(F2)
       families_assigned_to_day_[D2].remove(F2)

       for k in range(MAX_PREF_ALLOWED + 1):
           DD = PREFERENCES_FOR_FAMILY[F2][k]
           if(PREFERENCE_MATRIX[F2][DD] <= MAX_PREF_ALLOWED):
                swap_candidates_[D1][DD].append(F2);
                swap_candidates_[D2][DD].remove(F2);
                
    nmb_people_assigned_to_day_[D1] += (N2 - N1)
    nmb_people_assigned_to_day_[D2] += (N1 - N2)
    
    calculate_solution_cost()


# once in 1000 times accept move that increases the cost (by not more than 50) 
TOLRAND = 1000
TOL = 50
assCostDiffTOL = 10000000

def local_search_shift_and_swap(nmbIters):
    iter = 0
    while iter < nmbIters:
        iter+=1
        F1 = random.randint(0, NMB_FAMILIES - 1)
        D1 = assignment_[F1]
        r = random.randint(0, MAX_PREF_ALLOWED)
        D2 = PREFERENCES_FOR_FAMILY[F1][r]
        if(D1 == D2):
            continue
        if(PREFERENCE_MATRIX[F1][D2] > MAX_PREF_ALLOWED):
           continue
        F2 = -1 # shift
        if(random.randint(0, 100) < 10): #swap                        
            if len(swap_candidates_[D2][D1]) == 0:
                continue            
            r = random.randint(0, len(swap_candidates_[D2][D1]) - 1)
            F2 = swap_candidates_[D2][D1][r]                        
            if(PREFERENCE_MATRIX[F2][D1] > MAX_PREF_ALLOWED):
               continue

        if(check_move(F1, F2, D2) == False):
           continue

        ass_cost_diff = calculate_assignment_cost_diff_with_move(F1, F2, D2)
        if(ass_cost_diff > assCostDiffTOL):
           continue
        acc_cost_diff = calculate_accounting_cost_diff_with_move(F1, F2, D2)
        cost_diff = ass_cost_diff + acc_cost_diff
        tol = 0
        if(random.randint(0, TOLRAND) == 1):            
            tol = TOL

        if(cost_diff <= tol and ass_cost_diff <= assCostDiffTOL):
            perform_move(F1, F2, D2)


def LocalSearch(timeLimit, maxIters):
    startTime = time.time()
    iter = 0
    global assCostDiffTOL
    assCostDiffTOL = 0
    best_assignment = {}
    for i in range(NMB_FAMILIES):
        best_assignment[i] = assignment_[i]
    bestCost = cost_

    while(iter < maxIters):
        if(time.time() - startTime >= timeLimit):
            break
        iter+=1
        if(iter % 5 == 0):    	 
           update_solution(best_assignment)

        local_search_shift_and_swap(20000)

        #print(cost_, bestCost)
        if(cost_ < bestCost):
            bestCost = cost_
            print(iter, bestCost, int(time.time() - startTime))
            for i in range(NMB_FAMILIES):
               best_assignment[i] = assignment_[i]

        assCostDiffTOL += 1

    update_solution(best_assignment)


#linear regression (3D plane fitting)
def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results

DELTA = 1
LINEARIZE = False

# calculate linear regression for each day ()
def linearize():
  resultABC = []
  for d in range(NMB_DAYS):
    resultABC.append([])

  for D in range(NMB_DAYS):
     x = []
     x.append([])
     x.append([])
     y = []
     LB = max(MIN_PEOPLE_PERDAY, nmb_people_assigned_to_day_[D] - DELTA)
     UB = min(MAX_PEOPLE_PERDAY, nmb_people_assigned_to_day_[D] + DELTA)
     dom1 = {LB}
     for i in range (LB + 1, UB + 1):
       dom1.add(i)

     if D < NMB_DAYS - 1:
        LB = max(MIN_PEOPLE_PERDAY, nmb_people_assigned_to_day_[D + 1] - DELTA)
        UB = min(MAX_PEOPLE_PERDAY, nmb_people_assigned_to_day_[D + 1] + DELTA)
        dom2 = {LB}
        for i in range (LB + 1, UB + 1):
          dom2.add(i)
        for i in dom1:
          for j in dom2:
            x[0].append(i)
            x[1].append(j)
            y.append(ACCOUNTING_COST_MATRIX[i, j])
     else:
        for i in dom1:
             x[0].append(i)
             x[1].append(i)
             y.append(ACCOUNTING_COST_MATRIX[i, i])

     result = reg_m(y,x)

     if(nmb_people_assigned_to_day_[D] == 125):
        result.params[0] = 0
        result.params[1] = 10000
        result.params[2] = -125 * 10000

     resultABC[D] = result.params
     p = resultABC[D][0]
     resultABC[D][0] = resultABC[D][1]
     resultABC[D][1] = p

  return resultABC

# mixed integer program to solve the problem
# if LINEARIZED is false then it basically solves a classical assignment problem
# othervise we also take into account the accounting cost by linearizing it
# only few possibilities are allowed for daily occupancy so linearization is possible
def MIP(timeLimit):
    
    # Create the mip solver with the CBC backend.
    solver = pywraplp.Solver('simple_mip_program', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    x = {}
    C = {}
    key = {} # key[i, j] is the index od variable that corresponds to the edge (i, j)
    K = 0
    for i in range(NMB_FAMILIES):
        for j in range(NMB_DAYS):
          if PREFERENCE_MATRIX[i, j] <= MAX_PREF_ALLOWED:
              key[i, j] = K
              x[K] = solver.BoolVar('x[%i,%i]' % (i, j))
              C[K] = ASSIGNMENT_COST_MATRIX[i, j]
              K = K + 1        
          else:
              key[i, j] = -1

    nmbvar = solver.NumVariables()
    print('Number of variables =', solver.NumVariables())

    # Constraints

    # exactely one day for each family
    for i in range(NMB_FAMILIES):       
        expr = 0
        for j in range(NMB_DAYS):
           if key[i, j] >= 0:
              expr = expr + x[key[i, j]]
        solver.Add(expr == 1)

    # occupancy
    occ = {}
    for i in range(NMB_DAYS):        
        occ[i] = solver.IntVar(0.0, 1000.0, 'occ[%i,%i]' % (i, 0))

    for i in range(NMB_DAYS):
        expr = 0
        for j in range(NMB_FAMILIES):
           if key[j, i] >= 0:
              expr += (x[key[j, i]] * FAMILY_SIZE[j])
        LB = MIN_PEOPLE_PERDAY
        UB = MAX_PEOPLE_PERDAY
        if LINEARIZE:
            LB = max(nmb_people_assigned_to_day_[i] - DELTA, MIN_PEOPLE_PERDAY)
            UB = min(nmb_people_assigned_to_day_[i] + DELTA, MAX_PEOPLE_PERDAY)
        solver.Add(expr >= LB)
        solver.Add(expr <= UB)
        solver.Add(occ[i] == expr)

    print('Number of constraints =', solver.NumConstraints())
   
    # Objective
    obj = solver.Sum([C[i] * x[i] for i in range(nmbvar)])
    
    #2nd objective - linearized
    if LINEARIZE == True:
        obj2 = 0
        ABC = linearize()
        for D in range(NMB_DAYS - 1):
           a = ABC[D][0]
           b = ABC[D][1]
           c = ABC[D][2]                      
           if D < NMB_DAYS - 1:
              obj2 = obj2 + (a * occ[D] + b * occ[D + 1] + c)
           else:
              obj2 = obj2 + (a * occ[D] + b * occ[D] + c)
        obj = obj + 1 * obj2
    
    solver.Minimize(obj)

    #params
    solver.SetTimeLimit(1000 * timeLimit) ## time limit
    
    #solving
    status = solver.Solve()

    print('MIP Objective value =', solver.Objective().Value())      
    print('Problem solved in %f milliseconds' % solver.wall_time())

    #update solution
    assign = {}
    for i in range(NMB_FAMILIES):
        for j in range(NMB_DAYS):
            if key[i, j] >= 0:
               if x[key[i, j]].solution_value() > 0.99:
                  assign[i] = j

    update_solution(assign)

#main

random.seed(52) 

data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')

FAMILY_SIZE = data.n_people.values
ASSIGNMENT_COST_MATRIX = GetAssignmentCostMatrix(data) # Preference cost matrix
ACCOUNTING_COST_MATRIX = GetAccountingCostMatrix()     # Accounting cost matrix
PREFERENCE_MATRIX = GetPreferenceMatrix(data)          # Preference Matrix
PREFERENCES_FOR_FAMILY = GetPreferenceForFamiliesMatrix(data) 

# preprocessing to reduce the number of possible assignments (roughly 50% of assignments is eliminated)
preprocessing()

startTime = time.time()

# MIP to minimize assignment cost (don't care about accounting cost)
# gives assignment cost of cca 43,700 and does not take much time, so 60 seconds is more than enough
# accounting cost is huge
MIP(60)
print(calculate_solution_cost())


# local search to improve solution (we can improve a lot since accounting cost is huge)
# search is very simple: only shift (reallocate) and swap moves are performed
# shift move changes assignment of a single family
# swap move exchanges assignments of two families
# moves that increase the cost are accepted ocasionally (implemented in a very simple way), but not very often - just not to be stack quickly
# in the early phase of the local search we try to not increase the assignment cost too much
# (this is done by imposing the treshold on assignment cost change, which increases gradually as the search progresses)
# implementation could be better, but in c++ it was quite ok
# arrives to 72k-74k, depending on the seed and running time
# other types of local search can improve the cost, but we try to keep it simple here, it is used just to build a reasonable starting point for 
# the MIP that will be called later
LocalSearch(3600, 2000)
print(calculate_solution_cost())

# MIP with linearization for accounting cost
# occupancy of each day has to be in [current_occupacy - DELTA, current_occupacy + DELTA]
# linearization of accounting cost is done by approximating it with linear regression 
# approximation is done for each pair of days (D, D + 1) by computing linear regression for 9 points (DELTA = 1)
# approximation seems to be quite good when DELTA = 1
# other possibility is to make better approximation with quadratic function and then use MIQP (tried this in C++ and improved a bit)
# smaller running time limit should also work (maybe 5 minutes)
DELTA = 1
LINEARIZE = True
MIP(5 * 60)
print(calculate_solution_cost())

# try to improve obtained result by Local Search again
LocalSearch(10 * 3600, 500)

best_assignment = {}
for i in range(NMB_FAMILIES):
    best_assignment[i] = assignment_[i]
bestCost = cost_

         
# MIP can be called again starting from the new solution
# might improve a bit more
# you can try more if you want
for i in range(5):
    MIP(5 * 60)
    print(calculate_solution_cost())
    LocalSearch(3600, 100)
    print(calculate_solution_cost())
    if(cost_ < bestCost):
        bestCost = cost_
        print("bestCost: ", bestCost, " time: ", time.time() - startTime)
        for i in range(NMB_FAMILIES):
           best_assignment[i] = assignment_[i]
       
update_solution(best_assignment)

print("Best Solution: ", bestCost)
print("LocalSearch...")

LocalSearch(3600, 1000)

print("Final Solution: ", cost_)

#write solution
with open('submission.csv', mode='w') as csv_file:
   fieldnames = ['family_id', 'assigned_day']
   writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
   writer.writeheader()
   for i in range(NMB_FAMILIES):
      writer.writerow({'family_id': i, 'assigned_day': assignment_[i] + 1})
