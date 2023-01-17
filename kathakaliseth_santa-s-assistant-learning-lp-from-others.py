from itertools import product

import random

import numpy as np

import pandas as pd

from numba import njit



import matplotlib.pylab as plt

#%pylab inline

import seaborn as sns

from ortools.linear_solver import pywraplp



#from numpy.core import fromnumeric
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





def GetPreferenceCostMatrix(data):

    cost_matrix = np.zeros((N_FAMILIES, N_DAYS), dtype=np.int64)

    for i in range(N_FAMILIES):

        desired = data.values[i, :-1]

        cost_matrix[i, :] = get_penalty(FAMILY_SIZE[i], 9)  #10

        for j, day in enumerate(desired):

            cost_matrix[i, day-1] = get_penalty(FAMILY_SIZE[i], j)

    return cost_matrix





def GetAccountingCostMatrix():

    ac = np.zeros((1000, 1000), dtype=np.float64)

    for n in range(ac.shape[0]):

        for n_p1 in range(ac.shape[1]):

            diff = abs(n - n_p1)

            ac[n, n_p1] = max(0, (n - 125) / 400 * n**(0.5 + diff / 50.0))

    return ac
penalty_values = []

for j in (2,3,4,5,6,7,8):

    for i in range(10):

        #print('Penalty for Choice', i ,'for ',j,'member family is: ' ,get_penalty(j, i))

        penalty_values = pd.DataFrame([i,j,get_penalty(j, i)] for j in (2,3,4,5,6,7,8)  for i in range(10))

penalty_values.columns = ['Choice', 'Family_Size', 'Penalty']        

print(penalty_values)
import matplotlib as mpl

label_size = 22

mpl.rcParams['xtick.labelsize'] = label_size 

mpl.rcParams['ytick.labelsize'] = label_size

# Generate a mask for the upper triangle

sns.set_style("whitegrid")



ax = plt.figure(figsize=(16,8))



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

 

ax = sns.lineplot(x="Family_Size", y="Penalty", data=penalty_values,linewidth=4,hue="Choice",color='#666666',

                  marker='o',markersize=14,palette="Oranges")



plt.title("Penalty \n", loc="center",size=32,color='#34495E',alpha=0.8)

plt.xlabel('Family Size',color='#34495E',fontsize=20) 

plt.ylabel('Penalty',color='#34495E',fontsize=20)

plt.xticks(size=15,color='#008abc',rotation='horizontal', wrap=True)

plt.yticks(size=15,color='#006600')

plt.ylim(0,2500)

plt.legend(loc='best', labels=['Choice 0','Choice 1','Choice 2','Choice 3','Choice 4','Choice 5','Choice 6','Choice 7',

                              'Choice 8','Choice 9'],

           handlelength=6,fontsize=10,ncol=2,framealpha=0.99)
del penalty_values
# cost_function, etc.



# preference cost

@njit(fastmath=True)

def pcost(prediction):

    daily_occupancy = np.zeros(N_DAYS+1, dtype=np.int64)

    penalty = 0

    for (i, p) in enumerate(prediction):

        n = FAMILY_SIZE[i]

        penalty += PCOSTM[i, p]

        daily_occupancy[p] += n

    return penalty, daily_occupancy





# accounting cost

@njit(fastmath=True)

def acost(daily_occupancy):

    accounting_cost = 0

    n_out_of_range = 0

    daily_occupancy[-1] = daily_occupancy[-2]

    for day in range(N_DAYS):

        n_p1 = daily_occupancy[day + 1]

        n    = daily_occupancy[day]

        n_out_of_range += (n > MAX_OCCUPANCY) or (n < MIN_OCCUPANCY)

        accounting_cost += ACOSTM[n, n_p1]

    return accounting_cost, n_out_of_range





@njit(fastmath=True)

def acostd(daily_occupancy):

    accounting_cost = np.zeros(N_DAYS, dtype=np.float64)

    n_out_of_range = 0

    daily_occupancy[-1] = daily_occupancy[-2]

    for day in range(N_DAYS):

        n_p1 = daily_occupancy[day + 1]

        n    = daily_occupancy[day]

        n_out_of_range += (n > MAX_OCCUPANCY) or (n < MIN_OCCUPANCY)

        accounting_cost[day] = ACOSTM[n, n_p1]

    return accounting_cost, n_out_of_range





@njit(fastmath=True)

def pcostd(prediction):

    daily_occupancy = np.zeros(N_DAYS+1, dtype=np.int64)

    penalty = np.empty_like(prediction)

    for (i, p) in enumerate(prediction):

        n = FAMILY_SIZE[i]

        penalty[i] = PCOSTM[i, p]

        daily_occupancy[p] += n

    return penalty, daily_occupancy





@njit(fastmath=True)

def cost_stats(prediction):

    penalty, daily_occupancy = pcostd(prediction)

    accounting_cost, n_out_of_range = acostd(daily_occupancy)

    return penalty, accounting_cost, n_out_of_range, daily_occupancy[:-1]





@njit(fastmath=True)

def cost_function(prediction):

    penalty, daily_occupancy = pcost(prediction)

    accounting_cost, n_out_of_range = acost(daily_occupancy)

    return penalty + accounting_cost + n_out_of_range*100000000
# fixMinOccupancy, fixMaxOccupancy + helpers



@njit(fastmath=True)

def cost_function_(prediction):

    penalty, daily_occupancy = pcost(prediction)

    accounting_cost, n_out_of_range = acost(daily_occupancy)

    return penalty + accounting_cost, n_out_of_range





@njit(fastmath=True)

def findAnotherDay4Fam(prediction, fam, occupancy):

    old_day = prediction[fam]

    best_cost = np.inf

    best_day = fam

    n = FAMILY_SIZE[fam]

    

    daysrange = list(range(0,old_day))+list(range(old_day+1,N_DAYS))

    for day in daysrange:

        prediction[fam] = day

        new_cost, _ = cost_function_(prediction)

        

        if (new_cost<best_cost) and (occupancy[day]+n<=MAX_OCCUPANCY):

            best_cost = new_cost

            best_day = day

            

    prediction[fam] = old_day

    return best_day, best_cost





@njit(fastmath=True)

def bestFamAdd(prediction, day, occupancy):

    best_cost = np.inf

    best_fam = prediction[day]

    for fam in np.where(prediction!=day)[0]:

        old_day = prediction[fam]

        prediction[fam] = day

        new_cost, _ = cost_function_(prediction)

        prediction[fam] = old_day

        n = FAMILY_SIZE[fam]

        if (new_cost<best_cost) and (occupancy[old_day]-n>=MIN_OCCUPANCY):

            best_cost = new_cost

            best_fam = fam   

    return best_fam





@njit(fastmath=True)

def bestFamRemoval(prediction, day, occupancy):

    best_cost = np.inf

    best_day = day

    

    for fam in np.where(prediction==day)[0]:

        new_day, new_cost = findAnotherDay4Fam(prediction, fam, occupancy)

        if new_cost<best_cost:

            best_cost = new_cost

            best_fam = fam

            best_day = new_day

            

    return best_fam, best_day





@njit(fastmath=True)

def fixMaxOccupancy(prediction):

    penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)



    for day in np.where(occupancy>MAX_OCCUPANCY)[0]:

        while occupancy[day]>MAX_OCCUPANCY:

            fam, new_day = bestFamRemoval(prediction, day, occupancy)

            prediction[fam] = new_day

            penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)

            

            

@njit(fastmath=True)

def fixMinOccupancy(prediction):

    penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)



    for day in np.where(occupancy<MIN_OCCUPANCY)[0]:

        while occupancy[day]<MIN_OCCUPANCY:

            fam = bestFamAdd(prediction, day, occupancy)

            prediction[fam] = day

            penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)
  # swappers



def findBetterDay4Family(pred):

    fobs = np.argsort(FAMILY_SIZE)

    score = cost_function(pred)

    original_score = np.inf

    

    while original_score>score:

        original_score = score

        for family_id in fobs:

            for pick in range(9):    #10

                day = DESIRED[family_id, pick]

                oldvalue = pred[family_id]

                pred[family_id] = day

                new_score = cost_function(pred)

                if new_score<score:

                    score = new_score

                else:

                    pred[family_id] = oldvalue



        print(score, end='\r')

    print(score)

    



def stochastic_product_search(top_k, fam_size, original, 

                              verbose=1000, verbose2=50000,

                              n_iter=500, random_state=2019):

    """

    original (np.array): The original day assignments.

    

    At every iterations, randomly sample fam_size families. Then, given their top_k

    choices, compute the Cartesian product of the families' choices, and compute the

    score for each of those top_k^fam_size products.

    """

    

    best = original.copy()

    print('best',best)

    best_score = cost_function(best)

    print('best_score',best_score)

    

    np.random.seed(random_state)

    print('np.random.seed(random_state)',np.random.seed(random_state))



    for i in range(n_iter):

        fam_indices = np.random.choice(range(DESIRED.shape[0]), size=fam_size)

        #print('product(*DESIRED[fam_indices, :top_k].tolist())',product(*DESIRED[fam_indices, :top_k].tolist()))

        changes = np.array(list(product(*DESIRED[fam_indices, :top_k].tolist())))



        for change in changes:

            new = best.copy()

            new[fam_indices] = change



            new_score = cost_function(new)



            if new_score < best_score:

                best_score = new_score

                best = new

                

        if verbose and i % verbose == 0:

            print(f"Iteration #{i}: Best score is {best_score:.2f}      ", end='\r')

            

        if verbose2 and i % verbose2 == 0:

            print(f"Iteration #{i}: Best score is {best_score:.2f}      ")

    

    print(f"Final best score is {best_score:.2f}")

    return best
def solveSantaLP():

    

    S = pywraplp.Solver('SolveAssignmentProblem', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)



    x = {}

    for i in range(N_FAMILIES):

        for j in range(N_DAYS):

            x[i, j] = S.BoolVar('x[%i,%i]' % (i, j))

            

            

    daily_occupancy = [S.Sum([x[i, j] * FAMILY_SIZE[i] for i in range(N_FAMILIES)])

                                                       for j in range(N_DAYS)]

    

    family_presence = [S.Sum([x[i, j] for j in range(N_DAYS)])

                                      for i in range(N_FAMILIES)]



    

    

    # Objective

    preference_cost = S.Sum([PCOSTM[i, j] * x[i,j] for i in range(N_FAMILIES)

                                                   for j in range(N_DAYS)])

    

    S.Minimize(preference_cost)



    

    

    # Constraints

    for j in range(N_DAYS-1):

        S.Add(daily_occupancy[j]   - daily_occupancy[j+1] <= 32)

        S.Add(daily_occupancy[j+1] - daily_occupancy[j]   <= 31)

                

    for i in range(N_FAMILIES):

        S.Add(family_presence[i] == 1)



    for j in range(N_DAYS):

        S.Add(daily_occupancy[j] >= MIN_OCCUPANCY)

        S.Add(daily_occupancy[j] <= MAX_OCCUPANCY)



        

    res = S.Solve()

                  

    resdict = {0:'OPTIMAL', 1:'FEASIBLE', 2:'INFEASIBLE', 3:'UNBOUNDED', 

               4:'ABNORMAL', 5:'MODEL_INVALID', 6:'NOT_SOLVED'}

    

    print('Result:', resdict[res])



    l = []

    for i in range(N_FAMILIES):

        for j in range(N_DAYS):

            s = x[i, j].solution_value()

            if s>0:

                l.append((i, j, s))



    df = pd.DataFrame(l, columns=['family_id', 'day', 'n'])



    

    if len(df)!=N_FAMILIES:

        df = df.sort_values(['family_id', 'n']).drop_duplicates('family_id', keep='last') 

     

    return df.day.values
N_DAYS = 100

N_FAMILIES = 5000

MAX_OCCUPANCY = 300

MIN_OCCUPANCY = 125



data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')



FAMILY_SIZE = data.n_people.values

DESIRED     = data.values[:, :-1] - 1

PCOSTM = GetPreferenceCostMatrix(data) # Preference cost matrix

ACOSTM = GetAccountingCostMatrix()     # Accounting cost matrix
data.head()
pivot_data = data.groupby(['choice_0'])['n_people'].agg(['sum'])

pivot_data = pd.DataFrame(pivot_data)

pivot_data.columns = ['Occupancy for all First Choices']
pivot_data
import matplotlib as mpl

label_size = 22

mpl.rcParams['xtick.labelsize'] = label_size 

mpl.rcParams['ytick.labelsize'] = label_size

# Generate a mask for the upper triangle

sns.set_style("whitegrid")



ax = plt.figure(figsize=(22,6))



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

 

ax = sns.lineplot(data=pivot_data,color='#666666',marker='o',markersize=10,linewidth=3,alpha=0.5)

ax.plot([0,125],[125,125],'--',color='darkgreen')

plt.text(90, 50, 'Mimimum Occupancy', fontsize=16,weight='bold',alpha=0.85,color='darkgreen')



ax.plot([0,300],[300,300],'--',color='red')

plt.text(90, 370, 'Maximum Occupancy', fontsize=16,weight='bold',alpha=0.85,color='red')

plt.title("If All were given the FIRST CHOICE \n", loc="center",size=32,color='#34495E',alpha=0.8)

plt.xlabel('Days before Christmas',color='#34495E',fontsize=20) 

plt.ylabel('Occupancy',color='#34495E',fontsize=20)

plt.xticks(size=15,color='#008abc',rotation='horizontal', wrap=True)

plt.yticks(size=15,color='#006600')

plt.xlim(0,100)

plt.legend(loc='best', fontsize=16,framealpha=0.99)
prediction = solveSantaLP()

penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)

print('{}, {:.2f}, ({}, {})'.format(penalty.sum(), 

                                    accounting_cost.sum(), 

                                    occupancy.min(), 

                                    occupancy.max()))
fixMinOccupancy(prediction)

fixMaxOccupancy(prediction)

penalty, accounting_cost, n_out_of_range, occupancy = cost_stats(prediction)

print('{}, {:.2f}, ({}, {})'.format(penalty.sum(), 

                                    accounting_cost.sum(), 

                                    occupancy.min(), 

                                    occupancy.max()))
round2 = stochastic_product_search(

        top_k=2,

        fam_size=6, 

        original=prediction, 

        n_iter=350000, #250000

        verbose=1000,

        verbose2=50000,

        random_state=2019

        )
round3 = stochastic_product_search(

        top_k=2,

        fam_size=7, 

        original=round2, 

        n_iter=350000, #250000

        verbose=1000,

        verbose2=50000,

        random_state=2019

        )
round4 = stochastic_product_search(

        top_k=2,

        fam_size=8, 

        original=round3, 

        n_iter=350000, #250000

        verbose=1000,

        verbose2=50000,

        random_state=2019

        )
round5 = stochastic_product_search(

        top_k=2,

        fam_size=9, 

        original=round4, 

        n_iter=350000, #250000

        verbose=1000,

        verbose2=50000,

        random_state=2019

        )
final = stochastic_product_search(

        top_k=2,

        fam_size=12, 

        original=round5, 

        n_iter=150000, #250000

        verbose=1000,

        verbose2=50000,

        random_state=2019

        )
sub = pd.DataFrame(range(N_FAMILIES), columns=['family_id'])

sub['assigned_day'] = final+1

sub.to_csv('submission_5.csv', index=False)
df0 = data[['choice_0','n_people']]

df0.columns = ['choice_0','n_people']

df0
df1 = pd.DataFrame([prediction]).transpose()

df1.columns = ['Prediction']

df1
df2 = pd.DataFrame([round2]).transpose()

df2.columns = ['Round2_Days']

df2
df3 = pd.DataFrame([round3]).transpose()

df3.columns = ['Round3_Days']

df3
df4 = pd.DataFrame([round4]).transpose()

df4.columns = ['Round4_Days']

df4
df5 = pd.DataFrame([round5]).transpose()

df5.columns = ['Round5_Days']

df5
df6 = pd.DataFrame([final]).transpose()

df6.columns = ['Final_Days']

df6
df = pd.concat([df0,df1,df2,df3,df4,df5,df6], axis=1)

df
pivot0 = df.groupby(['choice_0'])['n_people'].agg(['sum'])

pivot0 = pd.DataFrame(pivot0)

pivot0.columns = ['First Choice']



pivot1 = df.groupby(['Prediction'])['n_people'].agg(['sum'])

pivot1 = pd.DataFrame(pivot1)

pivot1.columns = ['Prediction']



pivot2 = df.groupby(['Round2_Days'])['n_people'].agg(['sum'])

pivot2 = pd.DataFrame(pivot2)

pivot2.columns = ['Round 2']



pivot3 = df.groupby(['Round3_Days'])['n_people'].agg(['sum'])

pivot3 = pd.DataFrame(pivot3)

pivot3.columns = ['Round 3']



pivot4 = df.groupby(['Round4_Days'])['n_people'].agg(['sum'])

pivot4 = pd.DataFrame(pivot4)

pivot4.columns = ['Round 4']



pivot5 = df.groupby(['Round5_Days'])['n_people'].agg(['sum'])

pivot5 = pd.DataFrame(pivot5)

pivot5.columns = ['Round 5']



pivot6 = df.groupby(['Final_Days'])['n_people'].agg(['sum'])

pivot6 = pd.DataFrame(pivot6)

pivot6.columns = ['Final']
#import matplotlib as mpl

label_size = 22

mpl.rcParams['xtick.labelsize'] = label_size 

mpl.rcParams['ytick.labelsize'] = label_size

# Generate a mask for the upper triangle

#sns.set_style("whitegrid")

 

sns.set_style('whitegrid', {'legend.frameon':True,'figure.facecolor': '#0b5e15',})



ax = plt.figure(figsize=(22,14))



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

 

ax = sns.lineplot(data=pivot0,color='#666666',palette="Blues",marker='o',markersize=10,linewidth=3,alpha=0.7)

ax = sns.lineplot(data=pivot1,color='#666666',palette="copper",marker='o',markersize=10,linewidth=3,alpha=0.7)

#ax = sns.lineplot(data=pivot2,color='#666666',palette="copper",marker='o',markersize=10,linewidth=3,alpha=0.7)

ax = sns.lineplot(data=pivot3,color='#666666',palette="Oranges",marker='o',markersize=10,linewidth=3,alpha=0.7)

#ax = sns.lineplot(data=pivot4,color='#666666',palette="Oranges",marker='o',markersize=10,linewidth=3,alpha=0.7)

ax = sns.lineplot(data=pivot5,color='#666666',palette="YlGn_r",marker='o',markersize=10,linewidth=3,alpha=0.7)

ax = sns.lineplot(data=pivot6,color='#666666',palette="Greens",marker='o',markersize=10,linewidth=3,alpha=0.7)

ax.plot([0,125],[125,125],'--',color='darkgreen')

plt.text(72, 105, 'Mimimum Occupancy', fontsize=20,weight='bold',alpha=0.85,color='darkgreen')



ax.plot([0,300],[300,300],'--',color='red')

plt.text(72, 315, 'Maximum Occupancy', fontsize=20,weight='bold',alpha=0.85,color='red')

plt.title("Santa's Workshop Attendence Optimization\n", loc="center",size=34,color='#ffffff',weight='bold',alpha=1)

plt.xlabel('Days before Christmas',color='#ffffff',fontsize=20) 

plt.ylabel('Occupancy',color='#ffffff',fontsize=20)

plt.xticks(size=15,color='#ffffff',rotation='horizontal', wrap=True)

plt.yticks(size=15,color='#ffffff')

plt.xlim(0,100)

plt.ylim(0,450)

plt.legend(fancybox=True, framealpha=0.5,loc='best', fontsize=20,ncol=5)