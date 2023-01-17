!pip install pulp
import numpy as np

import pandas as pd

from pulp import *

from timeit import default_timer as timer

# Read data

data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')
##Cost function thanks to: https://www.kaggle.com/jesperdramsch/intro-to-santa-s-2019-viz-costs-22-s-and-search##

# Constants

N_FAM = len(data)

N_DAY = 100

MAX_OCC = 300

MIN_OCC = 125



# Choice cost dictionary

cost_dict = {0:  [  0,  0],

             1:  [ 50,  0],

             2:  [ 50,  9],

             3:  [100,  9],

             4:  [200,  9],

             5:  [200, 18],

             6:  [300, 18],

             7:  [300, 36],

             8:  [400, 36],

             9:  [500, 36 + 199],

             10: [500, 36 + 398],

            }



# Calculate cost for choice and family

def cost(choice, members, cost_dict):

    x = cost_dict[choice]

    return x[0] + members * x[1]



all_costs = {k: pd.Series([cost(k, x, cost_dict) for x in range(2,9)], index=range(2,9)) for k in cost_dict.keys()}



family_sizes = data.n_people.values.astype(np.int8)

family_cost_matrix = np.zeros((N_DAY,N_FAM)) # Cost for each family for each day



for i, el in enumerate(family_sizes):

    family_cost_matrix[:, i] += all_costs[10][el] # populate each day with the max cost

    for j, choice in enumerate(data.drop("n_people",axis=1).values[i,:]):

        family_cost_matrix[choice-1, i] = all_costs[j][el] # fill wishes into

        

def accounting_penalty_day(occupancy_day, occupancy_next_day):

    return max(0,((occupancy_day-MIN_OCC)/400)*occupancy_day**(0.5 + (abs(occupancy_day-occupancy_next_day))/50))



# Define matrix of accounting cost for assignments within constraints

accounting_matrix = np.zeros([MAX_OCC-MIN_OCC+1,MAX_OCC-MIN_OCC+1])

for i, x in enumerate(range(MIN_OCC,MAX_OCC+1)):

    for j, y in enumerate(range(MIN_OCC,MAX_OCC+1)):

        accounting_matrix[i,j] = accounting_penalty_day(x,y)
# Cost calculation function

def calculate_penalty(assignments, fam_sizes=family_sizes, 

                      fam_cost_matrix=family_cost_matrix, account_matrix=accounting_matrix):

    penalty = 0

    # Add one entry to occupancy matrix for day after last day

    daily_occupancy = np.zeros(N_DAY+1, dtype=np.int16)

    for index, (day, fam_size) in enumerate(zip(assignments, fam_sizes)):

        # Index of assignment day is one lower

        day_index = day-1

        # Add family size to daily occupancy for assignment day

        daily_occupancy[day_index] += fam_size

        penalty += fam_cost_matrix[day_index,index]

        

    daily_occupancy[-1] = daily_occupancy[-2]

    # Calculate accounting penalty, hard capacity constraints relaxed

    for day in range(N_DAY):

        n = daily_occupancy[day]

        n_next = daily_occupancy[day+1]

        violation = n < MIN_OCC or n > MAX_OCC

        if violation: 

            penalty += 1e11

        elif MIN_OCC <= n_next <= MAX_OCC:

            penalty += account_matrix[n-MIN_OCC, n_next-MIN_OCC]

        

    return penalty
def retrieve_indices(day_or_fam_value, value_is_fam_bool):

    indices = []

    for index, v in enumerate(x):

        var_name = v.name

        var_name_split = var_name.split('_')

        if value_is_fam_bool:

            value = int(var_name_split[1])

        else:

            # Minus 1 since day text one higher than index

            value = int(var_name_split[2]) - 1

            

        if value == day_or_fam_value:

            indices.append(index)

                

    return indices

    

##Altered integer programming (IP) representation of the problem##

santa_prob = LpProblem("SantaWorkshopTour",LpMinimize)



# Create array of decision variables and dictionary of costs

x = []

match_fam_sizes = []

cost_dict = {}

for fam in range(N_FAM):

    day_index_array = np.argsort(family_cost_matrix[:,fam])

    # Only first five choices

    for day in day_index_array[0:4]:

        # Binary variables, add one to day index to get true assignment day in name

        x.append(LpVariable(f'x_{fam}_{day+1}',0,1,LpInteger))

        match_fam_sizes.append(family_sizes[fam])

        # Add assignment choice cost to dictionary

        cost_dict[x[-1]] = family_cost_matrix[day,fam]



# Additional variables to have occupancy dependent constraints

z = []

# Binary variables for all but last day

for day in range(N_DAY):

    z.append(LpVariable(f'z_{day+1}',0,1,LpInteger))

        

# LP model objective function, ignoring accounting costs

santa_prob += lpSum([cost_dict[value]*x[index] for index, value in enumerate(x)])



# Add capacity constraints per day

M = MAX_OCC-MIN_OCC

for day in range(N_DAY):

    fam_indices = retrieve_indices(day, False)

    santa_prob += lpSum([match_fam_sizes[fam_index]*x[fam_index] for fam_index in fam_indices]) >= MIN_OCC

    santa_prob += lpSum([match_fam_sizes[fam_index]*x[fam_index] for fam_index in fam_indices]) - M*z[day] <= MIN_OCC

    

# Add constraint of maximum 1 assignment per family

for fam in range(N_FAM):

    day_indices = retrieve_indices(fam, True)

    santa_prob += lpSum([x[day_index] for day_index in day_indices]) == 1

    

# Add additional constraints regarding assignment difference; compensating for leaving out accounting costs

max_diff = 175

correction = 120

alpha = 0.2

# Define rhs value

rhs = max_diff + alpha*MIN_OCC

for day in range(N_DAY-1):

    fam_indices = retrieve_indices(day, False)

    fam_indices_plus = retrieve_indices(day+1, False)

    santa_prob += (lpSum([match_fam_sizes[fam_index]*(1+alpha)*x[fam_index] for fam_index in fam_indices])

                   -lpSum([match_fam_sizes[fam_index]*x[fam_index] for fam_index in fam_indices_plus])

                   + correction*z[day] <= rhs)

    santa_prob += (lpSum([match_fam_sizes[fam_index]*x[fam_index] for fam_index in fam_indices_plus])

                   -lpSum([match_fam_sizes[fam_index]*(1-alpha)*x[fam_index] for fam_index in fam_indices])

                   + correction*z[day] <= rhs)
# Solve the problem using PuLP's default solver, time limit in seconds

time_lim = 10000

messaging = 1

gap = 0

santa_prob.solve(pulp.PULP_CBC_CMD(maxSeconds=time_lim, msg=messaging, fracGap=gap))



# Print solution status and objective value

print('Status: ' + str(LpStatus[santa_prob.status]) + ', Value: ' + str(value(santa_prob.objective)))



# Define empty solution and fill with solver result

solution_df = pd.DataFrame(np.zeros(N_FAM), columns = ['assigned_day'], dtype=np.int8)

for index, v in enumerate(santa_prob.variables()):

    var_name = v.name

    if v.varValue == 1 and 'x' in var_name:

        var_name_split = var_name.split('_')

        fam = int(var_name_split[1])

        day = int(var_name_split[2])

        solution_df.iat[fam,0] = day

        

# Calculate solution penalty and write result to file

solution = solution_df['assigned_day']

solution_penalty = calculate_penalty(solution)

solution_df.to_csv(f'SantaPuLP_{int(solution_penalty)}.csv', index_label='family_id')
## Local Search##

# Function to perform local search on initial solution, within certain time limit

def local_search(best_solution, best_penalty, time_limit):

    start = timer()

    time_since_start = 0

    

    # Initialize additional solutions for reference in local search

    current_solution = best_solution.copy()

    current_penalty = best_penalty

    new_solution = best_solution.copy()

    new_penalty = best_penalty

    

    # Start by looping over first choices families

    choice_index = 0



    while time_since_start < time_limit:

        # Keep track of whether improvement was made

        improvement = False

        

        # Order families randomly

        random_permutation = np.random.permutation(N_FAM)

        for fam in random_permutation:

            # Calculate time since start

            time_since_start = round((timer()-start))

            if time_since_start > time_limit:

                break

                        

            # If choice index already assigned, continue

            if current_solution[fam] == data.iat[fam,choice_index]:

                continue

                

            # Make new solution by assigning choice_index to fam

            new_solution = current_solution.copy()

            new_solution[fam] = data.iat[fam,choice_index]

            # Recalculate penalty

            new_penalty = calculate_penalty(new_solution)

            # Change solution and penalty values depending on result

            if new_penalty < current_penalty:

                current_solution = new_solution.copy()

                current_penalty = new_penalty

                if new_penalty < best_penalty:

                    best_solution = new_solution.copy()

                    best_penalty = new_penalty

                    improvement = True

                    

        # Start with first choices again if improvement made

        if improvement:

            choice_index = 0

        else:

            choice_index += 1

            # If all choices have been looped, return

            if choice_index == 10:

                return best_solution

            

    return best_solution
# Perform local search on IP solution

ls_time_limit = 900

final_solution = local_search(solution, solution_penalty, ls_time_limit)

# Recalculate penalty to be sure

final_solution_penalty = calculate_penalty(final_solution)



# Write final solution to file, optionally use solution if local search returned error

final_solution_df = pd.DataFrame(final_solution, columns=['assigned_day'])

final_solution_df.to_csv(f'SantaPuLP_{int(final_solution_penalty)}.csv', index_label='family_id')