import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



def get_days(assigned_days, n_people):

    days = np.zeros(assigned_days.max(), int)

    for i, r in enumerate(assigned_days):

        days[r-1] += n_people[i]

    return days

def dataset_plot(dataset):

    desired = dataset.values[:,1:11]

    n_people = dataset.values[:,11]

    n_days = desired.max()

    plt.rcParams['figure.figsize'] = [20, 5]

    for i in range(10):

        plt.plot(np.arange(n_days), get_days(desired[:,i], n_people), alpha=0.9, label=f'choice {i}')

    days = np.zeros(desired.max(), int)

    for i, r in enumerate(desired):

        for j in range(10):

            days[r[j]-1] += n_people[i]

    plt.bar(np.arange(n_days), days/10, label='mean choices')

    plt.ylim(0, 1100)

    plt.legend()

    plt.show()
def example_ortools(desired, n_people, has_accounting=True):

    from ortools.linear_solver import pywraplp

    def accounting_penalty(day, next_day):

        return (day-125)*(day**(0.5 + abs(day-next_day)/50.0))/400

    NUM_THREADS = 4

    NUM_SECONDS = 3600

    FAMILY_COST = np.asarray([0,50,50,100,200,200,300,300,400,500])

    MEMBER_COST = np.asarray([0, 0, 9,  9,  9, 18, 18, 36, 36,235])

    num_days = desired.max()

    num_families = desired.shape[0]

    solver = pywraplp.Solver('Santa2019', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

#     solver.set_time_limit(NUM_SECONDS*NUM_THREADS*1000)

    solver.SetNumThreads(NUM_THREADS)

    C, B, I = {}, {}, {}

    for fid, choices in enumerate(desired):

        for cid in range(10):

            B[fid, choices[cid]-1] = solver.BoolVar('')

            C[fid, choices[cid]-1] = FAMILY_COST[cid] + n_people[fid] * MEMBER_COST[cid]



    for day in range(num_days):

        I[day] = solver.IntVar(125, 300, f'I{day}')

        solver.Add(solver.Sum(n_people[fid]*B[fid, day] for fid in range(num_families) if (fid,day) in B) == I[day])



    for fid in range(num_families):

        solver.Add(solver.Sum(B[fid, day] for day in range(num_days) if (fid,day) in B) == 1)



    objective = solver.Sum(C[fid, day]*B[fid, day] for fid, day in B)

    if has_accounting:

        Y = {}



        for day in range(num_days):

            next_day = np.clip(day+1, 0, num_days-1)

            gen = [(u,v) for v in range(176) for u in range(176)]

            for u,v in gen:

                Y[day,u,v] = solver.BoolVar('')

            solver.Add(solver.Sum(Y[day,u,v]*u for u,v in gen) == I[day]-125)

            solver.Add(solver.Sum(Y[day,u,v]*v for u,v in gen) == I[next_day]-125)

            solver.Add(solver.Sum(Y[day,u,v]   for u,v in gen) == 1)

            

        accounting_penalties = solver.Sum(accounting_penalty(u+125,v+125) * Y[day,u,v] for day,u,v in Y)

        objective += accounting_penalties



    solver.Minimize(objective)

    sol = solver.Solve()

    status = ['OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNBOUNDED', 'ABNORMAL', 'MODEL_INVALID', 'NOT_SOLVED']

    if status[sol] == 'OPTIMAL':

        print("Result: ", objective.solution_value())

        assigned_days = np.zeros(num_families, int)

        for fid, day in B:

            if B[fid, day].solution_value() > 0.5:

                assigned_days[fid] = day + 1

        return assigned_days
def example_cplex(desired, n_people, has_accounting=True): # can't run on kaggle notebooks 

    def accounting_penalty(day, next_day):

        return (day-125)*(day**(0.5 + abs(day-next_day)/50.0))/400



    from docplex.mp.model import Model

    FAMILY_COST = np.asarray([0,50,50,100,200,200,300,300,400,500])

    MEMBER_COST = np.asarray([0, 0, 9,  9,  9, 18, 18, 36, 36,235])

    num_days = desired.max()

    num_families = desired.shape[0]

    solver = Model(name='Santa2019')

    solver.parameters.mip.tolerances.mipgap = 0.00

    solver.parameters.mip.tolerances.absmipgap = 0.00

    C = {}

    for fid, choices in enumerate(desired):

        for cid in range(10):

            C[fid, choices[cid]-1] = FAMILY_COST[cid] + n_people[fid] * MEMBER_COST[cid]



    B = solver.binary_var_dict(C, name='B')

    I = solver.integer_var_list(num_days, lb=125, ub=300, name='I')



    for day in range(num_days):

        solver.add(solver.sum(n_people[fid]*B[fid, day] for fid in range(num_families) if (fid,day) in B) == I[day])



    for fid in range(num_families):

        solver.add(solver.sum(B[fid, day] for day in range(num_days) if (fid,day) in B) == 1)



    preference_cost = solver.sum(C[fid, day]*B[fid, day] for fid, day in B)

    if has_accounting:

        Y = solver.binary_var_cube(num_days, 176, 176, name='Y')



        for day in range(num_days):

            next_day = np.clip(day+1, 0, num_days-1)

            gen = [(u,v) for v in range(176) for u in range(176)]

            solver.add(solver.sum(Y[day,u,v]*u for u,v in gen) == I[day]-125)

            solver.add(solver.sum(Y[day,u,v]*v for u,v in gen) == I[next_day]-125)

            solver.add(solver.sum(Y[day,u,v]   for u,v in gen) == 1)

            

        gen = [(day,u,v) for day in range(num_days) for v in range(176) for u in range(176)]

        accounting_penalties = solver.sum(accounting_penalty(u+125,v+125) * Y[day,u,v] for day,u,v in gen)

        solver.minimize(accounting_penalties+preference_cost)

    else:

        solver.minimize(preference_cost)



    solver.print_information()

    sol = solver.solve(log_output=True)

    if sol:

        print(sol.objective_value)

        assigned_days = np.zeros(num_families, int)

        for fid, day in C:

            if sol[B[fid, day]] > 0:

                assigned_days[fid] = day + 1

        return assigned_days
for n_days in [10,20,40,80]:

    dataset_plot(pd.read_csv(f'/kaggle/input/santa-2019-{n_days}-uniform-days/family_data.csv'))

    dataset_plot(pd.read_csv(f'/kaggle/input/santa-2019-{n_days}-days/family_data.csv'))

#     dataset_plot(pd.read_csv(f'/kaggle/input/santa-revenge-of-the-accountants-{n_days}-uniform-days/family_data.csv'))

#     dataset_plot(pd.read_csv(f'/kaggle/input/santa-2019-revenge-of-the-accountants-{n_days}-days/family_data.csv'))
for n_days in [10,20,40,80]:

    ds = pd.read_csv(f'/kaggle/input/santa-2019-{n_days}-uniform-days/family_data.csv')

    %time example_ortools(ds.values[:,1:11], ds.values[:,11], False)
columns = ['competition', 'distribution', 'number days', 'optimum', 'Cplex(seconds)', 'OR-Tools(seconds)']

pd.DataFrame([

    ['Santa 2019', 'uniform', 10,   0, 0, 0],

    ['Santa 2019', 'uniform', 20,   0, 0, 1],

    ['Santa 2019', 'uniform', 40, 150, 2, 3],

    ['Santa 2019', 'uniform', 80,   0, 3, 4],



    ['Santa 2019', '', 10,  5148,  2,    1],

    ['Santa 2019', '', 20,  8481,  2, 1317],

    ['Santa 2019', '', 40, 12090,  3, 'Too long'],

    ['Santa 2019', '', 80, 32806, 15, 'Too long'],



    ['Santa Revenge', 'uniform', 10,   0, 0, 0],

    ['Santa Revenge', 'uniform', 20, 100, 1, 3],

    ['Santa Revenge', 'uniform', 40, 150, 3, 3],

    ['Santa Revenge', 'uniform', 80, 850, 5, 'Too long'],



    ['Santa Revenge', '', 10,  9206,  0, 'Too long'],

    ['Santa Revenge', '', 20, 14627,  3, 'Too long'],

    ['Santa Revenge', '', 40, 22565, 26, 'Too long'],

    ['Santa Revenge', '', 80, 55212, 44, 'Too long'],

], columns=columns)
pd.DataFrame([

    ['Santa 2019', 'uniform', 10, 251.5312,  86, 'Too long'],

    ['Santa 2019', 'uniform', 20, 992.2934, 761, 'Too long'],

    ['Santa 2019', 'uniform', 40, None, 'Too long', 'Too long'],

    ['Santa 2019', 'uniform', 80, None, 'Too long', 'Too long'],



    ['Santa 2019', '', 10,  7902.8137, 1960, 'Too long'],

    ['Santa 2019', '', 20, 12794.1957, 4504, 'Too long'],

    ['Santa 2019', '', 40, None, 'Too long', 'Too long'],

    ['Santa 2019', '', 80, None, 'Too long', 'Too long'],

], columns=columns)