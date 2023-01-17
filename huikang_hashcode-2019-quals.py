%reset -sf

!ls /kaggle/input/hashcode/qualification_round_2019.in/qualification_round_2019.in
a_path = "/kaggle/input/hashcode/qualification_round_2019.in/qualification_round_2019.in/d_pet_pictures.txt"

b_path = "/kaggle/input/hashcode/qualification_round_2019.in/qualification_round_2019.in/b_lovely_landscapes.txt"

c_path = "/kaggle/input/hashcode/qualification_round_2019.in/qualification_round_2019.in/c_memorable_moments.txt"

d_path = "/kaggle/input/hashcode/qualification_round_2019.in/qualification_round_2019.in/a_example.txt"

e_path = "/kaggle/input/hashcode/qualification_round_2019.in/qualification_round_2019.in/e_shiny_selfies.txt"
import functools

import numpy as np

with open(b_path, "r") as f:

    lines = [w.strip() for w in f.readlines()]
aaa = [] # H/V

bbb = [] # number of tags

ccc = [] # tags

ddd = [] # index

for i,line in enumerate(lines[1:]):

    a,b,*c = line.split()

    aaa.append(a)

    bbb.append(b)

    ccc.append(set(c))

    ddd.append(i)

    if len(ccc) > 60000: 

        break
aa = []

bb = []

cc = []

dd = []



for a,b,c,d in zip(aaa,bbb,ccc,ddd):

    if a == "V": continue

    aa.append(a)

    bb.append(b)

    cc.append(c)

    dd.append([d])
# @functools.lru_cache(maxsize=10**8)

def score(loc1, loc2):

    sett1 = ccc[loc1]

    sett2 = ccc[loc2]

    return 1000-min(len(sett1 - sett2),

                    len(sett2 - sett1), 

                    len(sett2.intersection(sett1)))

# score.cache_info()
"""Simple travelling salesman problem between cities."""



from __future__ import print_function

from ortools.constraint_solver import routing_enums_pb2

from ortools.constraint_solver import pywrapcp





def create_data_model():

    """Stores the data for the problem."""

    data = {}

    data['num_vehicles'] = 1

    data['depot'] = 0

    return data





def print_solution(manager, routing, assignment, printing=True):

    """Prints assignment on console."""

    print('Objective: {} miles'.format(assignment.ObjectiveValue()))

    index = routing.Start(0)

    plan_output = 'Route for vehicle 0:\n'

    route_distance = 0

    result = []

    while not routing.IsEnd(index):

        plan_output += ' {} ->'.format(manager.IndexToNode(index))

        previous_index = index

        result.append(index)

        index = assignment.Value(routing.NextVar(index))

        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

    plan_output += ' {}\n'.format(manager.IndexToNode(index))

    if printing:

        print(plan_output)

    plan_output += 'Route distance: {}miles\n'.format(route_distance)

    return result, assignment.ObjectiveValue()

    



"""Entry point of the program."""

# Instantiate the data problem.

data = create_data_model()



# Create the routing index manager.

manager = pywrapcp.RoutingIndexManager(len(ccc),

                                       data['num_vehicles'], data['depot'])



# Create Routing Model.

routing = pywrapcp.RoutingModel(manager)



def distance_callback(from_index, to_index):

    """Returns the distance between the two nodes."""

    # Convert from routing variable Index to distance matrix NodeIndex.

    from_node = manager.IndexToNode(from_index)

    to_node = manager.IndexToNode(to_index)

    return score(from_node,to_node)



transit_callback_index = routing.RegisterTransitCallback(distance_callback)



# Define cost of each arc.

routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)



# Setting first solution heuristic.

search_parameters = pywrapcp.DefaultRoutingSearchParameters()

search_parameters.first_solution_strategy = (

    routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)

# search_parameters.local_search_metaheuristic = (

#     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
# search_parameters.time_limit.seconds = 30

# search_parameters.lns_time_limit.seconds = 10

search_parameters.solution_limit = 1
%%time

# Solve the problem.

assignment = routing.SolveWithParameters(search_parameters)
# Print solution on console.

if assignment:

    result, objective = print_solution(manager, routing, assignment, printing=False)
# score.cache_info()
# 0 ROUTING_NOT_SOLVED: Problem not solved yet.

# 1 ROUTING_SUCCESS: Problem solved successfully.

# 2 ROUTING_FAIL: No solution found to the problem.

# 3 ROUTING_FAIL_TIMEOUT: Time limit reached before finding a solution.

# 4 ROUTING_INVALID: Model, model parameters, or flags are not valid.

print("Solver status: ", routing.status())
result

len(result)*1000 - objective
import matplotlib.pyplot as plt

plt.figure(figsize=(14,3))

plt.plot(result)

plt.show()
with open("b.out", "w") as f:

    f.write(str(len(result)))

    f.write("\n")

    f.write("\n".join([str(" ".join(str(x) for x in dd[s])) for s in result]))
!head b.out