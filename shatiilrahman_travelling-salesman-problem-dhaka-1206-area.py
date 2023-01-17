import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
df = pd.read_excel("../input/DistanceMatrixDhaka-1206.xlsx")

print(df)
df = df.drop(columns='column/row')

df = df.values

print(df)












from __future__ import print_function

from ortools.constraint_solver import routing_enums_pb2

from ortools.constraint_solver import pywrapcp

def create_data_model():

    """Stores the data for the problem."""

    data = {}

    data['distance_matrix'] = df

    data['num_vehicles'] = 1

    data['depot'] = 0

    return data
def print_solution(manager, routing, assignment):

    city = [0,1,2,3,4,5]

    k=0

    city_names = ["balughat","manikdi","hazimarket","shagufta","MES","matikata"]

    print('Objective: {} kilometers'.format(assignment.ObjectiveValue()))

    index = routing.Start(0)

    plan_output = 'Route for vehicle 0:\n'

    route_distance = 0

    while not routing.IsEnd(index):

        plan_output += ' {} ->'.format(manager.IndexToNode(index))

        previous_index = index

        city[k]=index

        k=k+1

        index = assignment.Value(routing.NextVar(index))

        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

    plan_output += ' {}\n'.format(manager.IndexToNode(index))

    

    print(plan_output)

    plan_output += 'Route distance: {}kilometers\n'.format(route_distance) 

    for i in range (0,6):

        if (city[i] == 0):

            print('balughat',end=' -> ')

        elif (city[i] == 1 ):

            print('manikdi',end=' -> ')

        elif (city[i] == 2 ):

            print('hazimarket',end=' -> ')

        elif (city[i] == 3 ):

            print('shagufta',end=' -> ')

        elif (city[i] == 4 ):

            print('MES',end=' -> ')

        elif (city[i]==5):

            print('matikata',end=' -> ')

    print('balughat')
def main():

    

    data = create_data_model()

    manager = pywrapcp.RoutingIndexManager(

        len(data['distance_matrix']), data['num_vehicles'], data['depot'])

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):

        """Returns the distance between the two nodes."""

        from_node = manager.IndexToNode(from_index)

        to_node = manager.IndexToNode(to_index)

        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (

        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    assignment = routing.SolveWithParameters(search_parameters)

    if assignment:

        print_solution(manager, routing, assignment)



if __name__ == '__main__':

    main()

# Source code: https://developers.google.com/optimization/routing/tsp


























