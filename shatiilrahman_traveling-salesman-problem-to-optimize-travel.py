"""Simple travelling salesman problem between cities."""



from __future__ import print_function

from ortools.constraint_solver import routing_enums_pb2

from ortools.constraint_solver import pywrapcp







def create_data_model():

    """Stores the data for the problem."""

    data = {}

    data['distance_matrix'] =[

    [  0, 290, 250,  230,  190,  334, 365,   40], # Dhaka

    [290,   0, 337,  453,  396,  560, 581,  244], # Syhlet

    [250, 337,   0,  495,  396,  540, 120,  240], # Chittagonj

    [230, 453, 495,    0,  360,  150, 595,  242], # Rajshahi

    [190, 396, 396,  360,    0,  356, 496,  253], # Jossore

    [334, 560, 540,  150,  356,    0, 674,  275], # Dinajpur

    [365, 581, 120,  595,  496,  674,   0,  397], # Coxsbazar

    [40,  244, 240,  242,  253,  275, 397,    0]] # Narsingdi

# distance between Dhaka to Syhlet is 290kms and so on

    data['num_vehicles'] = 1

    data['depot'] = 0

    return data





def print_solution(manager, routing, assignment):

    """Prints assignment on console."""

    print('Objective Distance: {} miles'.format(assignment.ObjectiveValue()))

    index = routing.Start(0)

    plan_output = 'Route for vehicle 0:\n'

    route_distance = 0

    while not routing.IsEnd(index):

        plan_output += ' {} ->'.format(manager.IndexToNode(index))

        previous_index = index

        index = assignment.Value(routing.NextVar(index))

        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

    plan_output += ' {}\n'.format(manager.IndexToNode(index))

    print(plan_output)

    plan_output += 'Route distance: {}miles\n'.format(route_distance)





def main():

    """Entry point of the program."""

    # Instantiate the data problem.

    data = create_data_model()



    # Create the routing index manager.

    manager = pywrapcp.RoutingIndexManager(

        len(data['distance_matrix']), data['num_vehicles'], data['depot'])



    # Create Routing Model.

    routing = pywrapcp.RoutingModel(manager)





    def distance_callback(from_index, to_index):

        """Returns the distance between the two nodes."""

        # Convert from routing variable Index to distance matrix NodeIndex.

        from_node = manager.IndexToNode(from_index)

        to_node = manager.IndexToNode(to_index)

        return data['distance_matrix'][from_node][to_node]



    transit_callback_index = routing.RegisterTransitCallback(distance_callback)



    # Define cost of each arc.

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)



    # Setting first solution heuristic.

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (

        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)



    # Solve the problem.

    assignment = routing.SolveWithParameters(search_parameters)



    # Print solution on console.

    if assignment:

        print_solution(manager, routing, assignment)





if __name__ == '__main__':

    main()





# Source code: https://developers.google.com/optimization/routing/tsp