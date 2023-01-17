from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
def create_data_model():
    data = {}
    # Locations in block units
    _locations = [(4, 4), # depot
           (2, 0), (8, 0), # locations to visit
           (0, 1), (1, 1),
           (5, 2), (7, 2),
           (3, 3), (6, 3),
           (5, 5), (8, 5),
           (1, 6), (2, 6),
           (3, 7), (6, 7),
           (0, 8), (7, 8)]
    data["locations"] = [(l[0] * 114, l[1] * 80) for l in _locations]
    data["num_locations"] = len(data["locations"])
    data["num_vehicles"] = 4
    data["depot"] = 0
    return data
def manhattan_distance(position_1, position_2):
    return (abs(position_1[0] - position_2[0]) + abs(position_1[1] - position_2[1]))

def create_distance_callback(data):
    _distances = {}

    for from_node in range(data["num_locations"]):
        _distances[from_node] = {}
        for to_node in range(data["num_locations"]):
            if from_node == to_node:
                _distances[from_node][to_node] = 0
            else:
                _distances[from_node][to_node] = (
            manhattan_distance(data["locations"][from_node],
                               data["locations"][to_node]))
    
    def distance_callback(from_node, to_node):
        return _distances[from_node][to_node]

    return distance_callback
def add_distance_dimension(routing, distance_callback):
    distance = 'Distance'
    maximum_distance = 3000  # Maximum distance per vehicle.
    routing.AddDimension(
      distance_callback,
      0,  # null slack
      maximum_distance,
      True,  # start cumul to zero
      distance)
    distance_dimension = routing.GetDimensionOrDie(distance)
    # Try to minimize the max distance among vehicles.
    distance_dimension.SetGlobalSpanCostCoefficient(100)
def print_solution(data, routing, assignment):
    total_distance = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} ->'.format(routing.IndexToNode(index))
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        plan_output += ' {}\n'.format(routing.IndexToNode(index))
        plan_output += 'Distance of route: {}m\n'.format(distance)
        print(plan_output)
        total_distance += distance
    print('Total distance of all routes: {}m'.format(total_distance))
data = create_data_model()
  # Create Routing Model
data
routing = pywrapcp.RoutingModel(
      data["num_locations"],
      data["num_vehicles"],
      data["depot"])
  # Define weight of each edge
distance_callback = create_distance_callback(data)
routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)

add_distance_dimension(routing, distance_callback)
  # Setting first solution heuristic (cheapest addition).
search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
search_parameters.first_solution_strategy = (
      routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC) # pylint: disable=no-member
  # Solve the problem.
assignment = routing.SolveWithParameters(search_parameters)
if assignment:
    print_solution(data, routing, assignment)
