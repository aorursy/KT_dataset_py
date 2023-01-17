# modules

from __future__ import print_function

import pandas as pd

import numpy as np

from math import radians, sin, cos, atan2, sqrt

import folium

from ortools.constraint_solver import routing_enums_pb2

from ortools.constraint_solver import pywrapcp
# import the data

df = pd.read_csv('../input/meteorite-landings/meteorite_landings_prod.csv')

print('Data Imported!')
# tidy up the data

df.dropna(inplace=True)

df.drop(['Unnamed: 0','nametype','lat_lng'], axis = 1, inplace = True)

df = df.loc[(df['latitude']!=0)&(df['longitude']!=0),:].copy()

df.shape
# set up the map

m = folium.Map(location=[39.0, -100.0],width = '100%',height='100%',min_zoom = 2.5,zoom_start=4.3)



# masks to filter data

cand_mask = (df['mass']>=30000)&(df['mass']<=50000)&(df['location_code']=='US')

other_mask = ((df['mass']<=30000)|(df['mass']>=50000))&(df['location_code']=='US')&(df['year']>=2000)





# candidates in BLUE

for _, row in df.loc[cand_mask,:].iterrows():

    folium.CircleMarker([row['latitude'],row['longitude']],radius=2.5,color='blue',fill_color = 'blue').add_to(m)



# unsuitable meteorites in RED

for _, row in df.loc[other_mask,:].iterrows():

    folium.CircleMarker([row['latitude'],row['longitude']],radius=2.5,color='red',fill_color = 'red').add_to(m)



# then Area 51 base in PINK

folium.CircleMarker([37.23,-115.81],radius=5,color='magenta',fill_color = 'magenta').add_to(m)



# display the chart

m
# 1. distance between points



def haversine_distance(lat1,long1,lat2,long2):

    

    # approximate radius of earth in km

    R = 6373.0

    

    # convert degrees to radians

    lat1  = radians(abs(lat1))

    long1 = radians(abs(long1))

    lat2  = radians(abs(lat2))

    long2 = radians(abs(long2))

    

    # calculate change

    dlong = long2 - long1

    dlat  = lat2 - lat1

    

    # Haversine formula for calculating distance between points on a sphere

    a    = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlong / 2)**2

    c    = 2 * atan2(sqrt(a), sqrt(1 - a))

    dist = round(R*c/1.6,0)

    

    return dist
# 2. function to create the distance matrix



def create_distance_matrix(lat,long):

    # add some exception handling

    n = len(lat)

        

    if len(long) == n:

        X = np.zeros((n,n))

        for i in range(n):

            for j in range(n):

                X[i,j] = haversine_distance(lat[i],long[i],lat[j],long[j])

        return X

        

    else:

        print('Check variable lengths')
# get data

area_51 = pd.DataFrame({'id':51,

                        'meteorite_name':'NA',

                        'recclass':'NA',

                        'fall':'NA',

                        'year':-1,

                        'mass':0,

                        'location':'United States',

                        'location_code':'US',

                        'location_type':'land',

                        'distance':0,

                        'latitude':37.23333333,

                        'longitude':-115.80833333},index=[0])

temp = df.loc[(df['mass']>=30000)&(df['mass']<=50000)&(df['location_code']=='US'),df.columns != 'geometry'].copy().reset_index(drop=True)

temp = pd.concat([area_51,temp],axis=0,ignore_index=True)





# data model

def create_data_model():

    """Stores the data for the problem."""

    data = {}

    data['distance_matrix'] = create_distance_matrix(temp['latitude'],temp['longitude'])

    data['demands'] = temp['mass']

    data['vehicle_capacities'] = [200000,200000,200000,200000]

    data['num_vehicles'] = 4

    data['depot'] = 0

    

    return data





# solution printer

def print_solution(data, manager, routing, assignment):

    """Prints assignment on console."""

    

    # Display dropped nodes.

    dropped_nodes = 'Dropped nodes:'

    for node in range(routing.Size()):

        if routing.IsStart(node) or routing.IsEnd(node):

            continue

        if assignment.Value(routing.NextVar(node)) == node:

            dropped_nodes += ' {}'.format(manager.IndexToNode(node))

    print(dropped_nodes)

    

    # Display routes

    total_distance = 0

    total_load = 0

    for vehicle_id in range(data['num_vehicles']):

        index = routing.Start(vehicle_id)

        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)

        route_distance = 0

        route_load = 0

        while not routing.IsEnd(index):

            node_index = manager.IndexToNode(index)

            route_load += data['demands'][node_index]

            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)

            previous_index = index

            index = assignment.Value(routing.NextVar(index))

            route_distance += routing.GetArcCostForVehicle(

                previous_index, index, vehicle_id)

        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),

                                                 route_load)

        plan_output += 'Distance of the route: {}m\n'.format(route_distance)

        plan_output += 'Load of the route: {}\n'.format(route_load)

        print(plan_output)

        total_distance += route_distance

        total_load += route_load

    print('Total Distance of all routes: {}m'.format(total_distance))

    print('Total Load of all routes: {}'.format(total_load))



    

# create list of routes for each vehicle

def list_routes(data, manager, routing, assignment):

    # empty dict to hold results

    x = {}

    

    for vehicle_id in range(data['num_vehicles']):

        index = routing.Start(vehicle_id) # set the start node

        x['vehicle_'+str(vehicle_id)+'_route'] = []

        while not routing.IsEnd(index):

            node_index = manager.IndexToNode(index) # get the node index

            x['vehicle_'+str(vehicle_id)+'_route'].append(node_index) # append the node index to the result list 

            index = assignment.Value(routing.NextVar(index)) # update the node index

            

        x['vehicle_'+str(vehicle_id)+'_route'].append(0)

    

    return x

        



def main():

    """Solve the CVRP problem."""

    # Instantiate the data problem.

    data = create_data_model()



    # Create the routing index manager.

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),

                                           data['num_vehicles'], data['depot'])



    # Create Routing Model.

    routing = pywrapcp.RoutingModel(manager)





    # Create and register a transit callback.

    def distance_callback(from_index, to_index):

        """Returns the distance between the two nodes."""

        # Convert from routing variable Index to distance matrix NodeIndex.

        from_node = manager.IndexToNode(from_index)

        to_node = manager.IndexToNode(to_index)

        return data['distance_matrix'][from_node][to_node]



    transit_callback_index = routing.RegisterTransitCallback(distance_callback)



    # Define cost of each arc.

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)





    # Add Capacity constraint.

    def demand_callback(from_index):

        """Returns the demand of the node."""

        # Convert from routing variable Index to demands NodeIndex.

        from_node = manager.IndexToNode(from_index)

        return data['demands'][from_node]



    demand_callback_index = routing.RegisterUnaryTransitCallback(

        demand_callback)

    routing.AddDimensionWithVehicleCapacity(

        demand_callback_index,0,data['vehicle_capacities'],True,'Capacity')

    

    # Allow the vehicle to skip over locations

    for node in range(1, len(data['distance_matrix'])):

        routing.AddDisjunction([manager.NodeToIndex(node)], # indices

                               int(data['demands'][manager.NodeToIndex(node)]), # penalty

                               1) # max cardinality



    # Setting first solution heuristic

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (

        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)



    # Solve the problem

    assignment = routing.SolveWithParameters(search_parameters)



    # Print solution on console

    if assignment:

        print_solution(data, manager, routing, assignment)

        return list_routes(data, manager, routing, assignment)
routes = main()
# validate for vehicle 0



veh_route = pd.DataFrame({'route_node':routes['vehicle_0_route']})



# add on meteorite weights, latitude, and longitude

veh_route = veh_route.merge(right=temp[['mass','latitude','longitude']],left_on='route_node',right_index=True,how='left')



# distances

veh_route['distance'] = 0



for j in range(1,len(veh_route)):

    while j <= (len(veh_route)-1):

        veh_route.loc[j,'distance'] = haversine_distance(veh_route.loc[j,'latitude'],

                                                         veh_route.loc[j,'longitude'],

                                                         veh_route.loc[j-1,'latitude'],

                                                         veh_route.loc[j-1,'longitude'])

        j+=1





# get a cumulative totals

veh_route['mass_total'] = veh_route['mass'].cumsum()

veh_route['dist_total'] = veh_route['distance'].cumsum()

        

veh_route
# function to create route details

def get_routes(node_list):

    

    # DataFrame to hold results

    x = pd.DataFrame({'route_node':node_list})

    

    # fetch latitude and longitude

    x = x.merge(right=temp[['latitude','longitude']],

                left_on='route_node',

                right_index=True,

                how='left')

    

    # create an origin->destination structure

    y = pd.DataFrame()

    y['origin_node'] = x['route_node']

    y['ori_lat'] = x['latitude']

    y['ori_lon'] = x['longitude']

    

    y['dest_node'] = y['origin_node'].shift(-1)

    y['dest_lat'] = y['ori_lat'].shift(-1)

    y['dest_lon'] = y['ori_lon'].shift(-1)

    

    y.dropna(inplace=True)

    

    return y
# put together the data

veh_route_1 = get_routes(routes['vehicle_0_route'])

veh_route_2 = get_routes(routes['vehicle_1_route'])

veh_route_3 = get_routes(routes['vehicle_2_route'])

veh_route_4 = get_routes(routes['vehicle_3_route'])
# plot the routes

m = folium.Map(location=[39.0, -100.0],width = '100%',height='100%',min_zoom = 2.5,zoom_start=4.3)



# candidates not selected

for _, row in df.loc[cand_mask,:].iterrows():

    folium.CircleMarker([row['latitude'],row['longitude']],radius=2.5,color='grey',fill_color = 'grey').add_to(m)



# vehicle 1

for _, row in veh_route_1.iterrows():

    folium.CircleMarker([row['ori_lat'],row['ori_lon']],radius=2.5,color='blue',fill_color = 'blue').add_to(m)

    folium.CircleMarker([row['dest_lat'],row['dest_lon']],radius=2.5,color='blue',fill_color = 'blue').add_to(m)

    folium.PolyLine([[row['ori_lat'],row['ori_lon']],[row['dest_lat'],row['dest_lon']]],color='blue').add_to(m)



# vehicle 2

for _, row in veh_route_2.iterrows():

    folium.CircleMarker([row['ori_lat'],row['ori_lon']],radius=2.5,color='purple',fill_color = 'purple').add_to(m)

    folium.CircleMarker([row['dest_lat'],row['dest_lon']],radius=2.5,color='purple',fill_color = 'purple').add_to(m)

    folium.PolyLine([[row['ori_lat'],row['ori_lon']],[row['dest_lat'],row['dest_lon']]],color='purple').add_to(m)



# vehicle 3

for _, row in veh_route_3.iterrows():

    folium.CircleMarker([row['ori_lat'],row['ori_lon']],radius=2.5,color='red',fill_color = 'red').add_to(m)

    folium.CircleMarker([row['dest_lat'],row['dest_lon']],radius=2.5,color='red',fill_color = 'red').add_to(m)

    folium.PolyLine([[row['ori_lat'],row['ori_lon']],[row['dest_lat'],row['dest_lon']]],color='red').add_to(m)



# vehicle 4

for _, row in veh_route_4.iterrows():

    folium.CircleMarker([row['ori_lat'],row['ori_lon']],radius=2.5,color='orange',fill_color = 'orange').add_to(m)

    folium.CircleMarker([row['dest_lat'],row['dest_lon']],radius=2.5,color='orange',fill_color = 'orange').add_to(m)

    folium.PolyLine([[row['ori_lat'],row['ori_lon']],[row['dest_lat'],row['dest_lon']]],color='orange').add_to(m)



# display the map    

m