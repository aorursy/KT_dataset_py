

import csv

import math

import sys

coordinates_list = []

shortest_distance = float('inf')

best_path = []

best_distance = float('inf')

input_data = "../input/cp2410-a1-cities/cities_first5.csv"

output_data = "output.csv"        

start_coordinate = [0.0, 0.0, 0.0]     # Start point. Format: [0.0, x-coord, y-coord]

end_coordinate = [0.0, 0.0, 0.0]       # End point. Format:[0.0, x-coord, y-coord]
with open(input_data, 'r') as csvfile:  # Read data from input file

        csvreader = csv.reader(csvfile)  

        col_names = next(csvreader)        

        for coordinate_STR in csvreader:        

            coordinate_FLT = []              # Converting data points from String to Float

            for i in coordinate_STR:

                coordinate_FLT.append(float(i))

            coordinates_list.append(coordinate_FLT)

        

def save_best_path():

    with open(output_data, 'w', newline='') as file:

        writer = csv.writer(file)

        writer.writerows(best_path)
if sys.getrecursionlimit() < len(coordinates_list): # Checks recursion limit

    sys.setrecursionlimit(2*len(coordinates_list))  # Extend recursion limit if hit
# Calculations below assume all coordinates given on a flat surface

def find_dist(pt1, pt2):                # Calculate distance between 2 points

    pt1 = int(pt1)                      

    pt2 = int(pt2)

    distance = math.sqrt(((coordinates_list[pt1][1] - coordinates_list[pt2][1])**2) + ((coordinates_list[pt1][2] - coordinates_list[pt2][2])**2))

    return distance



def find_start_dist(pt):                # Function to account for start point if given

    pt = int(pt)                        

    distance = math.sqrt(((start_coordinate[1] - coordinates_list[pt][1])**2) + ((start_coordinate[2] - coordinates_list[pt][2])**2))

    return distance



def find_end_dist(pt):                  # Function to account for end point if given

    pt = int(pt)                        

    distance = math.sqrt(((end_coordinate[1] - coordinates_list[pt][1])**2) + ((end_coordinate[2] - coordinates_list[pt][2])**2))

    return distance
# Recursive function to find shortest point-to-point distance

def min_coordinate_distance():

    global shortest_distance

    num_pts_left = len(coordinates_list)

    coordinates_left = coordinates_list.copy()    # Creates a separate list to be sorted by coordinates

    best_pt1 = 0.0

    best_pt2 = 0.0

    for i in range(num_pts_left - 1):             # Begins a loop to check every point

        for j in range(i+1, num_pts_left):        # Begins second loop to check against other points

            if find_dist(coordinates_left[i][0], coordinates_left[j][0]) < shortest_distance:    # Checks if better

                shortest_distance = find_dist(coordinates_left[i][0], coordinates_left[j][0])    # Saves better difference

                best_pt1 = coordinates_left[i][0]        # Saves point indexes

                best_pt2 = coordinates_left[j][0]

    print("Shortest distance is between coordinates: ", best_pt1, " and ", best_pt2)

    print("Shortest distance is: ", shortest_distance)

    return shortest_distance



def possible_path(num_pts_left, curr_distance, curr_pt):

    if (((num_pts_left) * shortest_distance) + curr_distance + find_end_dist(curr_pt)) < best_distance:

        return True

    else:

        return False

    

def better_path(curr_path, curr_dist):

    global best_path

    global best_distance

    if curr_dist < best_distance:

        best_path = curr_path.copy()

        best_distance = curr_dist
# Recursive function to plot every possible path

def plot_path(num_pts_left, curr_path, coordinates_left, coordinates_list, curr_dist):

    if len(curr_path) == 0 and len(coordinates_left) == 0 and coordinates_list != 0:    # Makes a separate list to track unused coordinates

        coordinates_left = coordinates_list.copy()

    if len(curr_path) == 0:

        curr_path.append(start_coordinate)               # Add start point

    for coordinate in coordinates_left:                  # Begins loop for each data point to plot possible paths

        curr_path.append(coordinate)                     # Add point in path to keep track of points used

        coordinates_left.remove(coordinate)              # Remove used points from list of unused points



        if len(start_coordinate) == 3 and len(curr_path) == 2:       # Check for start point

            curr_dist = find_start_dist(coordinate[0])               # Add distance from start to first point

        else:

            curr_dist += find_dist(curr_path[-2][0], coordinate[0])  # Update current path distance



        if num_pts_left == 1:                                   # Checking for last point

            curr_dist += find_end_dist(coordinate[0])                # Add distance if last point located

            better_path(curr_path, curr_dist)



        if possible_path(num_pts_left, curr_dist, coordinate[0]):

            plot_path(num_pts_left - 1, curr_path, coordinates_left, coordinates_list, curr_dist)



        curr_dist -= find_dist(curr_path[-2][0], coordinate[0])

        coordinates_left.insert(int(coordinate[0]), coordinate)

        coordinates_left.sort()

        curr_path.remove(coordinate)







min_coordinate_distance()

plot_path(len(coordinates_list), [], [], coordinates_list, 0)

best_path.remove(start_coordinate)

save_best_path()

print("Best possible distance is: ", best_distance)

print("Best path found: ", best_path)
import csv

import math

import sys

coordinates_list = []

shortest_distance = float('inf')

best_path = []

best_distance = float('inf')

input_data = "../input/cp2410-a1-cities/cities_10percent.csv"  

output_data = "output2.csv"        

start_coordinate = [0.0, 0.0, 0.0]     # Start point. Format: [0.0, x-coord, y-coord]

end_coordinate = [0.0, 0.0, 0.0]       # End point. Format:[0.0, x-coord, y-coord]



with open(input_data, 'r') as csvfile:  # Read data from input file

        csvreader = csv.reader(csvfile)  

        col_names = next(csvreader)        

        for coordinate_STR in csvreader:        

            coordinate_FLT = []              # Converting data points from String to Float

            for i in coordinate_STR:

                coordinate_FLT.append(float(i))

            coordinates_list.append(coordinate_FLT)

        

def save_best_path():

    with open(output_data, 'w', newline='') as file:

        writer = csv.writer(file)

        writer.writerows(best_path)

        

# Calculations below assume all coordinates given on a flat surface

def find_dist(pt1, pt2):                # Calculate distance between 2 points

    pt1 = int(pt1)                      

    pt2 = int(pt2)

    distance = math.sqrt(((coordinates_list[pt1][1] - coordinates_list[pt2][1])**2) + ((coordinates_list[pt1][2] - coordinates_list[pt2][2])**2))

    return distance



def find_start_dist(pt):                # Function to account for start point if given

    pt = int(pt)                        

    distance = math.sqrt(((start_coordinate[1] - coordinates_list[pt][1])**2) + ((start_coordinate[2] - coordinates_list[pt][2])**2))

    return distance



def find_end_dist(pt):                  # Function to account for end point if given

    pt = int(pt)                        

    distance = math.sqrt(((end_coordinate[1] - coordinates_list[pt][1])**2) + ((end_coordinate[2] - coordinates_list[pt][2])**2))

    return distance
coordinates_list_sortedX = []



coordinates_list_sortedX = coordinates_list.copy()

coordinates_list_sortedX.sort(key=lambda x: x[1])
def find_closest_pt(curr_coordinate, search_range):                   # Recursive function to find closest coordinate values

    curr_x_index = coordinates_list_sortedX.index(curr_coordinate)    # Defines the index of current point on sorted list

    possible_coordinates_1 = []

    possible_coordinates_2 = []

    dist_holder = float('inf')

    index_holder = 0.0

    if len(coordinates_list_sortedX) == 1:                              # Checks for last coordinate

        return False                                                    # Returns False if yes

    if curr_x_index > 0:                                                # Check for coordinates before current

        for element in coordinates_list_sortedX[curr_x_index-1::-1]:    # If yes, iterate backwards to locate said coordinates

            if abs(curr_coordinate[1] - element[1]) < search_range:

                possible_coordinates_1.append(element)                  # Saves points if x coords in range

            else:

                break                                                   # Ends loop if no more coordinates in range

    for element in coordinates_list_sortedX[curr_x_index + 1:]:         # Find coordinate by iterating forwards

        if abs(curr_coordinate[1] - element[1]) < search_range:

            possible_coordinates_1.append(element)                      # Saves points to list1 if x coords in range

        else:

            break                                                       # Ends loop if no more in range

    if curr_coordinate in possible_coordinates_1:

        possible_coordinates_1.remove(curr_coordinate)

    if len(possible_coordinates_1) > 0:                                      # While there are any possible points

        possible_coordinates_1.sort(key=lambda x: x[2])                      # Sort points by Y coords

        for element2 in possible_coordinates_1:

            if abs(curr_coordinate[2] - element2[2]) < search_range:           # Check if points Y coords are also in range

                possible_coordinates_2.append(element2)                        # Saves points to list2 if yes



        if len(possible_coordinates_2) > 0:                                  # If there are still possible points

            for point3 in possible_coordinates_2:                            # Check all point's distance for closest

                if find_dist(point3[0], curr_coordinate[0]) < dist_holder:

                    dist_holder = find_dist(point3[0], curr_coordinate[0])

                    index_holder = point3[0]

            return index_holder                                              # Returns index of closest point

    return find_closest_pt(curr_coordinate, (search_range + 100)*2)          # Re-curves with bigger range if no points
def plot_path():                                                        # Function that plots the best path

    global best_path                                                    # Initiates the global variables to record

    global best_distance                                                # Best path and distance

    coordinates_list_sortedX.insert(0, start_coordinate)                # Insert start point to the list

    best_path.append(coordinates_list[int(find_closest_pt(start_coordinate, 100))])    # Tracks first point

    best_distance = find_start_dist(find_closest_pt(start_coordinate, 100))            # Measures starting distance

    coordinates_list_sortedX.remove(start_coordinate)                                  # Removes starting point from points list



    while len(coordinates_list_sortedX) > 1:                                 # While there are unused coordinate remaining

        next_pt_index = find_closest_pt(best_path[-1], 100)                  # Find the closest coordinate to the last

        coordinates_list_sortedX.remove(best_path[-1])                       # Remove from list of unused

        best_path.append(coordinates_list[int(next_pt_index)])               # Add to path

        best_distance += find_dist(best_path[-2][0], best_path[-1][0])





    best_path.append(coordinates_list_sortedX[0])                            # Adds last coordinate to end of the path

    best_distance += find_dist(best_path[-2][0], best_path[-1][0])           # Adds distance to last coordinate

    best_distance += find_end_dist(best_path[-1][0])
plot_path()

save_best_path()

global best_path                # Initiates the global variables to print progress into console

global best_distance

print("Best distance is: ", best_distance)

print("Best path found: ", best_path) 