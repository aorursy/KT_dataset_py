import pandas as pd

import re

import math

import numpy as np

import itertools

import random

import matplotlib.pyplot as plt

import copy
file = open('../input/C108.txt','r')

number_of_strings  = 103

string=""

for i in range(6):

    string += file.readline()

s = re.findall('\d+ *\d*', string)

s2 = str.split(s[1], ' ')

max_vehicle_number = int(s2[0])

max_vehicle_capacity = int(s2[len(s2)-1])

print("MAX VEHICLE NUMBER: {}\nMAX VEHICLE CAPACITY: {}".format(max_vehicle_number, max_vehicle_capacity))

string = file.readline()

string.replace(string, '')

new_file = open('helper.csv', 'w+')

stri=file.readlines()

for i in range(number_of_strings):

    new_file.write(stri[i])

new_file.read()

file.close()

new_file.close()
problem1 = pd.read_csv('helper.csv', sep="\s+ |\n" )

problem1['VISITED'] = 0

problem1[:10]
customers_list_init = []

for i in range (len(problem1)):

    customers_list_init.append([problem1.iloc[i,0], problem1.iloc[i,1], problem1.iloc[i,2], problem1.iloc[i,3],\

                   problem1.iloc[i,4], problem1.iloc[i,5], problem1.iloc[i,6], problem1.iloc[i,7]])



depot = customers_list_init[0]

x = [x for [_,x,_,_,_,_,_,_] in customers_list_init]

y = [y for [_,_,y,_,_,_,_,_] in customers_list_init]
class Route:

    def __init__(self, customers: list, total_length = 0, current_time = 0, remaining_capacity = max_vehicle_capacity):

        self.customers = list(customers)

        self.total_length = 0

        self.current_time = 0

        self.remaining_capacity = int(max_vehicle_capacity)

        

    def __repr__(self):

        return "[" + " ".join(str(Number) for [Number,_,_,_,_,_,_,_] in self.customers) + "]"

    

    def path_between_customers(self, cust1, cust2):

        return math.hypot(cust2[1] - cust1[1], cust2[2] - cust1[2])

    

    @property

    def needed_view(self):

        time = 0

        result = [0, 0.0]

        depot = customers_list_init[0]

        for depot, customer in zip(self.customers, self.customers[1:]):

            start_time = max([customer[4], time + self.path_between_customers(depot, customer)])

            time = start_time + customer[6]

            result.append(customer[0])

            result.append(start_time)

        return " ".join(str(res) for res in result)
def may_be_a_route(route):

        route.remaining_capacity = max_vehicle_capacity

        route.total_length = 0

        time = 0

        for cust1, cust2 in zip(route.customers, route.customers[1:]):

            accumulated_time = max([cust2[4], time + route.path_between_customers(cust1, cust2)])

            if accumulated_time > cust2[5]:

                return False

            time = accumulated_time + cust2[6]

            route.remaining_capacity -= cust2[3]

            route.total_length += route.path_between_customers(cust1, cust2)

            if route.remaining_capacity < 0 or time > depot[5]:

                return False

        return True

    

def recount_obj_func(route):

        for cust1, cust2 in zip(route.customers, route.customers[1:]):

            route.total_length += route.path_between_customers(cust1, cust2)
def dummy_solution(customers, problem):

    routes = [Route([]) for i in range(120)]

    depot = customers_list_init[0]

    customers.sort(key = lambda x: x[4])

    for i in range(1,len(customers)):

        cust = customers[i]

        for route in routes:

            new_customers = route.customers + [cust]

            if may_be_a_route(Route([depot, *new_customers, depot])) and cust[7] == 0:

                recount_obj_func(Route([depot, *new_customers, depot]))

                cust[7] = 1

                problem.at[cust[0],'VISITED'] += 1

                route.customers.append(cust)

    return routes
def plot_solution(routes):

    plt.figure(figsize=(10,8))

    plt.plot(x,y, marker = 'o', color = 'black', ls = '')

    for route in routes:

        route = Route([depot, *route.customers, depot])

        print("Route:{}\n".format(route))

        for i in range(len(route.customers)):

            x1 = [X for [_,X,_,_,_,_,_,_] in route.customers]

            y1 = [Y for [_,_,Y,_,_,_,_,_] in route.customers]

            plt.plot(x1, y1)

    

    plt.show()
##Dummy colution computation



# customers_dummy = list(customers_list_init)

# routes_dummy = dummy_solution(customers_dummy, problem1)

# plot_solution(routes_dummy)

# check2 = []

# for route in routes_dummy:

#     check2.append(may_be_a_route(route))

# print(check2)
def object_function(routes):

    total_path = 0

    for route in routes:

        route = Route([depot, *route.customers, depot])

        recount_obj_func(route)

        total_path += route.total_length

    return total_path
def make_check(problem):

    check = []

    for i in range(len(problem)):

        check.append(problem.iloc[i,7])

    print(check)

    assert len(np.unique(check[1:])) == 1

    problem['VISITED'] = 0

    assert len(np.unique(check[1:])) == 1
##Deprecated

def two_opt(cust_l, c1,c2):

    if c1 == 0:

        return cust_l[c2:c1:-1] + [cust_l[c1]] + cust_l[c2+1:]

    return cust_l[:c1] + cust_l[c2:c1-1:-1] + cust_l[c2+1:]



##In-use

def one_move(route1, route2, customer_index, position):

    if len(route1.customers) == 0:

        return [], route2.customers

    if customer_index > len(route1.customers) or position  > len(route2.customers):

        return route1.customers, route2.customers

    if customer_index != len(route1.customers):

        return route1.customers[:customer_index] + route1.customers[customer_index + 1:], route2.customers[:position] + [route1.customers[customer_index]] + route2.customers[position:]

    elif customer_index == len(route1.customers) and position == len(route2.customers):

        return route1.customers, route2.customers

    elif customer_index == len(route1.customers):

        return route1.customers[:customer_index], route2.customers[:position] + [route1.customers[customer_index-1]] + route2.customers[position:]

    elif position == len(route2.customers):

        return route1.customers[:customer_index] + route1.customers[customer_index + 1:], route2.customers[:position] + [route1.customers[customer_index]]



def swap(route1, route2, cust1, cust2):

    if cust1 >= len(route1.customers) or cust2 >= len(route2.customers):

        return route1.customers, route2.customers

    return route1.customers[:cust1] + [route2.customers[cust2]] + route1.customers[cust1 + 1:], route2.customers[:cust2] + [route1.customers[cust1]] + route2.customers[cust2 + 1:]



def cross(route1, route2, i, j):

    return route1.customers[:i] + route2.customers[j:], route2.customers[:j] + route1.customers[i:]
def local_search(routes):

    best_result = [Route(route.customers) for route in routes]

    cant_be_optimized = False

    oper = [one_move,swap, cross]

    while not cant_be_optimized:

        for r in best_result:

            recount_obj_func(r)

        cant_be_optimized = True

        for r1, r2 in itertools.combinations(range(len(best_result)),2):

            #all routes

            for i in range(len(best_result[r1].customers)):

                for j in range(len(best_result[r2].customers)):

                    #all pairs ofcustomers indices in routes

                    random.shuffle(oper)

                    for pert in oper:

                        cust1, cust2 = pert(best_result[r1],best_result[r2],i,j)

                        new_r1 = Route(cust1)

                        new_r2 = Route(cust2)

                        new_r1_w_depot = Route([depot, *cust1, depot])

                        new_r2_w_depot = Route([depot, *cust2, depot])

                        if may_be_a_route(new_r1_w_depot) and may_be_a_route(new_r2_w_depot):

                            best_result_r1_w_depot = Route([depot, *best_result[r1].customers, depot])

                            best_result_r2_w_depot = Route([depot, *best_result[r2].customers, depot])

                            recount_obj_func(best_result_r1_w_depot)

                            recount_obj_func(best_result_r2_w_depot)

                            if new_r1_w_depot.total_length + new_r2_w_depot.total_length < best_result_r1_w_depot.total_length + best_result_r2_w_depot.total_length:

                                        best_result[r1] = new_r1

                                        best_result[r2] = new_r2

                                        cant_be_optimized = False

    return best_result



##LS result computation



# routes_optimized1 = local_search(routes_dummy)

# for route in routes_optimized1:

#     for cust in route.customers:

#         cust[7] += 1

#         problem1.at[cust[0],'VISITED'] += 1

# make_check(problem1)

# object_function(routes_optimized1)
def perturbation(routes):

    depot = customers_list_init[0]

    best_result1 = routes.copy()

    best_result1.append(Route([]))

    normal_route = Route([])

    number = -1

    for route in routes:

        number += 1

        if len(route.customers) != 0:

            normal_route = route

            break

            

    is_input = False

    for i in range(len(normal_route.customers)):

        is_input = False

        for route in best_result1[number+1:]:

            if not is_input:

                if len(route.customers) != 0:

                    for j in range(len(route.customers)):

                        cust1, cust2 = one_move(normal_route, route, 0, j)

                        if may_be_a_route(Route([depot, *cust2, depot])):

                            route.customers = cust2

                            normal_route.customers = cust1

                            is_input = True

                            break

                else:

                    cust1, cust2 = one_move(normal_route, route, 0, 0)

                    if may_be_a_route(Route([depot, *cust2, depot])):

                        route.customers = cust2

                        normal_route.customers = cust1

                        is_input = True

                        break

    best_result1[number] = normal_route

    return best_result1





def iterated_local_search(customers, problem):

    initial_solution = dummy_solution(customers, problem)



    best_result = list(local_search(initial_solution)) 

    best_result.append(Route([]))

    

    for r in best_result:

        recount_obj_func(Route([depot, *r.customers, depot]))

    

    for i in range(50):

        print("Step {}".format(i))



        br = copy.deepcopy(best_result)

        solution = perturbation(br)

        

        solution = local_search(solution)



        if object_function(solution) < object_function(best_result):

            best_result = solution

            continue

    return best_result
##ILS computation



# answer = iterated_local_search(customers_list_init, problem1)

# plot_solution(answer)

# for r in answer:

#     r = Route([depot, *r.customers, depot])

#     print(may_be_a_route(r))

#     print(r.total_length)

# object_function(answer)

# for route in answer:

#     for cust in route.customers:

#         cust[7] += 1

#         problem1.at[cust[0],'VISITED'] += 1

# make_check(problem1)

# object_function(answer)

# for route in answer:

#     print(Route([depot, *route.customers, depot]).needed_view)
distances = [[math.hypot(c2[1] - c1[1], c2[2] - c1[2]) for c1 in customers_list_init] for c2 \

             in customers_list_init]
def matrix_sum(matrix):

    mat_sum = 0

    for row in matrix:

        mat_sum += sum(row)

    return mat_sum/2 # graph is not oriented!



def solution_to_matrix(solution):

    matrix = [[0 for x in range(len(customers_list_init))] for y in range(len(customers_list_init))]

    for route in solution:

        route = Route([depot, *route.customers, depot])

        for cust1, cust2 in zip(route.customers, route.customers[1:]):

            matrix[cust1[0]][cust2[0]] = 1

            matrix[cust2[0]][cust1[0]] = 1 

    return matrix



def route_to_penalties_sum(route, penalty_list):

    route = Route([depot, *route.customers, depot])

    penalty_sum = 0

    for cust1, cust2 in zip(route.customers, route.customers[1:]):

        penalty_sum += penalty_list[cust1[0]][cust2[0]]

    return penalty_sum



def recount_obj_func_with_penalties(route, penalty_list):

    for cust1, cust2 in zip(route.customers, route.customers[1:]):

        route.total_length += route.path_between_customers(cust1, cust2)

    route.total_length += route_to_penalties_sum(route, penalty_list)
def local_search_for_penalty(routes, penalty):

    best_result = [Route(route.customers) for route in routes]

    cant_be_optimized = False

    oper = [one_move,swap, cross]

    step = 0

    old_best_obj = 999999

    while old_best_obj - object_function(best_result) > 0.00001:

        old_best_obj = object_function(best_result)

        for r in best_result:

            recount_obj_func_with_penalties(r, penalty)

        cant_be_optimized = True

        for r1, r2 in itertools.combinations(range(len(best_result)),2):

            #all routes

            for i in range(len(best_result[r1].customers)):

                for j in range(len(best_result[r2].customers)):

                    #all pairs of customers indices in routes

                    random.shuffle(oper)

                    for pert in oper:

                        cust1, cust2 = pert(best_result[r1],best_result[r2],i,j)

                        new_r1 = Route(cust1)

                        new_r2 = Route(cust2)

                        new_r1_w_depot = Route([depot, *cust1, depot])

                        new_r2_w_depot = Route([depot, *cust2, depot])

                        if may_be_a_route(new_r1_w_depot) and may_be_a_route(new_r2_w_depot):

                            best_result_r1_w_depot = Route([depot, *best_result[r1].customers, depot])

                            best_result_r2_w_depot = Route([depot, *best_result[r2].customers, depot])

                            recount_obj_func_with_penalties(best_result_r1_w_depot, penalty)

                            recount_obj_func_with_penalties(best_result_r2_w_depot, penalty)

                            if new_r1_w_depot.total_length + new_r2_w_depot.total_length < best_result_r1_w_depot.total_length + best_result_r2_w_depot.total_length:

                                        best_result[r1] = new_r1

                                        best_result[r2] = new_r2

                                        

                                        cant_be_optimized = False

        step += 1 

    return best_result
def guided_local_search(customers, problem):

    k = 0

    solution = dummy_solution(customers, problem1)

    lambd = 0.25



    # n x n matrix - list of lists    

    penalties = [[0 for x in range(len(customers_list_init))] for y in range(len(customers_list_init))]

    while k < 20:

        print("Step {}".format(k))

        

        #new augmented obj func. 

        #h = object_function(solution) + lambd * matrix_sum(solution_to_matrix(solution))

        

        solution = local_search_for_penalty(solution, penalties)

        mat = solution_to_matrix(solution)

        utils = [[mat[x][y] * distances[x][y]/(1 + penalties[x][y]) for x in range(len(customers_list_init))] \

                    for y in range(len(customers_list_init))]

        max_util = 0

        for row in utils:

            if max(row) > max_util:

                max_util = max(row)

        

        for i in range(len(utils)):

            for j in range(len(utils[i])):

                if utils[i][j] == max_util:

                    penalties[i][j] += 1

        k += 1

    return solution

    

    
##GLS computation



# answer = guided_local_search(customers_list_init, problem1)

# plot_solution(answer)

# print(object_function(answer))

# object_function(answer)

# for route in answer:

#     for cust in route.customers:

#         cust[7] += 1

#         problem1.at[cust[0],'VISITED'] += 1

# make_check(problem1)

# object_function(answer)

# for route in answer:

#     print(Route([depot, *route.customers, depot]).needed_view)