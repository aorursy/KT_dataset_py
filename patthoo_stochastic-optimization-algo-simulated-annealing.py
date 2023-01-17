# Author: Patrick_Ho
# Stochastic_optimization_algo
# Last modified: 20/8/2018

import random
import math

random.seed(100000)
# function to update new temperature
def update_T(t):
    t = t * 0.99
    return round(t, 7)

# generate a neighbor within a distance l from value u
def get_neighbor(u, d):
    distance = d
    lowerbound = u - distance
    upperbound = u + distance
    n = round(random.uniform(max(-1, lowerbound), min(upperbound, 1)), 6)

    return n

# calculate acceptance probability
def acceptance_p(old_sol, new_sol, t):
    p = math.exp((new_sol - old_sol) / t)
    return p
# *** PART A ***
# --------------
# define function z
def function_z1(u):
    outcome = u * math.sin(1 / (0.01 + u ** 2)) + (u ** 3) * math.sin(1 / (0.001 + u ** 4))
    return round(outcome,6)

def simulated_annealing_a(u = round(random.uniform(-1, 1), 6)):
    results = []

    # ***Simulated annealing***
    # u = random.uniform(-1, 1)  # generate random u in the range from -1 to 1
    z = function_z1(u)  # calculate the result of function z for u
    t = 1  # initial temperature
    print("u: " + str(u) + "\t\tz: " + str(z) + "\t\ttemperature: " + str(t))

    # the temperature cools down until 0
    while t > 0.0001:
        i = 0

        # create a cycle of 100 times
        while i <= 100:
            u_neighbor = get_neighbor(u, 0.05)  # get a neighbor of u
            new_z = function_z1(u_neighbor)  # new solution of function z

            ap = acceptance_p(z, new_z, t)  # calculate acceptance probability

            # if the new solution is better, so move to that solution
            if new_z > z:
                u = u_neighbor
                z = new_z

            # if acceptance probability is high, then the search is more likely to move to a worse solution
            if ap > random.random():
                u = u_neighbor
                z = new_z

            i += 1

        print("u: " + str(u) + "\t\tz: " + str(z) + "\t\ttemperature: " + str(t))
        results.append({"u": u, "z": z, "temperature": t})
        t = update_T(t)  # update temperature

    # finding maximum z with corresponding u and temperature
    best_z = max([x["z"] for x in results])
    for item in results:
        if item["z"] == best_z:
            best_u = item["u"]
            temp = item["temperature"]

    print("Maximum value of z: " + str(best_z))
    print("Correspoding value of u is: " + str(best_u))
    print("At temperature: " + str(temp))
    
# *** PART B ***
# --------------

# define function z = f(u,v)
def function_z2(u,v):
    outcome = u*(v**2)*math.sin(v/(0.01 + u**2)) + (u**3)*(v**2)*math.sin(v**3/(0.001 + u**4))
    return round(outcome,6)

def simulated_annealing_b(u = round(random.uniform(-1, 1),6), v = round(random.uniform(-1, 1),6)):
    results_b = []

    # ***Simulated annealing***
    z = function_z2(u, v)  # calculate the result of function z for u
    t = 1  # initial temperature
    print("u: " + str(u) + "\t\tv: " + str(v) + "\t\tz: " + str(z) + "\t\ttemperature: " + str(t))

    # the temperature cools down until 0
    while t > 0.0001:
        t = update_T(t)  # update temperature
        i = 0

        # create a cycle of 100 times
        while i <= 1000:
            u_neighbor = get_neighbor(u, 0.5)  # get a neighbor of u within distance of 0.05
            v_neighbor = get_neighbor(v, 0.5)  # get a neighbor of v within distance of 0.05
            new_z = function_z2(u_neighbor, v_neighbor)  # new solution of function z

            ap = acceptance_p(z, new_z, t)  # calculate acceptance probability

            # if the new solution is better, so move to that solution
            if new_z > z:
                u = u_neighbor
                v = v_neighbor
                z = new_z

            # if acceptance probability is high, then the search is more likely to move to a worse solution
            if ap > random.random():
                u = u_neighbor
                v = v_neighbor
                z = new_z

            i += 1

        print("u: " + str(u) + "\t\tv: " + str(v) + "\t\tz: " + str(z) + "\t\ttemperature: " + str(t))
        results_b.append({"u, v": [u, v], "z": z, "temperature": t})

    # finding maximum z with corresponding u, v and temperature
    best_z = max([x["z"] for x in results_b])
    for item in results_b:
        if item["z"] == best_z:
            best_u_v = item["u, v"]
            temp_b = item["temperature"]

    print("Maximum value of z: " + str(best_z))
    print("Correspoding value of (u, v) is: " + str(best_u_v))
    print("At temperature: " + str(temp_b))

# *** PART C ***
# --------------
# define function z = f(u,v,w)
def function_z3(u,v,w):
    outcome = (u*(v**2) + math.sin(math.pi*w))*math.sin(v/(0.01 + u**2))*math.sin((math.pi*w)/2) + \
              (u**3)*(v**2) * w * math.sin(v**3/(0.001*(math.sin((math.pi*w)/2)**2) + u**4 + (w-1)**2))
    return round(outcome,6)

def simulated_annealing_c(u = round(random.uniform(-1,1),6), v = round(random.uniform(-1,1),6), w = round(random.uniform(-1,1),6)):
    results_c = []

    # simulated annealing
    z = function_z3(u=u, v=v, w=w)  # calculate the result of function z for u, v and w
    t = 1  # initial temperature

    print("u: " + str(u) + "\t\tv: " + str(v) + "\t\tw: " + str(w) + "\t\tz: " + str(z) + "\t\ttemperature: " + str(t))

    # the temperature cools down until 0
    while t > 0.0001:
        t = update_T(t)  # update temperature
        i = 0
        # create a cycle of 1000 times
        while i <= 1000:
            u_neighbor = get_neighbor(u, 0.5)  # get a neighbor of u
            v_neighbor = get_neighbor(v, 0.5)  # get a neighbor of v
            w_neighbor = get_neighbor(w, 0.5)  # get a neighbor of w

            new_z = function_z3(u_neighbor, v_neighbor, w_neighbor)  # new solution of function z

            ap = acceptance_p(z, new_z, t)  # calculate acceptance probability

            # if the new solution is better, so move to that solution
            if new_z > z:
                u = u_neighbor
                v = v_neighbor
                w = w_neighbor
                z = new_z

            # if acceptance probability is high, then the search is more likely to move to a worse solution
            if ap > random.random():
                u = u_neighbor
                v = v_neighbor
                w = w_neighbor
                z = new_z

            i += 1

        print("u: " + str(u) + "\t\tv: " + str(v) + "\t\tw: " + str(w) + "\t\tz: " + str(z) + "\t\ttemperature: " + str(t))
        results_c.append({"u, v, w": [u, v, w], "z": z, "temperature": t})

    # finding maximum z with corresponding u, v, w and temperature
    best_z = max([x["z"] for x in results_c])
    for item in results_c:
        if item["z"] == best_z:
            best_u_v_w = item["u, v, w"]
            temp_c = item["temperature"]

    print("Maximum value of z: " + str(best_z))
    print("Correspoding value of (u, v, w) is: " + str(best_u_v_w))
    print("At temperature: " + str(temp_c))

# *** PART D ***
# --------------
# define function z = f(u,v,w,y)
def function_z4(u, v, w, y):
    outcome = function_z3(u,v,w)*y
    return round(outcome,6)

def simulated_annealing_d(u = round(random.uniform(-1,1),6), v = round(random.uniform(-1,1),6), w = round(random.uniform(-1,1),6), y = -1):
    results_d = []
    z = function_z4(u, v, w, y)  # calculate the result of function z for u, v, w and y
    t = 1  # initial temperature

    # for each value of y, find the best value of [u,v,w] which generates the largest value of z
    # then select the largest values of z from these regions -1, 0 and 1
    for y in [-1,0,1]:
        # the temperature cools down until 0
        best_z = z
        best_u = u
        best_v = v
        best_w = w

        new_t = t

        print("u: " + str(u) + "\t\tv: " + str(v) + "\t\tw: " + str(w) + "\t\ty: " + str(y) +
              "\t\tz: " + str(z) + "\t\ttemperature: " + str(t))

        while new_t > 0.001:
            new_t = update_T(new_t)  # update temperature
            i = 0

            while i <= 1000:
                u_neighbor = get_neighbor(best_u, 0.25)  # get a neighbor of u
                v_neighbor = get_neighbor(best_v, 0.25)  # get a neighbor of v
                w_neighbor = get_neighbor(best_w, 0.25)  # get a neighbor of w

                new_z = function_z4(u_neighbor, v_neighbor, w_neighbor, y)  # new solution of function z

                ap = acceptance_p(best_z, new_z, new_t)  # calculate acceptance probability

                if new_z > best_z:
                    best_u = u_neighbor
                    best_v = v_neighbor
                    best_w = w_neighbor
                    best_z = new_z

                if ap > random.random():
                    best_u = u_neighbor
                    best_v = v_neighbor
                    best_w = w_neighbor
                    best_z = new_z

                i += 1

            print("u: " + str(best_u) + "\t\tv: " + str(best_v) + "\t\tw: " + str(best_w) + "\t\ty: " + str(y) +
                  "\t\tz: " + str(best_z) + "\t\ttemperature: " + str(new_t))
            results_d.append({"u, v, w, y": [best_u, best_v, best_w, y], "z": best_z, "temperature": new_t})

    # finding maximum z with corresponding u, v, w and temperature
    max_z = max([x["z"] for x in results_d])
    for item in results_d:
        if item["z"] == max_z:
            best_value = item["u, v, w, y"]
            temp = item["temperature"]

    print("Maximum value of z: " + str(max_z))
    print("Correspoding value of (u, v, w, y) is: " + str(best_value))
    print("At temperature: " + str(temp))
u = input("Please give an input for u (default is a random value): ")
if u == "":
        u = round(random.uniform(-1, 1),6)
simulated_annealing_a(float(u))
u = input("Please give an input for u (default is a random value): ")
if u == "":
    u = round(random.uniform(-1, 1),6)

v = input("Please give an input for v (default is a random value): ")
if v == "":
    v = round(random.uniform(-1, 1),6)
        
simulated_annealing_b(float(u), float(v))
u = input("Please give an input for u (default is a random value): ")
if u == "":
    u = round(random.uniform(-1, 1),6)

v = input("Please give an input for v (default is a random value): ")
if v == "":
    v = round(random.uniform(-1, 1),6)

w = input("Please give an input for w (default is a random value): ")
if w == "":
    w = round(random.uniform(-1, 1),6)
        
simulated_annealing_c(float(u), float(v), float(w))
u = input("Please give an input for u (default is a random value): ")
if u == "":
    u = round(random.uniform(-1, 1),6)

v = input("Please give an input for v (default is a random value): ")
if v == "":
    v = round(random.uniform(-1, 1),6)

w = input("Please give an input for w (default is a random value): ")
if w == "":
    w = round(random.uniform(-1, 1),6)

y = input("Please give an input for y (default is a random value): ")
if y == "":
    y = random.randint(-1, 1)

simulated_annealing_d(float(u), float(v), float(w), float(y))