import random

import math    # cos() for Rastrigin

import copy    # array-copying convenience

import sys     # max float

# ------------------------------------

def show_vector(vector):

  for i in range(len(vector)):

    if i % 8 == 0: # 8 columns

      print("\n", end="")

    if vector[i] >= 0.0:

      print(' ', end="")

    print("%.4f" % vector[i], end="") # 4 decimals

    print(" ", end="")

  print("\n")

def error(position):

  err = 0.0

  for i in range(len(position)):

    xi = position[i]

    err += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10

  return err

# ------------------------------------

class Particle:

  def __init__(self, dim, minx, maxx, seed):

    self.rnd = random.Random(seed)

    self.position = [0.0 for i in range(dim)]

    self.velocity = [0.0 for i in range(dim)]

    self.best_part_pos = [0.0 for i in range(dim)]

    for i in range(dim):

      self.position[i] = ((maxx - minx) *

        self.rnd.random() + minx)

      self.velocity[i] = ((maxx - minx) *

        self.rnd.random() + minx)

    self.error = error(self.position) # curr error

    self.best_part_pos = copy.copy(self.position) 

    self.best_part_err = self.error # best error

def Solve(max_epochs, n, dim, minx, maxx):

  rnd = random.Random(0)

  # create n random particles

  swarm = [Particle(dim, minx, maxx, i) for i in range(n)] 

  best_swarm_pos = [0.0 for i in range(dim)] # not necess.

  best_swarm_err = sys.float_info.max # swarm best

  for i in range(n): # check each particle

    if swarm[i].error < best_swarm_err:

      best_swarm_err = swarm[i].error

      best_swarm_pos = copy.copy(swarm[i].position) 

  epoch = 0

  w = 0.729    # inertia

  c1 = 1.49445 # cognitive (particle)

  c2 = 1.49445 # social (swarm)

  while epoch < max_epochs:

    

    if epoch % 2 == 0 and epoch > 1:

      print("Epoch = " + str(epoch) +

        " best error = %.30f" % best_swarm_err)

    for i in range(n): # process each particle

      

      # compute new velocity of curr particle

      for k in range(dim): 

        r1 = rnd.random()    # randomizations

        r2 = rnd.random()

    

        swarm[i].velocity[k] = ( (w * swarm[i].velocity[k]) +

          (c1 * r1 * (swarm[i].best_part_pos[k] -

          swarm[i].position[k])) +  

          (c2 * r2 * (best_swarm_pos[k] -

          swarm[i].position[k])) )  

        if swarm[i].velocity[k] < minx:

          swarm[i].velocity[k] = minx

        elif swarm[i].velocity[k] > maxx:

          swarm[i].velocity[k] = maxx

      # compute new position using new velocity

      for k in range(dim): 

        swarm[i].position[k] += swarm[i].velocity[k]

  

      # compute error of new position

      swarm[i].error = error(swarm[i].position)

      # is new position a new best for the particle?

      if swarm[i].error < swarm[i].best_part_err:

        swarm[i].best_part_err = swarm[i].error

        swarm[i].best_part_pos = copy.copy(swarm[i].position)

      # is new position a new best overall?

      if swarm[i].error < best_swarm_err:

        best_swarm_err = swarm[i].error

        best_swarm_pos = copy.copy(swarm[i].position)

    

    # for-each particle

    epoch += 1

  # while

  return best_swarm_pos

# end Solve

print("\nBegin particle swarm optimization using Python demo\n")

dim = 3

print("Goal is to solve Rastrigin's function in " +

 str(dim) + " variables")

print("Function has known min = 0.0 at (", end="")

for i in range(dim-1):

  print("0, ", end="")

print("0)")

num_particles = 50

max_epochs = 100

print("Setting num_particles = " + str(num_particles))

print("Setting max_epochs    = " + str(max_epochs))

print("\nStarting PSO algorithm\n")

best_position = Solve(max_epochs, num_particles,

 dim, -10.0, 10.0)

print("\nPSO completed\n")

print("\nBest solution found:")

show_vector(best_position)

err = error(best_position)

print("Error of best solution = %.6f" % err)

print("\nEnd particle swarm demo\n")

import random

import math    # sqrt

# ------------------------------------

def show_vector(vector):

  for i in range(len(vector)):

    if i % 8 == 0: # 8 columns

      print("\n", end="")

    if vector[i] >= 0.0:

      print(' ', end="")

    print("%.4f" % vector[i], end="") # 4 decimals

    print(" ", end="")

  print("\n")

# ------------------------------------

def error(position):

  # Euclidean distance to (0, 0, .. 0)

  dim = len(position)

  target = [0.0 for i in range(dim)]

  dist = 0.0

  for i in range(dim):

    dist += (position[i] - target[i])**2

  return math.sqrt(dist)

# ------------------------------------

class Point:

  def __init__(self, dim, minx, maxx):

    self.position = [0.0 for i in range(dim)]

    for i in range(dim):

      self.position[i] = ((maxx - minx) *

        random.random() + minx)

    self.error = error(self.position) # curr error

# ------------------------------------

def Solve(dim, max_epochs, minx, maxx):

  points = [Point(dim, minx, maxx) for i in range(3)] # 3 points

  for i in range(dim): points[0].position[i] = minx

  for i in range(dim): points[2].position[i] = maxx

  best_idx = -1

  other_idx = -1

  worst_idx = -1

  centroid = [0.0 for i in range(dim)]

  expanded = [0.0 for i in range(dim)]

  reflected = [0.0 for i in range(dim)]

  contracted = [0.0 for i in range(dim)]

  arbitrary = [0.0 for i in range(dim)]

  epoch = 0

  while epoch < max_epochs:

    epoch += 1

        

    # identify best, other, worst

    if (points[0].error < points[1].error and

    points[0].error < points[2].error):

      if points[1].error < points[2].error:

        best_idx = 0; other_idx = 1; worst_idx = 2

      else:

        best_idx = 0; other_idx = 2; worst_idx = 1

    elif (points[1].error < points[0].error and

    points[1].error < points[2].error):

      if points[0].error < points[2].error:

        best_idx = 1; other_idx = 0; worst_idx = 2

      else:

        best_idx = 1; other_idx = 2; worst_idx = 0

    else:

      if points[0].error < points[1].error:

        best_idx = 2; other_idx = 0; worst_idx = 1

      else:

        best_idx = 2; other_idx = 1; worst_idx = 0

    if epoch <= 9 or epoch >= 30:

      print("--------------------")

      print("epoch = " + str(epoch) + " ", end="")

      print("best error = ", end="")

      print("%.6f" % points[best_idx].error, end="")

    if epoch == 10:

      print("--------------------")

      print(" . . . ")

    if points[best_idx].error < 1.0e-4:

      if epoch <= 9 or epoch >= 30:

        print(" reached small error. halting")

      break;

    # make the centroid

    for i in range(dim):

      centroid[i] = (points[other_idx].position[i] +

      points[best_idx].position[i]) / 2.0

    # try the expanded point

    for i in range(dim):

      expanded[i] = centroid[i] + (2.0 * (centroid[i] -

      points[worst_idx].position[i]))

    expanded_err = error(expanded)

    if expanded_err < points[worst_idx].error:

      if epoch <= 9 or epoch >= 30:

        print(" expanded found better error than worst error")

      for i in range(dim): 

        points[worst_idx].position[i] = expanded[i]

      points[worst_idx].error = expanded_err

      continue

    # try the reflected point

    for i in range(dim):

      reflected[i] = centroid[i] + (1.0 * (centroid[i] -

      points[worst_idx].position[i]))

    reflected_err = error(reflected)

    if reflected_err < points[worst_idx].error:

      if epoch <= 9 or epoch >= 30:

        print(" reflected found better error than worst error")

      for i in range(dim):

        points[worst_idx].position[i] = reflected[i]

      points[worst_idx].error = reflected_err

      continue

    # try the contracted point

    for i in range(dim):

      contracted[i] = centroid[i] + (-0.5 * (centroid[i] -

      points[worst_idx].position[i]))

    contracted_err = error(contracted)

    if contracted_err < points[worst_idx].error:

      if epoch <= 9 or epoch >= 30:

        print(" contracted found better error than worst error")

      for i in range(dim):

        points[worst_idx].position[i] = contracted[i]

      points[worst_idx].error = contracted_err

      continue

    # try a random point

    for i in range(dim):

      arbitrary[i] = ((maxx - minx) * random.random() + minx)

    arbitrary_err = error(arbitrary)

    if arbitrary_err < points[worst_idx].error:

      if epoch <= 9 or epoch >= 30:

        print(" arbitrary found better error than worst error")

      for i in range(dim):

        points[worst_idx].position[i] = arbitrary[i]

      points[worst_idx].error = arbitrary_err

      continue

    # could not find better point so shrink worst and other

    if epoch <= 9 or epoch >= 30:

      print(" shrinking")

    # 1. worst -> best

    for i in range(dim):

      points[worst_idx].position[i] = (points[worst_idx].position[i]

      + points[best_idx].position[i]) / 2.0

    points[worst_idx].error = error(points[worst_idx].position)

    # 2. other -> best

    for i in range(dim):

      points[other_idx].position[i] = (points[other_idx].position[i]

      + points[best_idx].position[i]) / 2.0

    points[other_idx].error = error(points[other_idx].position)

  # end-while

  print("--------------------")

  print("\nBest position found=")

  show_vector(points[best_idx].position)

# ------------------------------------

print("\nBegin simplex optimization using Python demo\n")

dim = 5

random.seed(0)

print("Goal is to solve the Sphere function in " +

 str(dim) + " variables")

print("Function has known min = 0.0 at (", end="")

for i in range(dim-1):

  print("0, ", end="")

print("0)")

max_epochs = 1000

print("Setting max_epochs    = " + str(max_epochs))

print("\nStarting simplex algorithm\n")

Solve(dim, max_epochs, -10.0, 10.0)

print("\nSimplex algorithm complete")

print("\nEnd simplex optimization demo\n")
import random

import copy    # array-copying convenience

import sys     # max float

# ------------------------------------

def show_path(path):

  for i in range(len(path)-1):

    print(str(path[i]) + " -> ", end="")

  print(path[len(path)-1])

# ------------------------------------

def error(path):

  d = 0.0  # total distance between cities

  for i in range(len(path)-1):

    if path[i] < path[i+1]:

      d += (path[i+1] - path[i]) * 1.0

    else:

      d += (path[i] - path[i+1]) * 1.5

  minDist = len(path)-1

  return d - minDist

# ------------------------------------

class Bee:

  def __init__(self, nc, seed):

    #self.rnd = random.Random(seed)

    self.status = 0  # 0 = inactive, 1 = active, 2 = scout

    self.path = [0 for i in range(nc)]  # potential solution

    

    for i in range(nc):

      self.path[i] = i  # [0,1,2, ...]

    

    random.shuffle(self.path)

    self.error = error(self.path) # bee's current error

# ------------------------------------

def solve(nc, nb, max_epochs):

  # solve TSP for nc cities using nb bees

  # optimal soln is [0,1,2, . . n-1]

  # assumes dist between adj cities is 1.0 or 1.5 

  # create nb random bees

  hive = [Bee(nc, i) for i in range(nb)] 

  best_err = sys.float_info.max  # dummy init value

  for i in range(nb):  # check each random bee

    if hive[i].error < best_err:

      best_err = hive[i].error

      best_path = copy.copy(hive[i].path)

  

  # assign initial statuses

  numActive = int(nb * 0.50)

  numScout = int(nb * 0.25)

  numInactive = nb - (numActive + numScout)

  for i in range(nb):

    if i < numInactive:

      hive[i].status = 0

    elif i < numInactive + numScout:

      hive[i].status = 2

    else:

      hive[i].status = 1

  

  epoch = 0

  while epoch < max_epochs:

    if best_err == 0.0: break

    for i in range(nb):  # process each bee

      if hive[i].status == 1:    # active bee

        # find a neighbor path and associated error

        neighbor_path = copy.copy(hive[i].path)

        ri = random.randint(0, nc-1)  # random index

        ai = 0  # adjacent index. assume last->first

        if ri < nc-1: ai = ri + 1

        tmp = neighbor_path[ri]

        neighbor_path[ri] = neighbor_path[ai]

        neighbor_path[ai] = tmp

        neighbor_err = error(neighbor_path)

        # check if neighbor path is better

        p = random.random()  # [0.0 to 1.0)

        if (neighbor_err < hive[i].error or

         (neighbor_err >= hive[i].error and p < 0.05)):

          hive[i].path = neighbor_path

          hive[i].error = neighbor_err

          # new best?

          if hive[i].error < best_err:

            best_err = hive[i].error

            best_path = hive[i].path

            print("epoch = " + str(epoch) +

              " new best path found ", end="")

            print("with error = " + str(best_err))

        # active bee code

      elif hive[i].status == 2:  # scout bee

        # make random path and error

        random_path = [0 for j in range(nc)]

        for j in range(nc):

          random_path[j] = j

        random.shuffle(random_path)

        random_err = error(random_path)

        # is it better?

        if random_err < hive[i].error:

          hive[i].path = random_path  # ref assignmnt

          hive[i].error = random_err

          # new best?

          if hive[i].error < best_err:

            best_err = hive[i].error

            best_path = hive[i].path

            print("epoch = " + str(epoch) +

              " new best path found ", end="")

            print("with error = " + str(best_err))

      elif hive[i].status == 0:  # inactive

        pass  # null statement

    # for-each bee

    

    epoch += 1

  # while

  

  print("\nBest path found:")

  show_path(best_path)

  print("\nError of best path = " + str(best_err))

# ------------------------------------

print("\nBegin simulated bee colony optimization using Python demo\n")

print("Goal is to solve a dummy Traveling Salesman Problem")

print("\nDistance between cities A and B is (B-A) * 1.0 if B > A")

print(" or (A-B) * 1.5 if A > B. For example, d(3,5) = 2.0")

print(" and d(8,3) = 7.5. In a real scenario you'd have a lookup table")

print("\nFor n cities, the optimal path is 0 -> 1 -> . . -> (n-1)")

print(" with a total path distance of n-1.\n") 

num_cities = 20

num_bees = 50

max_epochs =10000

print("Setting num_cities = " + str(num_cities))

print("Setting num_bees   = " + str(num_bees))

print("Setting max_epochs = " + str(max_epochs) + "\n")

random.seed(1)

solve(num_cities, num_bees, max_epochs)

print("\nEnd simulated bee colony demo\n")