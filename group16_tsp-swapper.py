from tqdm import tqdm

import numpy as np

import itertools

from collections import defaultdict

import math

from ortools.constraint_solver import routing_enums_pb2

from ortools.constraint_solver import pywrapcp

import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)
# Read our input

with open('../input/hashcode-photo-slideshow/d_pet_pictures.txt', 'r') as ifp:

    lines = ifp.readlines()



photos = []

all_tags = list()

photos_per_tag = defaultdict(list)

for i, line in enumerate(lines[1:]):

    orient, _, *tags = line.strip().split()

    photos.append((orient, set(tags)))

    for tag in tags:

        photos_per_tag[tag].append(i)
def cost(tags1, tags2):

    intersect = len(tags1.intersection(tags2))

    return min(len(tags1) - intersect, len(tags2) - intersect, intersect)



def cost2(photo1, photo2):

    if isinstance(photo1, tuple):

        tags1 = photos[photo1[0]][1].union(photos[photo1[1]][1])

    else:

        tags1 = photos[photo1][1]

        

    if isinstance(photo2, tuple):

        tags2 = photos[photo2[0]][1].union(photos[photo2[1]][1])

    else:

        tags2 = photos[photo2][1]

        

    return cost(tags1, tags2)



def sequence_cost(sequence):

    total_cost = 0

    for i in range(len(sequence) - 1):

        if sequence[i + 1] == -1:

            break

            

        if isinstance(sequence[i], tuple):

            old_tags = photos[sequence[i][0]][1].union(photos[sequence[i][1]][1])

        else:

            old_tags = photos[sequence[i]][1]

            

        if isinstance(sequence[i + 1], tuple):

            new_tags = photos[sequence[i + 1][0]][1].union(photos[sequence[i + 1][1]][1])

        else:

            new_tags = photos[sequence[i + 1]][1]

            

        total_cost += cost(old_tags, new_tags)

    return total_cost
# Read our submission

with open('../input/hashcode-greedy-solution-415k/submission.txt', 'r') as ifp:

    lines = ifp.readlines()



sequence = []

for i, line in enumerate(lines[1:]):

    if ' ' in line:

        sequence.append(tuple(map(int, line.strip().split())))

    else:

        sequence.append(int(line.strip()))

        

print(sequence_cost(sequence))
SUBSEQ_LEN = 5000

TIME_LIMIT = 300



new_sequence = sequence[:]

for i in range(len(sequence) // SUBSEQ_LEN):

    print('Swapping in block {}/{}'.format(i + 1, len(sequence) // SUBSEQ_LEN))

    subsequence = sequence[i*SUBSEQ_LEN:(i + 1)*SUBSEQ_LEN]

    COSTS = {}

    for r in range(len(subsequence)):

        COSTS[r + 1] = {}

        for c in range(len(subsequence)):

            # TSP will minimize costs, but we want to maximize. This is a simple hack

            COSTS[r + 1][c + 1] = 1000-cost2(subsequence[r], subsequence[c])

            

    # 0 will be a dummy starting location. We use the last element of the previous

    # subsequence as starting point to determine the costs and the first element

    # of the next subsequence as ending point.

    COSTS[0] = {}

    COSTS[0][0] = 0

    for c in range(len(subsequence)):

        COSTS[0][c + 1] = 0

        COSTS[c + 1][0] = 0

    if i == 0:

        for c in range(len(subsequence)):

            COSTS[c + 1][0] = cost2(subsequence[c], new_sequence[(i + 1)*SUBSEQ_LEN])

    elif i == len(sequence) // SUBSEQ_LEN - 1:

        for c in range(len(subsequence)):

            COSTS[0][c + 1] = cost2(new_sequence[i*SUBSEQ_LEN - 1], subsequence[c])

    else:

        for c in range(len(subsequence)):

            COSTS[0][c + 1] = cost2(new_sequence[i*SUBSEQ_LEN - 1], subsequence[c])

            COSTS[c + 1][0] = cost2(subsequence[c], new_sequence[(i + 1)*SUBSEQ_LEN])

        

    old_score = sequence_cost(subsequence)

    print('Old score = {}'.format(old_score))

            

    """Entry point of the program."""

    # Create the routing index manager.

    manager = pywrapcp.RoutingIndexManager(len(subsequence), 1, [0], [0])



    # Create Routing Model.

    routing = pywrapcp.RoutingModel(manager)



    distance_matrix = COSTS



    def distance_callback(from_index, to_index):

        """Returns the distance between the two nodes."""

        # Convert from routing variable Index to distance matrix NodeIndex.

        from_node = manager.IndexToNode(from_index)

        to_node = manager.IndexToNode(to_index)

        return distance_matrix[from_node][to_node]



    transit_callback_index = routing.RegisterTransitCallback(distance_callback)



    # Define cost of each arc.

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)



    # Setting initial solution (the order is just 1 until SUBSEQ_LEN (included))

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    initial_solution = routing.ReadAssignmentFromRoutes([list(range(1, SUBSEQ_LEN + 1))], True)

    search_parameters.time_limit.seconds = TIME_LIMIT



    # Solve the problem.

    solution = routing.SolveFromAssignmentWithParameters(initial_solution, search_parameters)

    

    indices = [routing.Start(0)]

    while not routing.IsEnd(indices[-1]):

        indices.append(solution.Value(routing.NextVar(indices[-1])))



    # i - 1 because our 0 corresponds to the dummy and 1 corresponds to sequence[0]

    new_sub_sequence = [subsequence[i - 1] for i in indices[1:]]   

    new_score = sequence_cost(new_sub_sequence)

    print('New score = {}'.format(new_score))

    

    # The swapping could reduce the score due to the transitions of the

    # beginning and ending of the subsequence in the total sequence

    if new_score > old_score:

        new_sequence = new_sequence[:i*SUBSEQ_LEN] + new_sub_sequence + new_sequence[(i + 1)*SUBSEQ_LEN:]

    

    total_score = sequence_cost(new_sequence)

    print('Current score = {}'.format(total_score))
print(sequence_cost(new_sequence))
with open('submission.txt', 'w+') as ofp:

    ofp.write('{}\n'.format(sum(np.array(new_sequence) != -1)))

    for p in new_sequence:

        if p == -1:

            break

            

        if isinstance(p, tuple):

            ofp.write('{} {}\n'.format(p[0], p[1]))

        else:

            ofp.write('{}\n'.format(p))
# CHECKS:

# 1) We dont want duplicates

# 2) We want vertical pictures to always be paired with another vertical picture

# 3) We don't want horizontal pictures to be paired

# 4) Preferably, we assign all of the pictures to slides

# 5) We cannot assign a picture to two different slides

done = set()

for i, p in enumerate(new_sequence):

    if p == -1:

        break

    if isinstance(p, tuple):

        assert p[0] != p[1]

        assert p[0] not in done

        assert photos[p[0]][0] == 'V'

        done.add(p[0])

        

        assert p[1] not in done

        assert photos[p[1]][0] == 'V'

        done.add(p[1])

    else:

        assert p not in done

        assert photos[p][0] == 'H'

        done.add(p)

print(i, len(done))

print(set(range(len(photos))) - done)