from tqdm import tqdm

import numpy as np



from collections import defaultdict

import itertools

from functools import lru_cache



import pickle



import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)
def cost(photo1, photo2):

    intersect = len(photo1.intersection(photo2))

    return min(len(photo1) - intersect, len(photo2) - intersect, intersect)



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



# Create some variables to store the solution in

sequence = [-1] * len(photos)

total_cost = 0



# Sample our first slide (must be horizontal)

sequence[0] = np.random.choice([i for i in range(len(photos)) if photos[i][0] == 'H'])

tags = photos[sequence[0]][1]

for tag in photos[sequence[0]][1]:

    photos_per_tag[tag].remove(sequence[0])

    

remaining_pics = list(set(range(len(photos))) - set(sequence))

remaining_horizontal_pics = [p for p in remaining_pics if photos[p][0] == 'H']

remaining_vertical_pics = [p for p in remaining_pics if photos[p][0] == 'V']



# Iteratively add a slide to the sequence

for i in tqdm(range(1, len(sequence))):

    # Fallback: In case we do not find any candidates, we just take 1 random horizontal or 2 random vertical pics

    if len(remaining_horizontal_pics) > 0:

        best_j = np.random.choice(remaining_horizontal_pics)

    elif len(remaining_vertical_pics) > 1:

        best_j = tuple(np.random.choice(remaining_vertical_pics, size=2, replace=False))

    else:

        break

        

    best_cost = total_cost

    

    # Get a list of K possible good candidates

    K = 2500

    k = 0.5

    vertical_candidates = set()

    horizontal_candidates = set()

    in_common_tags = defaultdict(int)

    for tag in tags:

        for p in photos_per_tag[tag]:

            in_common_tags[p] += 1

            

    if len(in_common_tags) > 0:

            

        max_tags = max(in_common_tags.values())

        for p in in_common_tags:

            if in_common_tags[p] == max_tags:

                if photos[p][0] == 'H':

                    horizontal_candidates.add(p)

                else:

                    vertical_candidates.add(p)

                    

        for p in in_common_tags:

            if len(horizontal_candidates) + len(vertical_candidates) > K:

                break



            if in_common_tags[p] >= k * max_tags:

                if photos[p][0] == 'H':

                    horizontal_candidates.add(p)

                else:

                    vertical_candidates.add(p)



        # Candidates consist of all possible horizontal candidates and all combinations of 2 vertical candidates

        candidates = list(horizontal_candidates) + list(itertools.combinations(vertical_candidates, 2))



        # Iterate over candidates and pick the one that increases the score the most.

        curr_best = 0

        old_cost = best_cost

        for j in candidates:

            if isinstance(j, tuple):

                new_tags = photos[j[0]][1].union(photos[j[1]][1])

            else:

                new_tags = photos[j][1]



            if len(new_tags) <= 2*curr_best:

                continue



            new_cost = total_cost + cost(tags, new_tags)



            if new_cost >= best_cost:

                best_cost = new_cost

                curr_best = new_cost - old_cost

                best_j = j



    # Assign a new picture to the next slide

    total_cost = best_cost

    sequence[i] = best_j

    

    if isinstance(best_j, tuple):

        tags = photos[sequence[i][0]][1].union(photos[sequence[i][1]][1])

        remaining_pics.remove(best_j[0])

        remaining_vertical_pics.remove(best_j[0])

        for tag in photos[sequence[i][0]][1]:

            photos_per_tag[tag].remove(sequence[i][0])

        remaining_pics.remove(best_j[1])

        remaining_vertical_pics.remove(best_j[1])

        for tag in photos[sequence[i][1]][1]:

            photos_per_tag[tag].remove(sequence[i][1])

    else:

        remaining_horizontal_pics.remove(best_j)

        remaining_pics.remove(best_j)

        tags = photos[sequence[i]][1]

        for tag in photos[sequence[i]][1]:

            photos_per_tag[tag].remove(sequence[i])
print('Score = {}'.format(sequence_cost(sequence)))
with open('submission.txt', 'w+') as ofp:

    ofp.write('{}\n'.format(sum(np.array(sequence) != -1)))

    for p in sequence:

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

for i, p in enumerate(sequence):

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

print(done - set(range(len(photos))))
!wc -l submission.txt
!tail submission.txt
!head submission.txt