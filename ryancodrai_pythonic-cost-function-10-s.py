import pandas as pd

import numpy as np

import numba as nu

from numba import jit
fpath = '/kaggle/input/santa-workshop-tour-2019/family_data.csv'

data = pd.read_csv(fpath, index_col='family_id')
# For i = 0, 1, ..., 4998, 4999 let family_members(i) denote the number of family members for the i'th family

family_members = nu.int16(data['n_people'].values)

print('family_members: %s' %  str(family_members.shape))



# For i = 0, 1, ..., 4998, 4999 and k = 0, 1, ..., 8, 9 let prefered(i, k) denote the k'th preference of day for the i'th family

prefered = nu.int16(data[[*[f'choice_{i}' for i in range(10)]]].values) - 1

print('prefered:       %s' % str(prefered.shape))
# For k = 0, 1, ..., 8, 9 let coefficient(k) denote the discount of Santa's Buffet & Helicopter Ride

# for one family member whose family has been allocated their k'th prefered day

coefficient = nu.int16([0, 0, 9, 9, 9, 18, 18, 36, 36, 235])

print('coefficient: %s' % str(coefficient.shape))



# For k = 0, 1, ..., 8, 9 let constant(k) denote the gift card value for one family being allocated their k'th prefered day

constant = nu.int16([0, 50, 50, 100, 200, 200, 300, 300, 400, 500])

print('constant:    %s' % str(constant.shape))



# For i = 0, 1, ..., 4998, 4999 and j = 0, 1, ..., 98, 99 let c1(i, j) denote the cost associated with the i'th family being allocated the j'th day.

# Initially assign c1(i, j) the cost associated with the i'th family not having been allocated any of their prefered days

c1 = np.tile(family_members * 434 + 500, (100, 1)).T



# For i = 0, 1, ..., 4998, 4999 and k = 0, 1, ..., 8, 9 let c1(i, p(k)) equal the cost of the i'th family being alllocated day p(k)

c1[np.repeat(np.arange(5000), 10), prefered.flatten()] = (np.tile(family_members, (10, 1)).T * coefficient + constant).flatten()

print('c1:          %s' % str(c1.shape))
# For a = 0, 1, ..., 498, 499 and b = 0, 1, ..., 498, 499 let c2(a, b) denote the accountancy term

# where a is the occupancy of the current day, and b is the occupancy of the previous day

c2 = np.zeros((500, 500))

for i in range(c2.shape[0]):

    for j in range(c2.shape[1]):

        if i < 125 or j < 125 or i > 300 or j > 300:

            c2[i][j] = float("inf")

        else:

            c2[i][j] = max(0, (i - 125) / 400 * i ** (0.5 + abs(i - j) / 50))

            

print('c2: %s' % str(c2.shape))
@jit(nopython=True, fastmath=True)

def score(preferences, occupancy):

    # Accountancy cost     

    score = c2[occupancy[-1], occupancy[-1]]

    for i in range(occupancy.size - 2, -1, -1):

        score += c2[occupancy[i], occupancy[i + 1]]

    # Preference cost    

    for i in range(preferences.size):

        score += c1[i, preferences[i]]

    return score
fpath = '/kaggle/input/santa-workshop-tour-2019/sample_submission.csv'

submission = pd.read_csv(fpath, index_col='family_id')



# For i = 0, 1, ..., 4998, 4999 let assigned(i) equal the day assigned to the i'th family

assigned = nu.int16(submission['assigned_day'].values) - 1

print('assigned:   %s' % str(assigned.shape))



# For j = 0, 1, ..., 98, 99 let attendance(j) equal the number of people attending on the j'th day

attendance = nu.int16(np.bincount(assigned,  family_members))

print('attendance: %s' % str(attendance.shape))



# Evaluate the solution

print('score:      %s' % score(assigned, attendance))
%timeit score(assigned, attendance)