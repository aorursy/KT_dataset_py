import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm

lambda_rentals = [3,4]

lambda_returns = [3,2]

discount = .9

reward_rental = 10

reward_move = -2

max_cars = 20

actions = np.arange(-5,6,1)

v = np.zeros([max_cars+1,max_cars+1])

p = np.zeros([max_cars+1,max_cars+1],dtype=int)

theta = 1e-4



keep_going = True

timestep = 0

while keep_going:

    # Policy Evaluation

    old_v = np.array(v,copy=True)

    for i in range(max_cars+1):

        for j in range(max_cars+1):

#             prob_rental = scipy.stats.poison.pmf(,lamda_rentals[0]) *\ scipy.stats.poisson.pmf(,lamda_rentals[1])

#             prob_return = scipy.stats.poison.pmf(,lamda_return[0]) *\ scipy.stats.poisson.pmf(,lamda_return[1])

            rented_cars_1 = min(i,lambda_rentals[0])

            rented_cars_2 = min(j,lambda_rentals[1])

            returned_cars_1 = min(lambda_returns[0],max_cars-i) # car can't return more than max_car - current cars

            returned_cars_2 = min(lambda_rentals[1],max_cars-j)

            if p[i,j] > 0 and(1-rented_cars_1+returned_cars_1) >= p[i,j]:

                moved_cars_1 = min(p[i,j],i-rented_cars_1+returned_cars_1)

                moved_car_2 = - moved_cars_1

                reward = moved_cars_1*reward_move

            else:

                moved_cars_2 = min(-p[i,j],j-rented_cars_2+returned_cars_2)

                moved_cars_1 = - moved_cars_2

                reward = moved_cars_2*reward_move

            reward += reward_rental*(rented_cars_1+rented_cars_2)

            no_cars_1 = i - rented_cars_1+ returned_cars_1 - moved_cars_1

            no_cars_2 = j - rented_cars_2+ returned_cars_2 - moved_cars_2

            v[i,j] = reward + discount*v[no_cars_1,no_cars_2]

    delta = np.abs(v-old_v).max()

    timestep += 1

    print(timestep)

    if delta<theta:

        break
delta = 1

x = np.arange(0, max_cars+1, delta)

y = np.arange(0, max_cars+1, delta)

X, Y = np.meshgrid(x, y)

Z = v

fig, ax = plt.subplots()

CS = ax.contour(X, Y, Z, cmap="RdBu_r")

ax.clabel(CS, inline=1, fontsize=10)