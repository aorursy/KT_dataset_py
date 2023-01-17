import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm
N_team_mate = 4

rank = 100

N_teams = 2000
def points(rank=rank, t=0, N_team_mate=N_team_mate, N_teams=N_teams, verbose=0):

#     assert (N_team_mate>0) & (N_teams>0) & (rank>0)

    coeffs = [100000/np.sqrt(N_team_mate)]

    coeffs.append(rank**(-0.75))

    coeffs.append(np.log10(1+np.log10(N_teams)))

    coeffs.append(np.exp(-t/500))

    res = np.prod(coeffs)

#     res= int(round(res))

    if verbose:

        print(f'In a competition of {N_teams} teams where your team of {N_team_mate} people is ranked {rank}, your points at day {t} are {int(round(res))}')

    return res

points(verbose=1)
R = np.arange(1, N_teams//3)

P = [points(rank=r, verbose=0) for r in R]

plt.plot(figsize=(8,8))

plt.plot(R, P, 'ro', markersize=2)

plt.title('Points function of ranking')

plt.show()
N = np.arange(1,5000)

P = [points(N_teams=n, verbose=0) for n in N]

plt.plot(figsize=(8,8))

plt.plot(N, P, 'ro', markersize=1)

plt.title('Points function of number of competitive teams')

plt.show()
N = np.arange(1,5)

P = [points(N_team_mate=n, verbose=0) for n in N]

plt.plot(figsize=(8,8))

plt.plot(N, P, 'ro', markersize=5)

plt.title('Points function of size of team')

plt.show()
T = np.arange(0,1000)

P = [points(t=t, verbose=0) for t in T]

plt.plot(figsize=(8,8))

plt.plot(T, P, 'ro', markersize=1)

plt.title('Points function of days')

plt.show()
np.exp(-365/500)
N1 = np.arange(1,5)

N=2000

R = np.arange(10,400)

fig = plt.figure(figsize=(10,8))

list_of_P = {}

for n in N1:

    P = points(rank=R, N_team_mate=n, N_teams=N)

    list_of_P[n] = P

    plt.plot(R, P, label=f'team size of {n}')

plt.axhline(y=1000, linewidth=2, c='grey', ls='--')

plt.axvline(x=100, linewidth=2, c='black', ls='--')

plt.legend()

plt.show()
for n, P in list_of_P.items():

    print(f'For team size {n}, you get 1000 points at rank {np.argmin([abs(p-1000) for p in P])}')

    print(f'For team size {n}, at rank 100 you get {int(round(points(rank=100, N_team_mate=n, N_teams=N)))} points')

    print('-'*100)