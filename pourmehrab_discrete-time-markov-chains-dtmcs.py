import numpy as np
# !pip install pulp
import pulp as pl # to compute steady states
from scipy.stats import poisson
from itertools import product
# we've got four states /0,1,2,3/
P = np.zeros((4,4))

# inventory: 0,      demand: k>=3
P[0, 0] = 1 - poisson.cdf(2, 1)
# inventory: 0,      demand: 2
P[0, 1] = poisson.pmf(2, 1)
# inventory: 0,      demand: 1
P[0, 2] = poisson.pmf(1, 1)
# inventory: 0,      demand: 0
P[0, 3] = poisson.pmf(0, 1)

# inventory: 1,      demand: k>=1
P[1, 0] = 1 - poisson.cdf(0, 1)
# inventory: 1,      demand: 0
P[1, 1] = poisson.pmf(0, 1)

# inventory: 2,      demand: k>=2
P[2, 0] = 1 - poisson.cdf(1, 1)
# inventory: 2,      demand: 1
P[2, 1] = poisson.pmf(1, 1)
# inventory: 2,      demand: 0
P[2, 2] = poisson.pmf(0, 1)

# inventory: 3,      demand: k>=3
P[3, 0] = 1 - poisson.cdf(2, 1)
# inventory: 3,      demand: 2
P[3, 1] = poisson.pmf(2, 1)
# inventory: 3,      demand: 1
P[3, 2] = poisson.pmf(1, 1)
# inventory: 3,      demand: 0
P[3, 3] = poisson.pmf(0, 1)

print('Transition Matrix\n\n',P)
def get_steady_state(P):
    _model = pl.LpProblem(f"Steady State Probabilities", pl.LpMaximize)
    J = P.shape[0]
    pi = [pl.LpVariable(f'pi_{j}', cat=pl.LpContinuous, lowBound=0) for j in range(J)]
    _model += pl.lpSum([pi[j] for j in range(J)]) == 1, 'sums2one'

    for j in range(J):
        _model += pl.lpSum([P[i, j] * pi[i] for i in range(J)] + [-pi[j]]) == 0, f'Constraint_{j}'

    _model.solve()
    sol = np.array([pi[j].varValue for j in range(J)], dtype=np.float)
    np.testing.assert_array_almost_equal(np.dot(np.transpose(P), sol), sol, err_msg="valid solution", verbose=True)

    return sol
    
    
pi = get_steady_state(P)
print('Steady-State Probabilities\n\n',pi)