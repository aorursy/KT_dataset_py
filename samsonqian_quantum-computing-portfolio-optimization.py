%pip install qiskit
%pip show qiskit
from qiskit import Aer

from qiskit.circuit.library import TwoLocal

from qiskit.aqua import QuantumInstance

from qiskit.finance.applications.ising import portfolio

from qiskit.optimization.applications.ising.common import sample_most_likely

from qiskit.finance.data_providers import RandomDataProvider

from qiskit.aqua.algorithms import VQE, QAOA, NumPyMinimumEigensolver

from qiskit.aqua.components.optimizers import COBYLA

import numpy as np

import matplotlib.pyplot as plt

import datetime

import os
# set number of assets (= number of qubits)

num_assets = 4



# Generate expected return and covariance matrix from (random) time-series

stocks = [("TICKER%s" % i) for i in range(num_assets)]

data = RandomDataProvider(tickers=stocks,

                 start=datetime.datetime(2016,1,1),

                 end=datetime.datetime(2016,1,30))

data.run()

mu = data.get_period_return_mean_vector()

sigma = data.get_period_return_covariance_matrix()
# plot sigma

plt.imshow(sigma, interpolation='nearest')

plt.show()
q = 0.5                   # set risk factor

budget = num_assets // 2  # set budget

penalty = num_assets      # set parameter to scale the budget penalty term



qubitOp, offset = portfolio.get_operator(mu, sigma, q, budget, penalty)
def index_to_selection(i, num_assets):

    s = "{0:b}".format(i).rjust(num_assets)

    x = np.array([1 if s[i]=='1' else 0 for i in reversed(range(num_assets))])

    return x



def print_result(result):

    selection = sample_most_likely(result.eigenstate)

    value = portfolio.portfolio_value(selection, mu, sigma, q, budget, penalty)

    print('Optimal: selection {}, value {:.4f}'.format(selection, value))

        

    eigenvector = result.eigenstate if isinstance(result.eigenstate, np.ndarray) else result.eigenstate.to_matrix()

    probabilities = np.abs(eigenvector)**2

    i_sorted = reversed(np.argsort(probabilities))

    print('\n----------------- Full result ---------------------')

    print('selection\tvalue\t\tprobability')

    print('---------------------------------------------------')

    for i in i_sorted:

        x = index_to_selection(i, num_assets)

        value = portfolio.portfolio_value(x, mu, sigma, q, budget, penalty)    

        probability = probabilities[i]

        print('%10s\t%.4f\t\t%.4f' %(x, value, probability))
exact_eigensolver = NumPyMinimumEigensolver(qubitOp)

result = exact_eigensolver.run()



print_result(result)
backend = Aer.get_backend('statevector_simulator')

seed = 50



cobyla = COBYLA()

cobyla.set_options(maxiter=500)

ry = TwoLocal(qubitOp.num_qubits, 'ry', 'cz', reps=3, entanglement='full')

vqe = VQE(qubitOp, ry, cobyla)

vqe.random_seed = seed



quantum_instance = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)



result = vqe.run(quantum_instance)



print_result(result)
backend = Aer.get_backend('statevector_simulator')

seed = 50



cobyla = COBYLA()

cobyla.set_options(maxiter=250)

qaoa = QAOA(qubitOp, cobyla, 3)



qaoa.random_seed = seed



quantum_instance = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)



result = qaoa.run(quantum_instance)



print_result(result)