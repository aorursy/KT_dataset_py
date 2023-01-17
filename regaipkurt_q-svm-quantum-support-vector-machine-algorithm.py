!pip install qiskit
import numpy as np

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

import qiskit

from qiskit import BasicAer, Aer

from qiskit.aqua import QuantumInstance

from qiskit.aqua.algorithms import QSVM

from qiskit.aqua.components.multiclass_extensions import one_against_rest, all_pairs

from qiskit.aqua.components.feature_maps import SecondOrderExpansion

from qiskit.aqua.input import ClassificationInput

from qiskit.aqua import run_algorithm



backend = Aer.get_backend('qasm_simulator')

iris = load_iris()
X, Y = iris.data, iris.target

print(X.shape)

print(len(set(Y)))

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

num_features = 4

training_size = 120

test_size = 30

feature_map = SecondOrderExpansion(feature_dimension=num_features, depth=3)

# With feature_map, we have defined the function we will use the quantum property space we mentioned above.
params = {

            'problem': {'name': 'classification', 'random_seed': 794},

            'algorithm': {

                'name': 'QSVM',

            },

            'backend': {'shots': 1},

            'multiclass_extension': {'name': 'OneAgainstRest'},

            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 3, 'entangler_map': [[1, 0]]}

            #'feature_map': {'name': 'SecondOrderExpansion', 'depth': 3}

        }

training_dataset={'A':train_x[train_y==0],

                'B':train_x[train_y==1],

                'C':train_x[train_y==2]}



test_dataset={'A':test_x[test_y==0],

                        'B':test_x[test_y==1],

                        'C':test_x[test_y==2]}



total_arr = np.concatenate((test_dataset['A'],test_dataset['B'],test_dataset['C']))

alg_input = ClassificationInput(training_dataset, test_dataset, total_arr)
result = run_algorithm(params, algo_input=alg_input, backend=backend)
result