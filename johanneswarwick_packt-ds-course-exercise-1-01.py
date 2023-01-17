algorithm = ['Linear Regression', 'Logistic Regression', 'RandomForest', 'a3c']

learning = ['Supervised', 'Supervised', 'Supervised', 'Reinforcement']

algorithm_type = ['Regression', 'Classification', 'Regression or Classification', 'Game AI']
algorithm.append('k-means')
print(algorithm)
learning.append('Unsupervised')
print(learning)
algorithm_type.append('Clustering')
print(algorithm_type)
machine_learning = {}
machine_learning['algorithm'] = algorithm
print(machine_learning)
machine_learning['learning'] = learning
machine_learning['algorithm_type'] = algorithm_type
print(machine_learning)
#machine_learning['algorithm'].remove('a3c')
print(machine_learning['algorithm'])
machine_learning['learning'].remove('Reinforcement')
machine_learning['algorithm_type'].remove('Game AI')
print(machine_learning)