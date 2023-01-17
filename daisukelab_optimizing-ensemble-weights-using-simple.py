# This is copy of Simple, just wanted to import..
# Visit https://github.com/chrisstroemel/Simple/blob/master/Simple.py
#  for find the latest code.

from heapq import heappush, heappop, heappushpop
import numpy
import math
import time
import matplotlib.pyplot as plotter

CAPACITY_INCREMENT = 1000


class _Simplex:
	def __init__(self, pointIndices, testCoords, contentFractions, objectiveScore, opportunityCost, contentFraction, difference):
		self.pointIndices = pointIndices
		self.testCoords = testCoords
		self.contentFractions = contentFractions
		self.contentFraction = contentFraction
		self.__objectiveScore = objectiveScore
		self.__opportunityCost = opportunityCost
		self.update(difference)

	def update(self, difference):
		self.acquisitionValue = -(self.__objectiveScore + (self.__opportunityCost * difference))
		self.difference = difference

	def __eq__(self, other):
		return self.acquisitionValue == other.acquisitionValue

	def __lt__(self, other):
		return self.acquisitionValue < other.acquisitionValue

class SimpleTuner:
	def __init__(self, cornerPoints, objectiveFunction, exploration_preference=0.15):
		self.__cornerPoints = cornerPoints
		self.__numberOfVertices = len(cornerPoints)
		self.queue = []
		self.capacity = self.__numberOfVertices + CAPACITY_INCREMENT
		self.testPoints = numpy.empty((self.capacity, self.__numberOfVertices))
		self.objective = objectiveFunction
		self.iterations = 0
		self.maxValue = None
		self.minValue = None
		self.bestCoords = []
		self.opportunityCostFactor = exploration_preference #/ self.__numberOfVertices
			

	def optimize(self, maxSteps=10):
		for step in range(maxSteps):
			#print(self.maxValue, self.iterations, self.bestCoords)
			if len(self.queue) > 0:
				targetSimplex = self.__getNextSimplex()
				newPointIndex = self.__testCoords(targetSimplex.testCoords)
				for i in range(0, self.__numberOfVertices):
					tempIndex = targetSimplex.pointIndices[i]
					targetSimplex.pointIndices[i] = newPointIndex
					newContentFraction = targetSimplex.contentFraction * targetSimplex.contentFractions[i]
					newSimplex = self.__makeSimplex(targetSimplex.pointIndices, newContentFraction)
					heappush(self.queue, newSimplex)
					targetSimplex.pointIndices[i] = tempIndex
			else:
				testPoint = self.__cornerPoints[self.iterations]
				testPoint.append(0)
				testPoint = numpy.array(testPoint, dtype=numpy.float64)
				self.__testCoords(testPoint)
				if self.iterations == (self.__numberOfVertices - 1):
					initialSimplex = self.__makeSimplex(numpy.arange(self.__numberOfVertices, dtype=numpy.intp), 1)
					heappush(self.queue, initialSimplex)
			self.iterations += 1

	def get_best(self):
		return (self.maxValue, self.bestCoords[0:-1])

	def __getNextSimplex(self):
		targetSimplex = heappop(self.queue)
		currentDifference = self.maxValue - self.minValue
		while currentDifference > targetSimplex.difference:
			targetSimplex.update(currentDifference)
			# if greater than because heapq is in ascending order
			if targetSimplex.acquisitionValue > self.queue[0].acquisitionValue:
				targetSimplex = heappushpop(self.queue, targetSimplex)
		return targetSimplex
		
	def __testCoords(self, testCoords):
		objectiveValue = self.objective(testCoords[0:-1])
		if self.maxValue == None or objectiveValue > self.maxValue: 
			self.maxValue = objectiveValue
			self.bestCoords = testCoords
			if self.minValue == None: self.minValue = objectiveValue
		elif objectiveValue < self.minValue:
			self.minValue = objectiveValue
		testCoords[-1] = objectiveValue
		if self.capacity == self.iterations:
			self.capacity += CAPACITY_INCREMENT
			self.testPoints.resize((self.capacity, self.__numberOfVertices))
		newPointIndex = self.iterations
		self.testPoints[newPointIndex] = testCoords
		return newPointIndex


	def __makeSimplex(self, pointIndices, contentFraction):
		vertexMatrix = self.testPoints[pointIndices]
		coordMatrix = vertexMatrix[:, 0:-1]
		barycenterLocation = numpy.sum(vertexMatrix, axis=0) / self.__numberOfVertices

		differences = coordMatrix - barycenterLocation[0:-1]
		distances = numpy.sqrt(numpy.sum(differences * differences, axis=1))
		totalDistance = numpy.sum(distances)
		barycentricTestCoords = distances / totalDistance

		euclideanTestCoords = vertexMatrix.T.dot(barycentricTestCoords)
		
		vertexValues = vertexMatrix[:,-1]

		testpointDifferences = coordMatrix - euclideanTestCoords[0:-1]
		testPointDistances = numpy.sqrt(numpy.sum(testpointDifferences * testpointDifferences, axis=1))



		inverseDistances = 1 / testPointDistances
		inverseSum = numpy.sum(inverseDistances)
		interpolatedValue = inverseDistances.dot(vertexValues) / inverseSum


		currentDifference = self.maxValue - self.minValue
		opportunityCost = self.opportunityCostFactor * math.log(contentFraction, self.__numberOfVertices)

		return _Simplex(pointIndices.copy(), euclideanTestCoords, barycentricTestCoords, interpolatedValue, opportunityCost, contentFraction, currentDifference)

	def plot(self):
		if self.__numberOfVertices != 3: raise RuntimeError('Plotting only supported in 2D')
		matrix = self.testPoints[0:self.iterations, :]

		x = matrix[:,0].flat
		y = matrix[:,1].flat
		z = matrix[:,2].flat

		coords = []
		acquisitions = []

		for triangle in self.queue:
			coords.append(triangle.pointIndices)
			acquisitions.append(-1 * triangle.acquisitionValue)


		plotter.figure()
		plotter.tricontourf(x, y, coords, z)
		plotter.triplot(x, y, coords, color='white', lw=0.5)
		plotter.colorbar()


		plotter.figure()
		plotter.tripcolor(x, y, coords, acquisitions)
		plotter.triplot(x, y, coords, color='white', lw=0.5)
		plotter.colorbar()

		plotter.show()
import pandas as pd
import numpy as np
import keras

# We are ensembling these four models.
model_id_list_to_ensemble = ['D6', 'E', 'E4', 'H2']

# And each results are stored here.
exampledir = "../input/tutasc-ensemble-example/"

# Load the training data, then make list of labels and dictionaries for converting labels and its numbers.
raw_y_train = pd.read_csv('../input/acoustic-scene-2018/y_train.csv', sep=',')['scene_label'].tolist()
labels = sorted(list(set(raw_y_train)))
label2int = {l:i for i, l in enumerate(labels)}
int2label = {i:l for i, l in enumerate(labels)}

# Make reference answers for valid as y_valid_ref, split from the raw training answers.
y_train_int = [label2int[l] for l in raw_y_train]
splitlist = pd.read_csv('../input/acoustic-scene-2018/crossvalidation_train.csv', sep=',')['set'].tolist()
y_valid_ref = keras.utils.to_categorical(np.array([y for i, y in enumerate(y_train_int) if splitlist[i] == 'test']))
# Load all predicted answers, and convert them to each result as class number.
raw_valid_preds = [np.load(exampledir + 'preds%s4valid.npy' % e) for e in model_id_list_to_ensemble]
ref_valid_cls = [np.argmax(y) for y in y_valid_ref]

# f() calculates combined prediction results by weighted ensembling.
def f(weights):
    norm_weights = weights / np.sum(weights)
    valid_preds = np.average(raw_valid_preds, axis=0, weights=norm_weights)
    y_valid_pred_cls = [np.argmax(pred) for pred in valid_preds]
    return y_valid_pred_cls

# Calculates accuracy for particular weights
def acc_function(weights):
    y_valid_pred_cls = f(weights)
    n_eq = [result == ref for result, ref in zip(y_valid_pred_cls, ref_valid_cls)]
    return np.sum(n_eq) / len(y_valid_pred_cls)

# This defines the search area, and other optimization parameters.
optimization_domain_vertices = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
number_of_iterations = 3000
exploration = 0.01 # optional, default 0.15

# Optimize weights
tuner = SimpleTuner(optimization_domain_vertices, acc_function, exploration_preference=exploration)
tuner.optimize(number_of_iterations)
best_objective_value, best_weights = tuner.get_best()

print('Best objective value =', best_objective_value)
print('Optimum weights =', best_weights)
print('Ensembled Accuracy (same as best objective value) =', acc_function(best_weights))
from scipy import optimize

# We need loss function to minimize
def loss_function(weights):
    y_valid_pred_cls = f(weights)
    n_lost = [result != ref for result, ref in zip(y_valid_pred_cls, ref_valid_cls)]
    return np.sum(n_lost) / len(y_valid_pred_cls)

opt_weights = optimize.minimize(loss_function,
                                [1/len(model_id_list_to_ensemble)] * len(model_id_list_to_ensemble),
                                constraints=({'type': 'eq','fun': lambda w: 1-sum(w)}),
                                method= 'Nelder-Mead', #'SLSQP',
                                bounds=[(0.0, 1.0)] * len(model_id_list_to_ensemble),
                                options = {'ftol':1e-10},
                            )['x']

print('Optimum weights =', opt_weights, 'with loss', loss_function(opt_weights))
print('Ensembled Accuracy =', acc_function(opt_weights))
