import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

%matplotlib inline
def euclid_distance(train_point, given_point):
  distance = np.sum((train_point - given_point)**2 )
  return np.sqrt(distance)
def calc_distance_from_all(all_points, given_point, predictions):
  all_distances = []
  for i,each in enumerate(all_points):
    distance = euclid_distance(each, given_point)
    all_distances.append((distance, int(predictions[i])))
  all_distances.sort(key=lambda tup:[0])
  return all_distances
def get_neighbours(distance, count):
  return distances[:count]
def predict(all_points, given_point, predictions):
  distances = calc_distance_from_all(all_points, given_point, predictions)
  neighbours = get_neighbours(distances, 4)
  op = [row[-1] for row in neighbours]
  prediction = max(set(op), key = op.count)
  return prediction
def accuracy(basex, basey, testx, testy):
  correct = 0
  for i in range(len(testx)):
    p = predict(basex, testx[i], basey)
    if p ==testy[i]:
      correct +=1
  return f"Accuracy: {correct*100/ len(testy)}"
wine = load_wine()
print(wine.DESCR)
x = pd.DataFrame(wine.data, columns = wine.feature_names)
y = pd.DataFrame(wine.target, columns = ['Target'])
x = (x-x.min())/(x.max()-x.min())
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3)
wine.feature_names
f1 = 'hue'
f2 = 'proline'
basex = np.array(xtrain[[f1,f2]])
basey = np.array(ytrain)
xtest = np.array(xtest[[f1,f2]])
ytest = np.array(ytest)
x = pd.DataFrame(basex)
y = basey
plt.scatter(x.iloc[:, 0], x.iloc[:,1], c=y, s=15)
plt.scatter(0.25, 0.2, c = "red", marker = 'x', s=100)