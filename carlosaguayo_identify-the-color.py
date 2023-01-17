import pandas as pd

dataset = pd.read_csv('../input/colors.csv').values



print (dataset[0])

print (dataset[1])

print (dataset[80])

print (dataset[81])

print (dataset[-2])

print (dataset[-1])

print (dataset.shape)
# scikit-learn is a Machine Learning library. It features various classification, 

# regression and clustering algorithms and is designed to interoperate with Python libraries like NumPy.

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder



import matplotlib.pyplot as plt

%matplotlib inline
blues = dataset[dataset[:,-1] == 'blue']

reds = dataset[dataset[:,-1] == 'red']

greens = dataset[dataset[:,-1] == 'green']



plt.scatter(x=reds[:,0], y=reds[:,1], c='red')

plt.scatter(x=greens[:,0], y=greens[:,1], c='green')

plt.scatter(x=blues[:,0], y=blues[:,1], c='blue')
x = dataset[:, :-1]

y = dataset[:, -1]



print (x.shape)

print (y.shape)

label_encoder = LabelEncoder()

label_encoded_y = label_encoder.fit_transform(y)



print (y)

print (label_encoded_y)
x_training, x_test, y_training, y_test = train_test_split(x,

                                                          label_encoded_y,

                                                          test_size=0.33)
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(x_training, y_training)
y_pred = knn.predict(x_test)



accuracy = accuracy_score(y_test, y_pred)

print("Accuracy KNN: %.2f%%" % (accuracy * 100.0))
my_point_x = 130

my_point_y = 50



import numpy as np

my_point = np.array([my_point_x, my_point_y])[:, np.newaxis].T



my_predicted_point = knn.predict(my_point)

print (label_encoder.inverse_transform(my_predicted_point))





my_point_x = 100

my_point_y = 80



import numpy as np

my_point = np.array([my_point_x, my_point_y])[:, np.newaxis].T



my_predicted_point = knn.predict(my_point)

print (label_encoder.inverse_transform(my_predicted_point))


