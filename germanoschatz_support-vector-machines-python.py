import numpy as np
import seaborn as sns
import pandas as pd  
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score


from sklearn.datasets import load_iris
#dict_keys(['data', 'target', 'feature_names', 'DESCR'])
iris = load_iris()
dataset = pd.DataFrame(iris.data, columns=iris.feature_names)
dataset['Species'] = iris.target
print(dataset.head())
#Design the target and feature variables
X = dataset.drop(['Species'], axis=1)
Y = dataset['Species']

#split the dataset 90% and 10%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=1)

#building the model using sklearn
model = SVC(kernel='linear')
model.fit(X_train, Y_train)

# model evaluation for testing set
y_test_predict = model.predict(X_test)

cm=confusion_matrix(Y_test, y_test_predict)
accuracy = np.trace(cm)/np.sum(cm)
print("Accuracy:",accuracy)
#-----pair PLOT------
from mlxtend.plotting import plot_decision_regions
plot=sns.pairplot(data=dataset, hue='Species', palette='Set2')
#plot.savefig("iris_pair.png")
#-----Boundaries PLOT------

model = SVC(kernel='linear').fit(X.iloc[:,[0,2]],Y)
fig=plt.figure()
plot_decision_regions(np.array(X.iloc[:,[0,2]]),np.array(Y), clf=model,legend=2)

plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.title('SVM on Iris')
plt.show()
#fig.savefig("SVC_IRIS.png",dpi=120)
#------SVR BOSTON-----------
from sklearn.svm import SVR
from sklearn.datasets import load_boston 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#dict_keys(['data', 'target', 'feature_names', 'DESCR'])
boston_dataset = load_boston()
dataset = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
dataset['MEDV'] = boston_dataset.target
print(dataset.head())
#Design the target and feature variables
X = dataset.drop(['MEDV'], axis=1)
Y = dataset['MEDV']

#split the dataset 90% and 10%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=1)
#building the model using sklearn
model = SVR(kernel='linear')
model.fit(X_train, Y_train)

# model evaluation for testing set
y_test_predict = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)
plt.scatter(Y_test, y_test_predict,  color='black')
print("R^2:",r2)
print("RMSE:",rmse)