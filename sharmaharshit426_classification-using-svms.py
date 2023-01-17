#For data visualization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')



#For loading our Wine dataset

from sklearn.datasets import load_wine



#For seperating the data into train and test splits

from sklearn.model_selection import train_test_split



#Importing the SVM module

from sklearn import svm



#We will be using metrics to calculate accuracy of our models

from sklearn import metrics
raw_data = load_wine()

raw_data
feature_names = raw_data['feature_names']

parameters, target = load_wine(return_X_y=True)

print(feature_names, parameters, target)
#We will need group parameters before we can plot them.

def param_seperator(param_array, index):

    seperated_parameter = []

    for i in range(len(param_array)):

        seperated_parameter.append(param_array[i][index])

    return seperated_parameter

alcohol = param_seperator(parameters, 0)

malic_acid = param_seperator(parameters, 1)

ash = param_seperator(parameters, 2)

alcalinity_of_ash = param_seperator(parameters, 3)

magnesium = param_seperator(parameters, 4) 

total_phenols = param_seperator(parameters, 5) 

flavanoids = param_seperator(parameters, 6) 

nonflavanoid_phenols = param_seperator(parameters, 7) 

proanthocyanins = param_seperator(parameters, 8)

color_intensity = param_seperator(parameters, 9)

hue = param_seperator(parameters, 10)

od280byod315_of_diluted_wines = param_seperator(parameters, 11)

proline = param_seperator(parameters, 12)

ax = sns.scatterplot(x=alcohol, y=color_intensity, hue=target, markers=["o", "v", "^"])

ax.set(xlabel='Alcohol', ylabel='Colour Intensity')

new_labels = ['Class 0', 'Class 1', 'Class 2']

plt.legend(labels=new_labels)
ax = sns.swarmplot(target, flavanoids)

ax.set(xlabel='Classes', ylabel='Flavonoids')

new_labels = ['Class 0', 'Class 1', 'Class 2']

plt.legend(labels=new_labels)

#Splitting our data into train and test sets

X_train, X_test, Y_train, Y_test= train_test_split(parameters, target, test_size=0.1)

model = svm.SVC()

model.fit(X_train, Y_train)

predictions = model.predict(X_test)

acc = metrics.accuracy_score(Y_test, predictions)

print(acc)
model_v2 =svm.SVC(kernel='linear')

model_v2.fit(X_train, Y_train)

predictions = model_v2.predict(X_test)

acc = metrics.accuracy_score(Y_test, predictions)

print(acc)