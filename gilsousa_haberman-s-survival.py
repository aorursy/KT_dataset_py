



import pandas

from pandas.tools.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn import cross_validation

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from matplotlib.ticker import FormatStrFormatter
#1. Age of patient at time of operation (numerical)

#2. Patient's year of operation (year - 1900, numerical)

#3. Number of positive axillary nodes detected (numerical)

#4. Survival status (class attribute)

         #1 = the patient survived 5 years or longer

         #2 = the patient died within 5 year

        

# Load dataset

url = "../input/haberman.csv"

names = ['Age', 'Year operation', 'Axillary nodes detected', 'Survival status']

dataset = pandas.read_csv(url, names=names)
dataset.head(5)
dataset.describe()
dataset.plot()

plt.show()
# histograms

dataset.hist()

plt.show()
#I made an adaptation of this reference online 

#----> http://machinelearningmastery.com/machine-learning-in-python-step-by-step/



array = dataset.values

X = array[:,:3]

Y = array[:,3]

validation_size = 0.30

seed = 10

X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, 

test_size=validation_size, random_state=seed)
#I made an adaptation of this reference online 

#----> http://machinelearningmastery.com/machine-learning-in-python-step-by-step/





# Test options and evaluation metric

num_folds = 20

num_instances = len(X_train)

seed = 10

scoring = 'accuracy'
#I made an adaptation of this reference online 

#----> http://machinelearningmastery.com/machine-learning-in-python-step-by-step/







# Spot Check Algorithms

algorithms = []

algorithms.append(('LR', LogisticRegression()))

algorithms.append(('LDA', LinearDiscriminantAnalysis()))

algorithms.append(('KNN', KNeighborsClassifier()))

algorithms.append(('CART', DecisionTreeClassifier()))

algorithms.append(('NB', GaussianNB()))

algorithms.append(('SVM', SVC()))

algorithms.append(('NN', MLPClassifier()))

algorithms.append(('RFC', RandomForestClassifier()))

# evaluate each model in turn

results = []

names = []

for name, algorithm in algorithms:

    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)

    cv_results = cross_validation.cross_val_score(algorithm, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Make predictions on validation dataset

knn =  GaussianNB()

knn.fit(X_train, Y_train)

predictions = knn.predict(X_validation)

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))
df_data = {'Age': [30,34, 35,38,40,50,43,45,34,34,46,50,45,38,42],

           'Year os operations': [65,64,63,64,66,64,64,64,63,63,64,67,64,65,67],

           'axillary nodes detected': [4,10,15,8,40,25,23,40,3,40,3,1,4,2,4]}

df = pandas.DataFrame(df_data)

print(df)
df.plot()

plt.show()
prediction = knn.predict(df)
print("Prediction of data survival status: {}".format(prediction))
plt.plot(prediction)

plt.ylabel('Status survival')

plt.xlabel('index = number of Occurrences')

plt.show()


