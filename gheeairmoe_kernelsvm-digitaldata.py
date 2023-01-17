import numpy

import pandas

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plot

from sklearn import svm

from sklearn.datasets import load_digits

digits = load_digits() # 1797 total image samples
'''EVALUATE & ANALYZE BASED ON DEFAULT PARAMS AND VALUES & ONLY TRAINING SET AND TESTING SET'''

# A dataframe keeping track of the progress and for plotting later

dataFrameT = pandas.DataFrame(columns=['Training Size', 'Accuracy'])

# Keep track of the best training size

bestTrainingSize = 0

# Keep track of the highest accuracy

highestAccuracy = 0.0

# Incease the training size from 10 to 90 percent of the whole set's size; train the model; check its accuracy; save to plot later

for size in range(1, 10):

    # Split the data - training set and testing set (for generalization)

    featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(digits.data, digits.target, train_size=size/10, random_state=1234)

    # Create Classifier

    model = svm.SVC(random_state=1234, gamma='scale')

    # Train the model

    model.fit(featuresWholeTrainingSet, targetsWholeTrainingSet)

    # Make the model predict

    modelPrediction = model.predict(featuresTestingSet)

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetsTestingSet, modelPrediction)*100

    

    # Save the data to plot later

    trainingSize = size*10

    dataFrameT = dataFrameT.append({'Training Size': trainingSize, 'Accuracy': modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestTrainingSize = trainingSize

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(12, 6))

plot.xlabel('Training Size', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Default SVM Classifier - Test set", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('Training Size', 'Accuracy', data=dataFrameT, marker='o')

plot.text(bestTrainingSize, highestAccuracy, "Best Training size: {}\nAccuracy: {}%".format(bestTrainingSize, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

plot.grid()
'''EVALUATE & ANALYZE BASED ON DEFAULT PARAMS AND VALUES & USING A VALIDATION SET'''

# Split the data - training set and testing set (for generalization)

featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(digits.data, digits.target, test_size=0.2, random_state=1234)

# A dataframe keeping track of the progress and for plotting later

dataFrameV = pandas.DataFrame(columns=['Training Size', 'Accuracy'])

# Keep track of the best training size

bestTrainingSize = 0

# Keep track of the highest accuracy

highestAccuracy = 0.0

# Incease the training size from 10 to 90 percent of the whole set's size; train the model; check its accuracy; save to plot later

for size in range(1, 10):

    # Split the training set above into 2 sets: smaller training set and a validation set

    featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=size/10, random_state=1234)

    # Create Classifier

    model = svm.SVC(random_state=1234, gamma='scale')

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    # Make the model predict

    modelPrediction = model.predict(featuresValidationSet)

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

    # Save the data to plot later

    trainingSize = size*10

    dataFrameV = dataFrameV.append({'Training Size': trainingSize, 'Accuracy': modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestTrainingSize = trainingSize

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(12, 6))

plot.xlabel('Training Size', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Default SVM Classifier - Validation Set", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('Training Size', 'Accuracy', data=dataFrameV, marker='o')

plot.text(bestTrainingSize, highestAccuracy, "Best Training size: {}\nAccuracy: {}%".format(bestTrainingSize, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline', color='red')

plot.grid()
'''Evaluate the kernel parameter'''

# A dataframe keeping track of the progress and for plotting later

dataFrameK = pandas.DataFrame(columns=['linear', 'poly', 'rbf', 'sigmoid','Accuracy'])

# The ranges 

kernalRange = ['linear', 'poly', 'rbf', 'sigmoid']

# The best value

bestkernal = 'colonel'

# The highest accuracy

highestAccuracy = 0.0

# Split the training set above into 2 sets: 60% training set and 40% validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.9, random_state=1234)

# Iterate through the range of values for parameter max_depth

for colonel in kernalRange:

    # Create the classifier with the current ranged-value and all other params set to default

    model = svm.SVC(random_state=1234, gamma='scale', kernel=colonel) 

                                                             

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

    # Save the data and plot later

    dataFrameK = dataFrameK.append({colonel:colonel, 'Accuracy':modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestkernal = colonel

        highestAccuracy = modelAccuracy



plot.figure(figsize=(15, 6))

plot.xlabel('Kernel Type', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: SVC Classifier Range of kernel Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('linear', dataFrameK['Accuracy'][0], marker='o')

plot.text('linear', dataFrameK['Accuracy'][0], "  {}%".format(dataFrameK['Accuracy'][0]))

plot.plot('poly', dataFrameK['Accuracy'][1], marker='o')

plot.text('poly', dataFrameK['Accuracy'][1], "  {}%".format(dataFrameK['Accuracy'][1]))

plot.plot('rbf', dataFrameK['Accuracy'][2], marker='o')

plot.text('rbf', dataFrameK['Accuracy'][2], "  {}%".format(dataFrameK['Accuracy'][2]))

plot.plot('sigmoid', dataFrameK['Accuracy'][3], marker='o')

plot.text('sigmoid', dataFrameK['Accuracy'][3], "{}%".format(dataFrameK['Accuracy'][3]), horizontalalignment='right')

plot.grid()
'''we will go for rbf, since it was in lecture'''

'''Evaluate the C parameter'''

# A dataframe keeping track of the progress and for plotting later

dataFrameC = pandas.DataFrame(columns=['C', 'Accuracy'])

# The range

cRange = numpy.arange(0.1, 1.1, 0.1)

# The best value

bestC = -1

# The highest accuracy

highestAccuracy = 0.0

# Split the training set above into 2 sets: 60% training set and 40% validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.9, random_state=1234)

# Iterate through the range of values for parameter max_depth

for value in cRange:

    # Create the classifier with the current ranged-value and all other params set to default

    model = svm.SVC(random_state=1234, gamma='scale', kernel='rbf', C=value)                                                           

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    # Save the data and plot later

    dataFrameC = dataFrameC.append({'C':value, 'Accuracy':modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestC = value

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(15, 6))

plot.xlabel('C', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: SVC Classifier Range of C Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('C', 'Accuracy', data=dataFrameC, marker='o')

plot.text(bestC, highestAccuracy, "Best C Value: {}\nAccuracy: {}%".format(bestC, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline', color='red')

plot.grid()

print("C=1 yields 99.30% same as C=0.5\nWe will use default values")
'''EVALUATE ON PARAM VALUE IDENTIFIED ON TEST DATASET'''

# Split the data - training set and testing set (for generalization)

featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(digits.data, digits.target, test_size=0.2, random_state=1234)

# Split the training set above into 2 sets: 60% training set and 40% validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.9, random_state=1234)

# Create the classifier with the current ranged-value and all other params set to default

model = svm.SVC(random_state=1234, gamma='scale', kernel='rbf', C=1.0)                                                           

# Train the model

model.fit(featuresTrainingSet, targetTrainingSet)    

# Base on what the model has learned set it to predict the outcome of the validation set

modelPrediction = model.predict(featuresTestingSet)

# Determine the model's accuracy on the validation set

modelAccuracy = accuracy_score(targetsTestingSet, modelPrediction)*100

print('Model Accuracy against the testing set: {}%'.format(modelAccuracy))
'''USE CROSS VALIDATION'''

from sklearn.model_selection import cross_validate

# Split the data - training set and testing set (for generalization)

featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(digits.data, digits.target, test_size=0.2, random_state=1234)

# Split the training set above into 2 sets: 60% training set and 40% validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.9, random_state=1234)

# Optimal identified params

model = svm.SVC(random_state=1234, gamma='scale', kernel='rbf', C=1.0)

# cross_val_score will split the training data for me - in this case 4 sets

crossVal = cross_validate(model,featuresWholeTrainingSet, targetsWholeTrainingSet, cv=4, scoring='accuracy', return_estimator=True)

print("The accuracy scores of the 4 models under cross-validation: {}".format(list(crossVal['test_score'])))

# Get the classifier that did the best

scores = list(crossVal['test_score'])

index = scores.index(max(scores))

bestModel = crossVal['estimator'][index]

print("Best Model's accuracy using cross-validation: {0:.2f}%".format(scores[index]*100))



'''NOW RUN ON TEST DATASET'''

# Base on what the model has learned set it to predict the outcome of the TEST set

bestModelPrediction = bestModel.predict(featuresTestingSet)

# Determine the model's accuracy on the TEST set

modelAccuracy = accuracy_score(targetsTestingSet, bestModelPrediction)*100

print("Best Model's accuracy against Test Dataset: {0:.2f}%".format(modelAccuracy))