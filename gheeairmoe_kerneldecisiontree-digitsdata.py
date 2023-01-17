import numpy

import pandas

import matplotlib.pyplot as plot

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

# REFERENCE: https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

digits = load_digits() # 1797 total image samples
'''EVALUATE & ANALYZE BASED ON DEFAULT PARAMS AND VALUES & ONLY TRAINING SET AND TESTING SET'''

# A dataframe keeping track of the progress and for plotting later

dataFrame = pandas.DataFrame(columns=['Training Size', 'Accuracy'])

# Keep track of the best training size

bestTrainingSize = 0

# Keep track of the highest accuracy

highestAccuracy = 0.0

# Incease the training size from 10 to 90 percent of the whole set's size; train the model; check its accuracy; save to plot later

for size in range(1, 10):

    # Split the data - training set and testing set (for generalization)

    featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(digits.data, digits.target, train_size=size/10, random_state=1234)

    # Create the decision tree classifier with the default parameters

    model = DecisionTreeClassifier(criterion='gini', splitter='best', 

                               max_depth=None, min_samples_split=2, 

                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, 

                               max_features=None, random_state=1234, 

                               max_leaf_nodes=None, min_impurity_decrease=0.0, 

                               min_impurity_split=None, class_weight=None, 

                               presort=False)

    

    # Train the model

    model.fit(featuresWholeTrainingSet, targetsWholeTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresTestingSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetsTestingSet, modelPrediction)*100

    

    # Save the data to plot later

    trainingSize = size*10

    dataFrame = dataFrame.append({'Training Size': trainingSize, 'Accuracy': modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestTrainingSize = trainingSize

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(12, 6))

plot.xlabel('Training Size', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Default Decision Tree Classifier Trg/Test dataset", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('Training Size', 'Accuracy', data=dataFrame, marker='o')

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

    

    # Create the decision tree classifier with the default parameters

    model = DecisionTreeClassifier(criterion='gini', splitter='best', 

                               max_depth=None, min_samples_split=2, 

                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, 

                               max_features=None, random_state=1234, 

                               max_leaf_nodes=None, min_impurity_decrease=0.0, 

                               min_impurity_split=None, class_weight=None, 

                               presort=False)

    

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

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

plot.title("LEARNING CURVE: Default Decision Tree Classifier Trg/Val set", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('Training Size', 'Accuracy', data=dataFrameV, marker='o')

plot.text(bestTrainingSize, highestAccuracy, "Best Training size: {}\nAccuracy: {}%".format(bestTrainingSize, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

plot.grid()

print(model)
# Split the data - training set and testing set (for generalization)

featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(digits.data, digits.target, test_size=0.2, random_state=1234)

# A dataframe keeping track of the progress and for plotting later

dataFrameMD = pandas.DataFrame(columns=['Max_depth', 'Accuracy'])

# The ranges of the tree depth

depthRange = list(range(1,16))

# The best depth value

bestDepth = -1

# The highest accuracy

highestAccuracy = 0.0

# Split the training set above into 2 sets: 70% training set and 30% validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.6, random_state=1234)

# Iterate through the range of values for parameter max_depth

for depth in depthRange:

    # Create the decision tree classifier with the current ranged-value and all other params set to default

    model = DecisionTreeClassifier(max_depth=depth,random_state=1234) 

                                                             

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

    # Save the data and plot later

    dataFrameMD = dataFrameMD.append({'Max_depth':depth, 'Accuracy':modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestDepth = depth

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(12, 6))

plot.xlabel('Tree Depth', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Decision Tree Classifier Range of max_depth Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('Max_depth', 'Accuracy', data=dataFrameMD, marker='o')

plot.text(bestDepth, highestAccuracy, "Best Depth: {}\nAccuracy: {}%".format(bestDepth, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

plot.grid()
# A dataframe keeping track of the progress and for plotting later

dataFrameMSS = pandas.DataFrame(columns=['min_sample_split', 'Accuracy'])

# The ranges of the tree depth

splitRange = list(range(2,336, 2))

# The best split value

bestSplit = -1

# The highest accuracy

highestAccuracy = 0.0

# Split the training set above into 2 sets: 70% training set and 30% validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.6, random_state=1234)

# Iterate through the range of values for parameter min_sample_split

for splitValue in splitRange:

    # Create the decision tree classifier with the current ranged-value and all other params set to default

    model = DecisionTreeClassifier(max_depth=11, min_samples_split=splitValue,random_state=1234) 

                                                        

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100



    # Save the data and plot later

    dataFrameMSS = dataFrameMSS.append({'min_sample_split':splitValue, 'Accuracy':modelAccuracy}, ignore_index=True)



    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestSplit = splitValue

        highestAccuracy = modelAccuracy

        

# print(dataFrameMSS)

plot.figure(figsize=(24, 6))

plot.xlabel('min_sample_split value', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Decision Tree Classifier Range of min_sample_split Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('min_sample_split', 'Accuracy', data=dataFrameMSS, marker='o')

plot.text(bestSplit, highestAccuracy, "Best Split: {}\nAccuracy: {}%".format(bestSplit, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline', color='red')

plot.grid()
# A dataframe keeping track of the progress and for plotting later

dataFrameMID = pandas.DataFrame(columns=['min_impurity_decrease', 'Accuracy'])

# The ranges of the min_impurity_decrease

impureRange = list(numpy.arange(0.0, 0.8, 0.01))

# The best split value

bestSplit = -1

# The highest accuracy

highestAccuracy = 0.0

# Split the training set above into 2 sets: 70% training set and 30% validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.6, random_state=1234)

# Iterate through the range of values for parameter min_sample_split

for splitValue in impureRange:

    # Create the decision tree classifier with the current ranged-value and all other params set to default

    model = DecisionTreeClassifier(max_depth=11, min_samples_split=2, min_impurity_decrease=splitValue,random_state=1234) 

                                                        

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100



    # Save the data and plot later

    dataFrameMID = dataFrameMID.append({'min_impurity_decrease':splitValue, 'Accuracy':modelAccuracy}, ignore_index=True)



    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestSplit = splitValue

        highestAccuracy = modelAccuracy

        

# print(dataFrameMSS)

plot.figure(figsize=(12, 6))

plot.xlabel('min_impurity_decrease', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Decision Tree Classifier Range of min_impurity_decrease Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('min_impurity_decrease', 'Accuracy', data=dataFrameMID, marker='o')

plot.text(bestSplit, highestAccuracy, "Best Split: {}\nAccuracy: {}%".format(bestSplit, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline', color='red')

plot.grid()
'''TEST ACCURACY OF HYPERPARAMETERS ANALYZED - RUN AGAINST TEST SET'''

# Split the data - training set and testing set (for generalization)

featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(digits.data, digits.target, test_size=0.2, random_state=1234)

# Split the training set into 2 sets: 70% training set and 30% validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.6, random_state=1234)

# Create the decision tree classifier with hyperparameters we analyzed and found isolated optimal values for highest accuracy

model = DecisionTreeClassifier(max_depth=11, min_samples_split=2, min_impurity_decrease=0.0,random_state=1234)

# Train the model

model.fit(featuresTrainingSet, targetTrainingSet)

# Base on what the model has learned set it to predict the outcome of the TEST set

modelPrediction = model.predict(featuresTestingSet)

# Determine the model's accuracy on the TEST set

modelAccuracy = accuracy_score(targetsTestingSet, modelPrediction)*100

print("Decision Tree's generalization using the test dataset: {0:.2f}".format(modelAccuracy))
'''USE CROSS VALIDATION'''

from sklearn.model_selection import cross_validate

# Optimal identified params

model = DecisionTreeClassifier(max_depth=11, min_samples_split=2, min_impurity_decrease=0.0,random_state=1234)

# cross_val_score will split the training data for me - in this case 4 sets

crossVal = cross_validate(model,featuresWholeTrainingSet, targetsWholeTrainingSet, cv=4, scoring='accuracy', return_estimator=True)

print("The accuracy scores of the 4 models under cross-validation: {}".format(list(crossVal['test_score'])))

# Get the classifier that did the best

scores = list(crossVal['test_score'])

index = scores.index(max(scores))

bestModel = crossVal['estimator'][index]

print("Best Model's accuracy using cross-validation on Trg/Val datasets: {0:.2f}%".format(scores[index]*100))



'''NOW RUN ON TEST DATASET'''

# Base on what the model has learned set it to predict the outcome of the TEST set

bestModelPrediction = bestModel.predict(featuresTestingSet)

# Determine the model's accuracy on the TEST set

modelAccuracy = accuracy_score(targetsTestingSet, bestModelPrediction)*100

print("Best Model's accuracy against Test Dataset: {0:.2f}%".format(modelAccuracy))