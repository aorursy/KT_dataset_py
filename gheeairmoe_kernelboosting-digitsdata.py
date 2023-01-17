import numpy

import pandas

import matplotlib.pyplot as plot

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

# REFERENCE: https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

digits = load_digits() # 1797 total image samples
'''IMPROVE THE DECISION TREE FROM PREVIOUS ANALYSIS BY USING BOOSTING'''

# Split the data - training set and testing set (for generalization)

featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(digits.data, digits.target, test_size=0.2, random_state=1234)

# Split the training set 

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.6, random_state=1234)



'''Determine the best n_estimator parameter value'''

# A dataframe keeping track of the progress and for plotting later

dataFrameE = pandas.DataFrame(columns=['n_estimators', 'Accuracy'])

# The range

estimatorRange = list(range(1, 100))

# The best  value

bestEsti = -1

# The highest accuracy

highestAccuracy = 0.0



# Iterate through the range of values for parameter n_estimators

for esti in estimatorRange:

    # Create the decision tree classifier with hyperparameters we analyzed and found isolated optimal values for highest accuracy

    tree = DecisionTreeClassifier(max_depth=11, min_samples_split=2, min_impurity_decrease=0.0,random_state=1234) 

    model = AdaBoostClassifier(random_state=1234, base_estimator=tree, n_estimators=esti)

                                                             

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

    # Save the data and plot later

    dataFrameE = dataFrameE.append({'n_estimator':esti, 'Accuracy':modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestEsti = esti

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(15, 6))

plot.xlabel('n_estimator', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Boosting Decision Tree on n_estimator Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('n_estimator', 'Accuracy', data=dataFrameE, marker='o')

plot.text(bestEsti, highestAccuracy, "Best Value: {}\nAccuracy: {}%".format(bestEsti, highestAccuracy), horizontalalignment='left', verticalalignment='center_baseline', color='red')

plot.grid()

'''Determine the best learning_rate param value'''

# Split the training set above into 2 sets: 70% training set and 30% validation set '''since decisionTree is used here and it's best size was 70%

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.6, random_state=1234)



# A dataframe keeping track of the progress and for plotting later

dataFrameLR = pandas.DataFrame(columns=['learning_rate', 'Accuracy'])

# The range

rateRange = numpy.arange(0.1, 2.1, 0.1)#list(range(1, 100))

# The best  value

bestRate = -1

# The highest accuracy

highestAccuracy = 0.0



# Iterate through the range of values for parameter n_estimators

for rate in rateRange:

    # Create the decision tree classifier with the current ranged-value and all other params set to default

    tree = DecisionTreeClassifier(max_depth=11, min_samples_split=2, min_impurity_decrease=0.0,random_state=1234) 

    model = AdaBoostClassifier(random_state=1234, base_estimator=tree, n_estimators=1, learning_rate=rate)

                                                             

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

    # Save the data and plot later

    dataFrameLR = dataFrameLR.append({'learning_rate':rate, 'Accuracy':modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestRate = rate

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(15, 6))

plot.xlabel('learning_rate', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Boosting Decision Tree on learning_rate Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('learning_rate', 'Accuracy', data=dataFrameLR, marker='o')

plot.text(bestRate, highestAccuracy, "Best Value: {}\nAccuracy: {}%".format(bestRate, highestAccuracy), horizontalalignment='left', verticalalignment='center_baseline', color='blue')

plot.grid()
'''What is accuracy based on the two most accurate param values analyzed'''

# Split the data - training set and testing set (for generalization)

featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(digits.data, digits.target, test_size=0.2, random_state=1234)

# Split the training set 

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.6, random_state=1234)

# Create the decision tree classifier with the current ranged-value and all other params set to default

tree = DecisionTreeClassifier(max_depth=11, min_samples_split=2, min_impurity_decrease=0.0,random_state=1234) 

model = AdaBoostClassifier(random_state=1234, base_estimator=tree, learning_rate=0.1, n_estimators=1)



# Train the model

model.fit(featuresTrainingSet, targetTrainingSet)



# Base on what the model has learned set it to predict the outcome of the validation set

modelPrediction = model.predict(featuresValidationSet)



# Determine the model's accuracy on the validation set

modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

print("n_estimator = {}".format(1))

print("learning_rate = {}".format(0.1))

print("model accuracy: {0:.2f}%".format(modelAccuracy))
'''EVALUATE AGAINST TEST DATASET'''

# Create the decision tree classifier with hyperparameters we analyzed and found isolated optimal values for highest accuracy

tree = DecisionTreeClassifier(max_depth=11, min_samples_split=2, min_impurity_decrease=0.0,random_state=1234)

model = AdaBoostClassifier(random_state=1234, base_estimator=tree, learning_rate=0.1, n_estimators=1)

# Train the model

model.fit(featuresTrainingSet, targetTrainingSet)

# Base on what the model has learned set it to predict the outcome of the TEST set

modelPrediction = model.predict(featuresTestingSet)

# Determine the model's accuracy on the TEST set

modelAccuracy = accuracy_score(targetsTestingSet, modelPrediction)*100

print("Isolated Param Value Discovery\nTest-set Accuracy: {}\nlearning_rate: {}\nn_estimators: {}".format(modelAccuracy, 0.1, 1))
'''USE CROSS VALIDATION ON ISOLATED VALUES'''

from sklearn.model_selection import cross_validate

# Create the decision tree classifier with hyperparameters we analyzed and found isolated optimal values for highest accuracy

tree = DecisionTreeClassifier(max_depth=11, min_samples_split=2, min_impurity_decrease=0.0,random_state=1234)

model = AdaBoostClassifier(random_state=1234, base_estimator=tree, learning_rate=0.1, n_estimators=1)

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