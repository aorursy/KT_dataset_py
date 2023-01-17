import numpy

import pandas

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plot



# Import the data

allData = pandas.read_csv('/kaggle/input/mobile-price-classification/train.csv', sep=',')



# Names of the features

features = list(allData.columns)



# Separate the features/attributes from the target

featuresData = allData.values[:, :19]

targetData = allData.values[:, 20]



# Split the data - training set and testing set (for generalization)

featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(featuresData, targetData, test_size=0.2, random_state=1234)

# Split the training set above into 2 sets: 70% training set and 30% validation set '''since decisionTree is used here and it's best size was 70%

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.7, random_state=1234)



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

    # Create the decision tree classifier with the current ranged-value and all other params set to default

    tree = DecisionTreeClassifier(max_depth=10, min_samples_split=8, random_state=1234) 

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

plot.text(bestEsti, highestAccuracy, "Best Value: {}\nAccuracy: {}%".format(bestEsti, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

plot.grid()

'''Determine the best learning_rate param value'''

# Split the training set above into 2 sets: 70% training set and 30% validation set '''since decisionTree is used here and it's best size was 70%

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.7, random_state=1234)



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

    tree = DecisionTreeClassifier(max_depth=10, min_samples_split=8, random_state=1234) 

    model = AdaBoostClassifier(random_state=1234, base_estimator=tree, n_estimators=30, learning_rate=rate)

                                                             

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

plot.text(bestRate, highestAccuracy, "Best Value: {}\nAccuracy: {}%".format(bestRate, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

plot.grid()
'''What is accuracy based on the two most accurate param values analyzed'''

# Create the decision tree classifier with the current ranged-value and all other params set to default

tree = DecisionTreeClassifier(max_depth=10, min_samples_split=8, random_state=1234) 

model = AdaBoostClassifier(random_state=1234, base_estimator=tree, learning_rate=0.8, n_estimators=30)



# Train the model

model.fit(featuresTrainingSet, targetTrainingSet)



# Base on what the model has learned set it to predict the outcome of the validation set

modelPrediction = model.predict(featuresValidationSet)



# Determine the model's accuracy on the validation set

modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

print("n_estimator = {}".format(30))

print("learning_rate = {}".format(0.8))

print("model accuracy: {}%".format(modelAccuracy))
'''Brute force permutation in determining the best combination of learning_rate value and n_estimator value'''

# Split the training set above into 2 sets: 70% training set and 30% validation set '''since decisionTree is used here and it's best size was 70%

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.7, random_state=1234)

# A dataframe keeping track of the progress and for plotting later

dataFrameE = pandas.DataFrame(columns=['n_estimators', 'Accuracy'])

'''since the above graph show high accuracy in this range for this param'''

estimatorRange = list(range(28,60)) 

# The best  value

bestEsti = -1

# The highest accuracy

highestAccuracy = 0.0

# A dataframe keeping track of the progress and for plotting later

dataFrameLR = pandas.DataFrame(columns=['learning_rate', 'Accuracy'])

'''since the above graph show high accuracy in this range for this param'''

rateRange = numpy.arange(0.6, 1.3, 0.1) 

# The best  value

bestRate = -1

tree = DecisionTreeClassifier(max_depth=10, min_samples_split=8, random_state=1234)

# Iterate through the range of values for parameter n_estimators

for esti in estimatorRange:

    for rate in rateRange:

        # Create the decision tree classifier with the current ranged-value and all other params set to default

        model = AdaBoostClassifier(random_state=1234, base_estimator=tree, n_estimators=esti, learning_rate=rate)



        # Train the model

        model.fit(featuresTrainingSet, targetTrainingSet)



        # Base on what the model has learned set it to predict the outcome of the validation set

        modelPrediction = model.predict(featuresValidationSet)



        # Determine the model's accuracy on the validation set

        modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100



        # Save the data and plot later

        dataFrameE = dataFrameE.append({'n_estimator':esti, 'Accuracy':modelAccuracy}, ignore_index=True)

        # Save the data and plot later

        dataFrameLR = dataFrameLR.append({'learning_rate':rate, 'Accuracy':modelAccuracy}, ignore_index=True)



        # Save the highest accuracy based on the training size

        if modelAccuracy > highestAccuracy:

            bestEsti = esti

            bestRate = rate

            highestAccuracy = modelAccuracy

        

# plot.figure(figsize=(15, 6))

# plot.xlabel('n_estimator / learning_rate', color='white')

# plot.ylabel('Accuracy %', color='white')

# plot.title("LEARNING CURVE: Boosting Decision Tree on n_estimator / learning_rate Param", color='white')

# plot.xticks(color='white')

# plot.yticks(color='white')

# plot.plot('n_estimator', 'Accuracy', data=dataFrameE, marker='o')

# plot.plot('learning_rate', 'Accuracy', data=dataFrameLR, marker='o', color='red')

# # plot.text(bestEsti, highestAccuracy, "Best Value: {}\nAccuracy: {}%".format(bestEsti, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

# plot.grid()



graph = plot.figure(figsize=(15,6))

graph1 = graph.add_subplot(111, label='1')

graph1.scatter('n_estimator', 'Accuracy', data=dataFrameE, marker='o')

graph1.set_xlabel('n_estimator', color='white')

graph1.set_ylabel('Accuracy %', color='white')

graph1.xaxis.tick_top()

graph1.xaxis.set_label_position('top')

graph2 = graph.add_subplot(111, label='2', frame_on=False)

graph2.scatter('learning_rate', 'Accuracy', data=dataFrameLR, marker='o', color='red')

graph2.set_xlabel('learning_rate', color='white')

graph2.set_ylabel('Accuracy %', color='white')

# plot.text(bestEsti, highestAccuracy, "Best Estimator Value: {}".format(bestEsti), horizontalalignment='right', verticalalignment='center_baseline')

# plot.text(bestRate, highestAccuracy, "Best Learning Rate Value: {}".format(bestRate), horizontalalignment='right', verticalalignment='center_baseline')



print("Best n_estimator value: {}".format(bestEsti))

print("Best learning_rate value: {0:.2f}".format(bestRate))

print("Above params reach {0:.2f}% Accuracy".format(highestAccuracy))
'''Determin over/under fitting based on running model against test set'''

'''TEST ACCURACY OF HYPERPARAMETERS ANALYZED AND OPTIMIZED IN ISOLATION - RUN AGAINST TEST SET'''

# Create the decision tree classifier with hyperparameters we analyzed and found isolated optimal values for highest accuracy

tree = DecisionTreeClassifier(max_depth=10, min_samples_split=8, random_state=1234) 

model = AdaBoostClassifier(random_state=1234, base_estimator=tree, learning_rate=0.8, n_estimators=30)

# Train the model

model.fit(featuresTrainingSet, targetTrainingSet)

# Base on what the model has learned set it to predict the outcome of the TEST set

modelPrediction = model.predict(featuresTestingSet)

# Determine the model's accuracy on the TEST set

modelAccuracy = accuracy_score(targetsTestingSet, modelPrediction)*100

print("Isolated Param Value Discovery\nTest-set Accuracy: {}\nlearning_rate: {}\nn_estimators: {}".format(modelAccuracy, 0.8, 30))

print()



'''TEST ACCURACY OF HYPERPARMETES ANALYZED THROUGH PERMUTATIONS OF TWO PARAMS'''

# Create the decision tree classifier with hyperparameters 

tree = DecisionTreeClassifier(max_depth=10, min_samples_split=8, random_state=1234) 

model = AdaBoostClassifier(random_state=1234, base_estimator=tree, learning_rate=1.0, n_estimators=58)

# Train the model

model.fit(featuresTrainingSet, targetTrainingSet)

# Base on what the model has learned set it to predict the outcome of the TEST set

modelPrediction = model.predict(featuresTestingSet)

# Determine the model's accuracy on the TEST set

modelAccuracy = accuracy_score(targetsTestingSet, modelPrediction)*100

print("Permutated Param Value Discovery\nTest-set Accuracy: {}\nlearning_rate: {}\nn_estimators: {}".format(modelAccuracy, 1.0, 58))
'''USE CROSS VALIDATION ON ISOLATED VALUES'''

from sklearn.model_selection import cross_validate

# Create the decision tree classifier with hyperparameters we analyzed and found isolated optimal values for highest accuracy

tree = DecisionTreeClassifier(max_depth=10, min_samples_split=8, random_state=1234) 

model = AdaBoostClassifier(random_state=1234, base_estimator=tree, learning_rate=0.8, n_estimators=30)

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
'''USE CROSS VALIDATION OF PERMUTATED VALUES'''

from sklearn.model_selection import cross_validate

tree = DecisionTreeClassifier(max_depth=10, min_samples_split=8, random_state=1234) 

model = AdaBoostClassifier(random_state=1234, base_estimator=tree, learning_rate=1.0, n_estimators=58)

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