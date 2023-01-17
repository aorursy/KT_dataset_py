import numpy

import pandas

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plot

from sklearn.neural_network import MLPClassifier



# Import the data

allData = pandas.read_csv('/kaggle/input/mobile-price-classification/train.csv', sep=',')



# Names of the features

features = list(allData.columns)



# Separate the features/attributes from the target

featuresData = allData.values[:, :19]

targetData = allData.values[:, 20]



# Split the data - training set and testing set (for generalization)

featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(featuresData, targetData, test_size=0.2, random_state=1234)



'''Determine the best training size per default classifer parameter values'''

# A dataframe keeping track of the progress and for plotting later

dataFrame = pandas.DataFrame(columns=['Training Size', 'Accuracy'])

# Keep track of the best training size

bestTrainingSize = 0

# Keep track of the highest accuracy

highestAccuracy = 0.0

mprint = True

# Incease the training size from 10 to 90 percent of the whole set's size; train the model; check its accuracy; save to plot later

for size in range(1, 10):

    # Split the training set above into 2 sets: smaller training set and a validation set

    featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=size/10, random_state=1234)

    

    # Create the decision tree classifier with the default parameters

    model = MLPClassifier(random_state=1234, solver='sgd', activation='logistic') # NOTE activation='identity' doesnt have convergence problem and 66% peak acc on dataset size 10%

    if mprint: 

        print(model)

        mprint = False

    

    print()

    print('Training size: {}'.format(size/10))

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

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

plot.title("LEARNING CURVE: Default Neural Network Classifier - Training set / Validation Set", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('Training Size', 'Accuracy', data=dataFrame, marker='o')

plot.text(bestTrainingSize, highestAccuracy, "Best Training size: {}\nAccuracy: {}%".format(bestTrainingSize, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline', color='red')

plot.grid()
'''Determine the best training size against the whole traning set size vs the test set size'''



'''Determine the best training size per default classifer parameter values'''

# A dataframe keeping track of the progress and for plotting later

dataFrameT = pandas.DataFrame(columns=['Training Size', 'Accuracy'])

# Keep track of the best training size

bestTrainingSize = 0

# Keep track of the highest accuracy

highestAccuracy = 0.0

mprint = True

# Incease the training size from 10 to 90 percent of the whole set's size; train the model; check its accuracy; save to plot later

for size in range(1, 10):

    # Split the data - training set and testing set (for generalization)

    featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(featuresData, targetData, train_size=size/10, random_state=1234)

    

    # Create the decision tree classifier with the default parameters

    model = MLPClassifier(random_state=1234, solver='sgd', activation='logistic') # NOTE activation='identity' doesnt have convergence problem and 66% peak acc on dataset size 10%

    if mprint: 

        print(model)

        mprint = False

    

    print()

    print('Training size: {}'.format(size/10))

    # Train the model

    model.fit(featuresWholeTrainingSet, targetsWholeTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

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

plot.title("LEARNING CURVE: Default Neural Network Classifier - All Training Set", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('Training Size', 'Accuracy', data=dataFrameT, marker='o')

plot.text(bestTrainingSize, highestAccuracy, "Best Training size: {}\nAccuracy: {}%".format(bestTrainingSize, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline', color='red')

plot.grid()
'''Determine best learning_rate value for best training size identified above'''

# Split the data - training set and testing set (for generalization)

featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(featuresData, targetData, test_size=0.2, random_state=1234)

# A dataframe keeping track of the progress and for plotting later

dataFrameLR = pandas.DataFrame(columns=['learning_rate', 'Accuracy'])

# The range

lrRange = ['constant', 'invscaling', 'adaptive']

# The best  value

bestlr = ''

# The highest accuracy

highestAccuracy = 0.0

# Split the training set above into 2 sets: smaller training set and a validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.8, random_state=1234)

# Iterate through the range of values for parameter learning_rate

for lr in lrRange:

    # Create the  with the current ranged-value and all other params set to default

    model = MLPClassifier(random_state=1234, solver='sgd', activation='logistic', learning_rate=lr)

    print(lr)

                                                             

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

    # Save the data and plot later

    dataFrameLR = dataFrameLR.append({'learning_rate':lr, 'Accuracy':modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestlr = lr

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(15, 6))

plot.xlabel('learning_rate', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Neural Network on learning_rate Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('learning_rate', 'Accuracy', data=dataFrameLR, marker='o')

plot.text(bestlr, highestAccuracy, "Best Value: {}\nAccuracy: {}%".format(bestlr, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

plot.grid()

print(model.n_layers_)

print(model.n_outputs_)

print(model.classes_)
'''Determine best learning_rate_INIT value for best training size identified above'''

# A dataframe keeping track of the progress and for plotting later

dataFrameLRI = pandas.DataFrame(columns=['learning_rate_init', 'Accuracy'])

# The range

lriRange = numpy.arange(0.00001, 0.01, 0.0005)

# The best  value

bestlri = -1

# The highest accuracy

highestAccuracy = 0.0

# Split the training set above into 2 sets: smaller training set and a validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.8, random_state=1234)

# Iterate through the range of values for parameter learning_rate_init

for lri in lriRange:

    # Create the  with the current ranged-value and all other params set to default

    model = MLPClassifier(random_state=1234, solver='sgd', activation='logistic', learning_rate='adaptive', learning_rate_init=lri)

    print(lri, end=' ')

                                                             

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

    # Save the data and plot later

    dataFrameLRI = dataFrameLRI.append({'learning_rate_init':lri, 'Accuracy':modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestlri = lri

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(15, 6))

plot.xlabel('learning_rate_init', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Neural Network on learning_rate_init Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('learning_rate_init', 'Accuracy', data=dataFrameLRI, marker='o')

plot.text(bestlri, highestAccuracy, "Best Value: {}\nAccuracy: {}%".format(bestlri, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

plot.grid()
'''Determine best learning_rate_INIT value for best training size identified above'''

# A dataframe keeping track of the progress and for plotting later

dataFrameLRI = pandas.DataFrame(columns=['learning_rate_init', 'Accuracy'])

# The range

lriRange = numpy.arange(0.00001, 0.002, 0.00001)

# The best  value

bestlri = -1

# The highest accuracy

highestAccuracy = 0.0

# Split the training set above into 2 sets: smaller training set and a validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.8, random_state=1234)

# Iterate through the range of values for parameter learning_rate_init

for lri in lriRange:

    # Create the  with the current ranged-value and all other params set to default

    model = MLPClassifier(random_state=1234, solver='sgd', activation='logistic', learning_rate='adaptive', learning_rate_init=lri)

                                                             

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

    # Save the data and plot later

    dataFrameLRI = dataFrameLRI.append({'learning_rate_init':lri, 'Accuracy':modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestlri = lri

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(15, 6))

plot.xlabel('learning_rate_init', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Neural Network on learning_rate_init Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('learning_rate_init', 'Accuracy', data=dataFrameLRI, marker='o')

plot.text(bestlri, highestAccuracy, "Best Value: {}\nAccuracy: {}%".format(bestlri, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

plot.grid()
'''Determine best momentum value for best training size identified above'''

# A dataframe keeping track of the progress and for plotting later

dataFrameM = pandas.DataFrame(columns=['momentum', 'Accuracy'])

# The range

momRange = numpy.arange(0.01, 1.0, 0.01)

# The best  value

bestMom = -1

# The highest accuracy

highestAccuracy = 0.0

# Split the training set above into 2 sets: smaller training set and a validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.8, random_state=1234)

# Iterate through the range of values for parameter momentum

for mom in momRange:

    # Create the  with the current ranged-value and all other params set to default

    model = MLPClassifier(random_state=1234, solver='sgd', activation='logistic', learning_rate='adaptive', learning_rate_init=0.00101, momentum=mom)

                                                             

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

    # Save the data and plot later

    dataFrameM = dataFrameM.append({'momentum':mom, 'Accuracy':modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestMom = mom

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(15, 6))

plot.xlabel('momentum', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Neural Network on momentum Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('momentum', 'Accuracy', data=dataFrameM, marker='o')

plot.text(bestMom, highestAccuracy, "Best Value: {}\nAccuracy: {}%".format(bestMom, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

plot.grid()
'''Determine best max_iter value for best parameters and their values identified above'''

# A dataframe keeping track of the progress and for plotting later

dataFrameMI = pandas.DataFrame(columns=['max_iter', 'Accuracy'])

# The range

iterRange = list(range(1, 502, 50))

# The best  value

bestIter = -1

# The highest accuracy

highestAccuracy = 0.0

# Split the training set above into 2 sets: smaller training set and a validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.8, random_state=1234)

# Iterate through the range of values for parameter momentum

for itera in iterRange:

    # Create the  with the current ranged-value and all other params set to default

    model = MLPClassifier(random_state=1234, solver='sgd', activation='logistic', learning_rate='adaptive', learning_rate_init=0.00101, momentum=0.9, max_iter=itera)

                                                             

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

    # Save the data and plot later

    dataFrameMI = dataFrameMI.append({'max_iter':itera, 'Accuracy':modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestIter = itera

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(15, 6))

plot.xlabel('max_iter', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Neural Network on max_iter Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('max_iter', 'Accuracy', data=dataFrameMI, marker='o')

plot.text(bestIter, highestAccuracy, "Best Value: {}\nAccuracy: {}%".format(bestIter, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

plot.grid()
'''Determine best hidden_layer_sizes value for best parameters and their values identified above'''

# A dataframe keeping track of the progress and for plotting later

dataFrameHLS = pandas.DataFrame(columns=['hidden_layer_sizes', 'Accuracy'])

# The range

layerRange = list(range(50, 1000, 50))

# The best  value

bestLayer = -1

# The highest accuracy

highestAccuracy = 0.0

# Split the training set above into 2 sets: smaller training set and a validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.8, random_state=1234)

# Iterate through the range of values for parameter momentum

for layer in layerRange:

    # Create the  with the current ranged-value and all other params set to default

    model = MLPClassifier(random_state=1234, solver='sgd', activation='logistic', learning_rate='adaptive', learning_rate_init=0.00101, momentum=0.9, max_iter=200, hidden_layer_sizes=layer)

                                                             

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

    # Save the data and plot later

    dataFrameHLS = dataFrameHLS.append({'hidden_layer_sizes':layer, 'Accuracy':modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestLayer = layer

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(15, 6))

plot.xlabel('hidden_layer_sizes', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Neural Network on hidden_layer_sizes Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('hidden_layer_sizes', 'Accuracy', data=dataFrameHLS, marker='o')

plot.text(bestLayer, highestAccuracy, "Best Value: {}\nAccuracy: {}%".format(bestLayer, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

plot.grid()
'''Determine best hidden_layer_sizes value for best parameters and their values identified above'''

# A dataframe keeping track of the progress and for plotting later

dataFrameHLS = pandas.DataFrame(columns=['hidden_layer_sizes', 'Accuracy'])

# The range

layerRange = list(range(400, 1500, 50))

# The best  value

bestLayer = -1

# The highest accuracy

highestAccuracy = 0.0

# Split the training set above into 2 sets: smaller training set and a validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.8, random_state=1234)

# Iterate through the range of values for parameter momentum

for layer in layerRange:

    # Create the  with the current ranged-value and all other params set to default

    model = MLPClassifier(random_state=1234, solver='sgd', activation='logistic', learning_rate='adaptive', learning_rate_init=0.00101, momentum=0.9, max_iter=200, hidden_layer_sizes=layer)

                                                             

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

    # Save the data and plot later

    dataFrameHLS = dataFrameHLS.append({'hidden_layer_sizes':layer, 'Accuracy':modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestLayer = layer

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(15, 6))

plot.xlabel('hidden_layer_sizes', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Neural Network on hidden_layer_sizes Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('hidden_layer_sizes', 'Accuracy', data=dataFrameHLS, marker='o')

plot.text(bestLayer, highestAccuracy, "Best Value: {}\nAccuracy: {}%".format(bestLayer, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

plot.grid()
'''Determine best hidden_layer_sizes value for best parameters and their values identified above'''

# A dataframe keeping track of the progress and for plotting later

dataFrameHLS = pandas.DataFrame(columns=['hidden_layer_sizes', 'Accuracy'])

# The range

layerRange = list(range(1200, 1600, 50))

# The best  value

bestLayer = -1

# The highest accuracy

highestAccuracy = 0.0

# Split the training set above into 2 sets: smaller training set and a validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.8, random_state=1234)

# Iterate through the range of values for parameter momentum

for layer in layerRange:

    # Create the  with the current ranged-value and all other params set to default

    model = MLPClassifier(random_state=1234, solver='sgd', activation='logistic', learning_rate='adaptive', learning_rate_init=0.00101, momentum=0.9, max_iter=200, hidden_layer_sizes=layer)

                                                             

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

    # Save the data and plot later

    dataFrameHLS = dataFrameHLS.append({'hidden_layer_sizes':layer, 'Accuracy':modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestLayer = layer

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(15, 6))

plot.xlabel('hidden_layer_sizes', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Neural Network on hidden_layer_sizes Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('hidden_layer_sizes', 'Accuracy', data=dataFrameHLS, marker='o')

plot.text(bestLayer, highestAccuracy, "Best Value: {}\nAccuracy: {}%".format(bestLayer, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

plot.grid()
'''Determine best hidden_layer_sizes value for best parameters and their values identified above'''

# A dataframe keeping track of the progress and for plotting later

dataFrameHLS = pandas.DataFrame(columns=['hidden_layer_sizes', 'Accuracy'])

# The range

layerRange = list(range(1500, 2000, 50))

# The best  value

bestLayer = -1

# The highest accuracy

highestAccuracy = 0.0

# Split the training set above into 2 sets: smaller training set and a validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.8, random_state=1234)

# Iterate through the range of values for parameter momentum

for layer in layerRange:

    # Create the  with the current ranged-value and all other params set to default

    model = MLPClassifier(random_state=1234, solver='sgd', activation='logistic', learning_rate='adaptive', learning_rate_init=0.00101, momentum=0.9, max_iter=200, hidden_layer_sizes=layer)

                                                             

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

    # Save the data and plot later

    dataFrameHLS = dataFrameHLS.append({'hidden_layer_sizes':layer, 'Accuracy':modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestLayer = layer

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(15, 6))

plot.xlabel('hidden_layer_sizes', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Neural Network on hidden_layer_sizes Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('hidden_layer_sizes', 'Accuracy', data=dataFrameHLS, marker='o')

plot.text(bestLayer, highestAccuracy, "Best Value: {}\nAccuracy: {}%".format(bestLayer, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

plot.grid()
'''Determine best hidden_layer_sizes value for best parameters and their values identified above'''

# A dataframe keeping track of the progress and for plotting later

dataFrameHLS = pandas.DataFrame(columns=['hidden_layer_sizes', 'Accuracy'])

# The range

layerRange = list(range(1800, 2500, 50))

# The best  value

bestLayer = -1

# The highest accuracy

highestAccuracy = 0.0

# Split the training set above into 2 sets: smaller training set and a validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.8, random_state=1234)

# Iterate through the range of values for parameter momentum

for layer in layerRange:

    # Create the  with the current ranged-value and all other params set to default

    model = MLPClassifier(random_state=1234, solver='sgd', activation='logistic', learning_rate='adaptive', learning_rate_init=0.00101, momentum=0.9, max_iter=200, hidden_layer_sizes=layer)

                                                             

    # Train the model

    model.fit(featuresTrainingSet, targetTrainingSet)

    

    # Base on what the model has learned set it to predict the outcome of the validation set

    modelPrediction = model.predict(featuresValidationSet)

    

    # Determine the model's accuracy on the validation set

    modelAccuracy = accuracy_score(targetValidationSet, modelPrediction)*100

    

    # Save the data and plot later

    dataFrameHLS = dataFrameHLS.append({'hidden_layer_sizes':layer, 'Accuracy':modelAccuracy}, ignore_index=True)

    

    # Save the highest accuracy based on the training size

    if modelAccuracy > highestAccuracy:

        bestLayer = layer

        highestAccuracy = modelAccuracy

        

plot.figure(figsize=(15, 6))

plot.xlabel('hidden_layer_sizes', color='white')

plot.ylabel('Accuracy %', color='white')

plot.title("LEARNING CURVE: Neural Network on hidden_layer_sizes Param", color='white')

plot.xticks(color='white')

plot.yticks(color='white')

plot.plot('hidden_layer_sizes', 'Accuracy', data=dataFrameHLS, marker='o')

plot.text(bestLayer, highestAccuracy, "Best Value: {}\nAccuracy: {}%".format(bestLayer, highestAccuracy), horizontalalignment='right', verticalalignment='center_baseline')

plot.grid()
'''DETERMINE THE OVERFIT/UNDERFIT OF THE MODEL BASED ON OPTIMAL PARAM VALUES FOUND AGAINST THE TEST SET'''

# Split the data - training set and testing set (for generalization)

featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(featuresData, targetData, test_size=0.2, random_state=1234)

# Create the  with the current ranged-value and all other params set to default

model = MLPClassifier(random_state=1234, solver='sgd', activation='logistic', learning_rate='adaptive', learning_rate_init=0.00101, momentum=0.9, max_iter=200, hidden_layer_sizes=2450)



# Train the model

model.fit(featuresTrainingSet, targetTrainingSet)



# Base on what the model has learned set it to predict the outcome of the validation set

modelPrediction = model.predict(featuresTestingSet)



# Determine the model's accuracy on the validation set

modelAccuracy = accuracy_score(targetsTestingSet, modelPrediction)*100

print(model)

print("Model's Accuracy: {0:.2f}".format(modelAccuracy))
'''DETERMINE THE OVERFIT/UNDERFIT OF THE MODEL BASED ON OPTIMAL PARAM VALUES FOUND AGAINST THE TEST SET'''

# Split the data - training set and testing set (for generalization)

featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(featuresData, targetData, test_size=0.2, random_state=1234)

# Create the  with the current ranged-value and all other params set to default

model = MLPClassifier(random_state=1234, solver='sgd', activation='logistic', learning_rate='adaptive', learning_rate_init=0.00101, momentum=0.9, max_iter=200, hidden_layer_sizes=1850)



# Train the model

model.fit(featuresTrainingSet, targetTrainingSet)



# Base on what the model has learned set it to predict the outcome of the validation set

modelPrediction = model.predict(featuresTestingSet)



# Determine the model's accuracy on the validation set

modelAccuracy = accuracy_score(targetsTestingSet, modelPrediction)*100

print(model)

print("Model's Accuracy: {0:.2f}".format(modelAccuracy))
'''USE CROSS VALIDATION'''

from sklearn.model_selection import cross_validate

# Split the data - training set and testing set (for generalization)

featuresWholeTrainingSet, featuresTestingSet, targetsWholeTrainingSet, targetsTestingSet = train_test_split(featuresData, targetData, test_size=0.2, random_state=1234)

# Split the training set above into 2 sets: 60% training set and 40% validation set

featuresTrainingSet, featuresValidationSet, targetTrainingSet, targetValidationSet = train_test_split(featuresWholeTrainingSet, targetsWholeTrainingSet, train_size=0.8, random_state=1234)

# Optimal identified params

model = MLPClassifier(random_state=1234, solver='sgd', activation='logistic', learning_rate='adaptive', learning_rate_init=0.00101, momentum=0.9, max_iter=200, hidden_layer_sizes=2450)

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