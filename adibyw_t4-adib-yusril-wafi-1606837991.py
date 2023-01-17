%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np
data = pd.read_csv("../input/dataset/diabetes.csv")

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

target = ['Outcome']

print(data.shape)

data.head()
for column in data:

    sorted_col = data.sort_values(column)

    x_max = sorted_col[column].iloc[-1]

    x_min = sorted_col[column].iloc[0]

    range_x = x_max - x_min

    

    for i in range(data.shape[0]):

        data.loc[i, column] = (data[column][i] - x_min) / range_x

    

data.head()
training = data.loc[:(768 * 80 // 100)]

test = data.loc[(768 * 80 // 100 + 1):]



x_training, y_training = training[features].values, training[target].values

x_test, y_test = test[features].values, test[target].values

print(x_training.shape, y_training.shape)

print(x_test.shape, y_test.shape)
# Make a prediction with weights

def predict(row, weights):

    activation = weights[0]

    for i in range(len(row)-1):

        activation += weights[i + 1] * row[i]

    return 1.0 if activation > 0.0 else 0.0

 

# Estimate Perceptron weights using stochastic gradient descent

def train_weights(train, l_rate, n_epoch):

    weights = [0.0 for i in range(len(train[0]))]    # Weights[0] is bias

    for epoch in range(n_epoch):

        sum_error = 0.0

        for row in train:

            prediction = predict(row, weights)

            error = row[-1] - prediction

            sum_error += error**2

            weights[0] = weights[0] + l_rate * error

            for i in range(len(row)-1):

                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]



    return weights



# Perceptron Algorithm With Stochastic Gradient Descent

def perceptron(train, test, l_rate, n_epoch):

    predictions = list()

    weights = train_weights(train, l_rate, n_epoch)

    for row in test:

        prediction = predict(row, weights)

        predictions.append(prediction)

    return predictions



l_rate = 0.1

n_epoch = 50

predicted = perceptron(training.values, test.values, l_rate, n_epoch)



for i in range(10):

    print(f'Predicted: {predicted[i]}, Actual:{y_test[i][0]}')
# Calculate accuracy percentage

def accuracy_metric(actual, predicted):

    correct = 0

    for i in range(len(actual)):

        if actual[i][0] == predicted[i]:

            correct += 1

    return correct / float(len(actual)) * 100.0



accuracy = accuracy_metric(y_test,predicted)

print(f'Accuracy = {accuracy}')
# Calculate precision percentage

def precision_metric(actual, predicted):

    correct = 0

    for i in range(len(actual)):

        if actual[i][0] == 1 and predicted[i] == 1:

            correct += 1

    return correct / predicted.count(1) * 100.0



precision = precision_metric(y_test,predicted)

print(f'Precision = {precision}')
# Calculate recall percentage

def recall_metric(actual, predicted):

    correct = 0

    for i in range(len(actual)):

        if actual[i][0] == 1 and predicted[i] == 1:

            correct += 1

    return correct / test[test['Outcome'] == 1].shape[0] * 100.0



recall = recall_metric(y_test,predicted)

print(f'Recall = {recall}')
from random import randrange, seed



# Create a random subsample from the dataset with replacement

def subsample(dataset, ratio=1.0):

    sample = list()

    total_sample = round(len(dataset) * ratio)

    while len(sample) < total_sample:

        index = randrange(len(dataset))

        sample.append(dataset[index])

    return sample



# Make a prediction with a list of bagged data

def bagging_predict(samples, test, l_rate, n_epoch):

    predictions = [perceptron(sample, test, l_rate, n_epoch) for sample in samples]

    

    voted_predictions = list()

    

    for i in range(len(predictions[0])):

        count_0 = 0

        count_1 = 0

        

        for prediction in predictions:

            if prediction[i] == 0.0:

                count_0 += 1

            else:

                count_1 += 1

        

        if count_0 <= count_1:

            voted_predictions.append(1.0) 

        else:

            voted_predictions.append(0.0) 

    

    return voted_predictions



# Bootstrap Aggregation Algorithm

def bagging(train, test, l_rate, n_epoch, sample_size, total_samples):

    samples = list()

    

    for i in range(total_samples):

        sample = subsample(train, sample_size)

        samples.append(sample)

        

    prediction = bagging_predict(samples, test, l_rate, n_epoch)

    return prediction



def evaluate_bagging(train, test, l_rate, n_epoch, sample_size, total_samples):

    predicted = bagging(train, test, l_rate, n_epoch, sample_size, total_samples)

    

    accuracy = accuracy_metric(y_test, predicted)

    precision = precision_metric(y_test, predicted)

    recall = recall_metric(y_test, predicted)

    

    return accuracy, precision, recall



seed(1)

sample_size = 0.50

l_rate = 0.1

n_epoch = 20



for n_sample in [1, 5, 10, 50]:

    accuracy, precision, recall = evaluate_bagging(training.values, test.values, l_rate, n_epoch, sample_size, n_sample)

    

    print(f'Number of Samples: {n_sample}')

    print(f'Accuracy: {accuracy}')

    print(f'Precision = {precision}')

    print(f'Recall = {recall}')

    print("="*50)
from sklearn.neural_network import MLPClassifier



# Make a prediction with a list of bagged data

def bagging_predict(samples, test):

    predictions = list()

    

    for sample in samples:

        x_training = [arr[:8] for arr in sample]

        y_training = [arr[-1] for arr in sample]

        

        clf.fit(x_training, y_training)

        predictions.append(clf.predict(x_test))

    

    voted_predictions = list()

    

    for i in range(len(predictions[0])):

        count_0 = 0

        count_1 = 0

        

        for prediction in predictions:

            if prediction[i] == 0.0:

                count_0 += 1

            else:

                count_1 += 1

        

        if count_0 <= count_1:

            voted_predictions.append(1.0) 

        else:

            voted_predictions.append(0.0) 

    

    return voted_predictions



# Bootstrap Aggregation Algorithm

def bagging(train, test, sample_size, total_samples):

    samples = list()

    

    for i in range(total_samples):

        sample = subsample(train, sample_size)

        samples.append(sample)

        

    prediction = bagging_predict(samples, test)

    return prediction



def evaluate_bagging(train, test, sample_size, total_samples):

    predicted = bagging(train, test, sample_size, total_samples)

    

    accuracy = accuracy_metric(y_test, predicted)

    precision = precision_metric(y_test, predicted)

    recall = recall_metric(y_test, predicted)

    

    return accuracy, precision, recall



seed(2)

sample_size = 0.50



clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8,2), random_state=2)





for n_sample in [1, 5, 10, 50, 70]:

    accuracy, precision, recall = evaluate_bagging(training.values, test.values, sample_size, n_sample)

    

    print(f'Number of Samples: {n_sample}')

    print(f'Accuracy: {accuracy}')

    print(f'Precision = {precision}')

    print(f'Recall = {recall}')

    print("="*50)