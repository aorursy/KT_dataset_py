import numpy as np # linear algebra

from math import isnan

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt
train_set = pd.read_csv('/kaggle/input/titanic/train.csv')

test_set = pd.read_csv('/kaggle/input/titanic/test.csv')



# Pick out the features

fare = train_set['Fare'].to_numpy()

age = train_set['Age'].to_numpy()

gender = train_set['Sex']

survived = train_set['Survived'].to_numpy()



# Correlations between fare and survival rate

fare_graph = sns.kdeplot(train_set['Fare'][(train_set['Survived'] == 1)], color = "Blue", shade = True)

fare_graph = sns.kdeplot(train_set['Fare'][(train_set['Survived'] == 0)], color = "Red", shade = True)

fare_graph.set_xlabel("Fare")

fare_graph.set_ylabel("Survival Frequency")

fare_graph = fare_graph.legend(["Survived", "Died"])

plt.title("Survival rate by Fare")
# Correlation between age and survival rate

age_graph = sns.kdeplot(train_set['Age'][(train_set['Survived'] == 1)], color = "Blue", shade = True)

age_graph = sns.kdeplot(train_set['Age'][(train_set['Survived'] == 0)], color = "Red", shade = True)

age_graph.set_xlabel("Age")

age_graph.set_ylabel("Survival Frequency")

age_graph = age_graph.legend(["Survived", "Died"])

plt.title("Survival rate by Age")
# Correlation between gender and survival rate

sns.barplot(data=train_set, x='Sex', y='Survived')

plt.title("Survival rate by Gender")
# Processes a data set

def process_data_set(data, train):

    # Number of items

    m = len(data)

    

    ids = data['PassengerId']

    

    # Fetch the fare, age and gender

    fare = data['Fare']

    age = data['Age']

    sex = data['Sex']

    

    # Replace NaN values

    fare = np.nan_to_num(fare)

    

    age_fixed = np.zeros(age.shape)

    age_med = np.median(age[~np.isnan(age)]) # maybe not the most accurate but there was very little correlation

    for i in range(len(age)):

        if isnan(age[i]):

            age_fixed[i] = age_med

        else:

            age_fixed[i] = age[i]

    

    # Make sex a numerical field

    is_male = np.zeros(sex.shape)

    for i in range(len(sex)):

        if sex[i] == "male":

            is_male[i] = 1

    

    processed = np.array([fare, age_fixed, is_male])

    

    # normalize columns

    p_norm = np.linalg.norm(processed, axis=1, keepdims=True)

    processed /= p_norm

    

    if train:

        return ids, processed, data['Survived'].to_numpy().reshape((1, m))

    else:

        return ids, processed



# process the data

train_ids, train_inputs, train_labels = process_data_set(train_set, True)

test_ids, test_inputs = process_data_set(test_set, False)



print("Training set shape: " + str(train_inputs.shape))

print("Test set shape: " + str(test_inputs.shape))



print(train_inputs)
# Sigmoid function

def sig(x):

    return 1 / (1 + np.exp(-x))
# Predicts values for a data set

# A = sig(w.T * X + b)

def predict(w, b, X):

    return sig(np.dot(w.T, X) + b)
# Gets the accuracy of the parameters

def get_accuracy(w, b, inputs, labels):

    # predict values

    A = predict(w, b, inputs)

    Y_pred = np.round(A)

    

    # check that they match

    return 100 * np.sum(Y_pred == labels) / inputs.shape[1]
# Performs forward and backward propagation

def propagate(w, b, inputs, labels):

    # number of inputs

    m = inputs.shape[0]

    

    # forward propagation

    A = predict(w, b, inputs)

    cost = -1 / m * np.sum(labels * np.log(A) + (1 - labels) * np.log(1 - A))

    

    # back propagation

    dw = 1 / m * np.dot(inputs, (A - labels).T)

    db = 1 / m * np.sum(A - labels)

    

    # return gradients and cost

    return dw, db, cost
# Optimizes the parameters using gradient descent

def optimize(w, b, inputs, labels, num_iter, learning_rate):

    for i in range(num_iter):

        # get gradients and cost

        dw, db, cost = propagate(w, b, inputs, labels)

        

        # update parameters

        w = w - dw * learning_rate

        b = b - db * learning_rate

        

        # print cost every x iter

        if (i + 1) % (num_iter / 10) == 0 or i == 0:

            print("Cost on iteration %i: %.2f" % (i + 1, cost))

        

    # return parameters found

    return w, b
# Initialize weights

num_features = 3

w = np.random.randn(num_features, 1)

b = 0



# Optimizer weights

w, b = optimize(w, b, train_inputs, train_labels, 200000, 0.04)



# Get accuracy

accuracy = get_accuracy(w, b, train_inputs, train_labels)

print("Accuracy: %.2f" % accuracy, end="")

print("%")
# predict all the test set values

test_pred = predict(w, b, test_inputs)

test_pred = np.round(test_pred)



# format it like the example submission

output = {'PassengerId': test_ids, 'Survived': test_pred[0].astype(np.int32)}

dataframe = pd.DataFrame(output, columns = ['PassengerId', 'Survived'])



# convert to csv and save

dataframe.to_csv(r"./titanic_submission.csv", index=False, header=True)