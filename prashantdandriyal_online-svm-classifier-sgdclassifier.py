# Install Dependencies



import numpy as np 

import pandas as pd

import time

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt
# Import Data



from subprocess import check_output

from datetime import time

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/dataset.csv")

print(df.shape)

df.head()
# Data Preprocess, remove un-required features



df = df.drop(["date", "time", "username"], axis=1)

print(df.describe())

data = df.values

X = data[:, 1:]  # all rows, no label

y = data[:, 0]  # all rows, label only
# Model Definition

def svmClassifier(scaledData_X, scaledData_Y, firstRun=0, model=None):

    '''

    "It accepts the scaled features (training data) and trains a SVM Regressor"

    Args:

        Accepts scaled training dataset and 'firstRun' flag to distinguish between first time and online training.

    Returns:

        Loaded Model    

    '''

    

    try:

        # for reproducible result

        np.random.seed(3)

        #HyperParameter and model element asigning

        xTrain = scaledData_X

        yTrain = scaledData_Y



        if yTrain.shape[0] != 1:

            yTrain = yTrain.reshape(-1,1)



        xShape = xTrain.shape

        # print(xShape)

        yShape = yTrain.shape

        # print("input: ", xShape[1], " output: ", yShape)



        # Make sure model is only created once and trained numerously

        if(firstRun==1):

            model = SGDClassifier(loss="hinge", penalty="l2", alpha=0.0001, max_iter=3000, tol=None, shuffle=True, verbose=0, learning_rate='adaptive', eta0=0.01, early_stopping=False)

            # Suggested number of passes for convergence

            model.n_iter = np.ceil(10**6 / len(scaledData_Y))

            # Train model using fit

            model.fit(xTrain, yTrain)

        else:

            # Use the model passed to the function

            # As partial fit only runs for 1 epoch, we repeat for all samples.

            for _ in range(3):

                for i in range(xShape[0]):

                    #print(f'for i={i}, \t actual {xTrain[i].shape}')

                    x = xTrain[i].reshape(1, -1)

                    #print(f'Now, {x.shape}')

                    model.partial_fit(x, yTrain[i])

            

        

        print('Done Training')

        message = "Successfully trained SVMClassifier"



    except Exception as e:

        message = e

        model = {}



    return message, model

# Data Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Scale the data to be between -1 and 1

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
# Train

message, model = svmClassifier(X_train, y_train, 1); print(message)

model.score(X_test, y_test)
# Things done only once...

X_split = np.array_split(X, 4)

y_split = np.array_split(y, 4)

scaler = StandardScaler()
# Part 1: fit for first time

X_train, X_test, y_train, y_test = train_test_split(X_split[0],y_split[0],random_state=999)



# Scale the data to be between -1 and 1

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



message, model = svmClassifier(X_train, y_train, 1); print(message)

model.score(X_test, y_test)

#Save model

import pickle

modelSaveFile = 'model_data_id.sav'

pickle.dump(model, open(modelSaveFile, 'wb'))
# Part 2: partial_fit

X_train, X_test, y_train, y_test = train_test_split(X_split[1],y_split[1],random_state=999)



# Load the model

model = pickle.load(open(modelSaveFile, 'rb'))

# Scale the data to be between -1 and 1

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



message, model = svmClassifier(X_train, y_train, 0, model); print(message)

print(model.score(X_test, y_test))



# Save model

modelSaveFile = 'model_data_id2.sav'

pickle.dump(model, open(modelSaveFile, 'wb'))
# Part 3: partial_fit

X_train, X_test, y_train, y_test = train_test_split(X_split[2],y_split[2],random_state=999)



# Load the model

model = pickle.load(open(modelSaveFile, 'rb'))

# Scale the data to be between -1 and 1

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



message, model = svmClassifier(X_train, y_train, 0, model); print(message)

print(model.score(X_test, y_test))



# Save model

modelSaveFile = 'model_data_id3.sav'

pickle.dump(model, open(modelSaveFile, 'wb'))
# Part 4: partial_fit

X_train, X_test, y_train, y_test = train_test_split(X_split[3],y_split[3],random_state=999)



# Load the model

model = pickle.load(open(modelSaveFile, 'rb'))

# Scale the data to be between -1 and 1

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



message, model = svmClassifier(X_train, y_train, 0, model); print(message)

print(model.score(X_test, y_test))



# Save model

modelSaveFile = 'model_data_id4.sav'

pickle.dump(model, open(modelSaveFile, 'wb'))
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



def metrics(yActual, yPredicted):

    print(f'accuracy_score: {accuracy_score(yActual, yPredicted)}')

    print(f"f1 score: {f1_score(yActual, yPredicted, average='weighted')}")



metrics(y_test, model.predict(X_test))