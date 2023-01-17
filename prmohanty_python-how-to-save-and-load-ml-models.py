# Import Required packages 

#-------------------------



# Import the Logistic Regression Module from Scikit Learn

from sklearn.linear_model import LogisticRegression  



# Import the IRIS Dataset to be used in this Kernel

from sklearn.datasets import load_iris  



# Load the Module to split the Dataset into Train & Test 

from sklearn.model_selection import train_test_split

# Load the data

Iris_data = load_iris()  

# Split data

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Iris_data.data, 

                                                Iris_data.target, 

                                                test_size=0.3, 

                                                random_state=4)  
# Define the Model

LR_Model = LogisticRegression(C=0.1,  

                               max_iter=20, 

                               fit_intercept=True, 

                               n_jobs=3, 

                               solver='liblinear')



# Train the Model

LR_Model.fit(Xtrain, Ytrain)  
# Import pickle Package



import pickle

# Save the Modle to file in the current working directory



Pkl_Filename = "Pickle_RL_Model.pkl"  



with open(Pkl_Filename, 'wb') as file:  

    pickle.dump(LR_Model, file)

# Load the Model back from file

with open(Pkl_Filename, 'rb') as file:  

    Pickled_LR_Model = pickle.load(file)



Pickled_LR_Model
# Use the Reloaded Model to 

# Calculate the accuracy score and predict target values



# Calculate the Score 

score = Pickled_LR_Model.score(Xtest, Ytest)  

# Print the Score

print("Test score: {0:.2f} %".format(100 * score))  



# Predict the Labels using the reloaded Model

Ypredict = Pickled_LR_Model.predict(Xtest)  



Ypredict
# Import Joblib Module from Scikit Learn



from sklearn.externals import joblib

# Save RL_Model to file in the current working directory



joblib_file = "joblib_RL_Model.pkl"  

joblib.dump(LR_Model, joblib_file)

# Load from file



joblib_LR_model = joblib.load(joblib_file)





joblib_LR_model
# Use the Reloaded Joblib Model to 

# Calculate the accuracy score and predict target values



# Calculate the Score 

score = joblib_LR_model.score(Xtest, Ytest)  

# Print the Score

print("Test score: {0:.2f} %".format(100 * score))  



# Predict the Labels using the reloaded Model

Ypredict = joblib_LR_model.predict(Xtest)  



Ypredict
# Import required packages



import json  

import numpy as np

class MyLogReg(LogisticRegression):



    # Override the class constructor

    def __init__(self, C=1.0, solver='liblinear', max_iter=100, X_train=None, Y_train=None):

        LogisticRegression.__init__(self, C=C, solver=solver, max_iter=max_iter)

        self.X_train = X_train

        self.Y_train = Y_train



    # A method for saving object data to JSON file

    def save_json(self, filepath):

        dict_ = {}

        dict_['C'] = self.C

        dict_['max_iter'] = self.max_iter

        dict_['solver'] = self.solver

        dict_['X_train'] = self.X_train.tolist() if self.X_train is not None else 'None'

        dict_['Y_train'] = self.Y_train.tolist() if self.Y_train is not None else 'None'



        # Creat json and save to file

        json_txt = json.dumps(dict_, indent=4)

        with open(filepath, 'w') as file:

            file.write(json_txt)



    # A method for loading data from JSON file

    def load_json(self, filepath):

        with open(filepath, 'r') as file:

            dict_ = json.load(file)



        self.C = dict_['C']

        self.max_iter = dict_['max_iter']

        self.solver = dict_['solver']

        self.X_train = np.asarray(dict_['X_train']) if dict_['X_train'] != 'None' else None

        self.Y_train = np.asarray(dict_['Y_train']) if dict_['Y_train'] != 'None' else None
filepath = "mylogreg.json"



# Create a model and train it

mylogreg = MyLogReg(X_train=Xtrain, Y_train=Ytrain)  

mylogreg.save_json(filepath)



# Create a new object and load its data from JSON file

json_mylogreg = MyLogReg()  

json_mylogreg.load_json(filepath)  

json_mylogreg  