from datetime import datetime
from datetime import timedelta
start = datetime.now()          ### Measuring time
TotalTime = timedelta(0)        ### Measuring total elapsed time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import git

from sklearn import linear_model
from sklearn import ensemble
from sklearn import neighbors
from sklearn import neural_network
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict

PathInputFolder = '../input/'                  # Folder of input data
PathTrain = '../input/train.csv.gz'            # Input TRAIN data
PathTest = '../input/test.csv.gz'              # Input TEST data
PathSample = '../input/sampleSubmission.csv'   # Input SAMPLE data

# PathOutputFolder = '../output/'                # Folder of input data
# PathOutput = '../output/sdss_predict.csv'      # Output file

print("#### Time for initialization: ", datetime.now() - start, '\n')
TotalTime += datetime.now() - start ### Measuring time total
def Models():

    # Linear models
    forest = ensemble.RandomForestRegressor(n_estimators=200, min_samples_split=15,min_samples_leaf=5)
    extra = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=30, min_samples_split=15 ,min_samples_leaf=5)
    linear = linear_model.LinearRegression()
    neural_1 = neural_network.MLPRegressor(hidden_layer_sizes=(120,), activation='relu',learning_rate_init=0.0001)
    ridge = linear_model.Ridge()
    KN = neighbors.KNeighborsRegressor(n_neighbors=150, weights='uniform')

    # Polynomial models
    polyness = make_pipeline(PolynomialFeatures(4), neural_1)

    ############ MODIFY TO CHOOSE MODEL ############
    model = extra
    ############ MODIFY TO CHOOSE MODEL ############

    print("Choosen model: ", model, '\n')

    return(model)
start = datetime.now() ### Measuring time

model = Models()

print("#### Time for choosing model: ", datetime.now() - start, '\n')
TotalTime += datetime.now() - start ### Measuring time total
def Train(model):

    # Read in parameters
    TrainDataFrame = pd.read_csv(PathTrain)                     # Load train data from .csv
    print("Train data ready!")
    x = TrainDataFrame[['u','g','r','i','z']].values            # Format x for model.fit
    y = TrainDataFrame['redshift'].values                       # Format y for model.fit

    # Fit choosen model
    model.fit(x,y)
    print("Success! Model fitted!")

    return(model, x, y)
start = datetime.now()       ### Measuring time

model, x, y = Train(model)

print("#### Time for reading in and fitting on train: ", datetime.now() - start, '\n')
TotalTime += datetime.now() - start ### Measuring time total
def Test(model):

    TestDataFrame =  pd.read_csv(PathTest)                 # Read test data
    print("Test data ready!")
    x_test = TestDataFrame[['u','g','r','i','z']].values   # Get test x
    y_pred = model.predict(x_test)                         # Predict test y from x
    print("Test prediction finished!")

    # Write y in the sample submission table (order is good)
    ResultDataFrame =  pd.read_csv(PathSample)
    ResultDataFrame['redshift'] =  y_pred

    # Save submission (no index column)
    ResultDataFrame.to_csv("sdss_predict.csv",index=False, mode='w+')

    return(model, y_pred)
start = datetime.now()       ### Measuring time

model, y_pred_model = Test(model)

print("#### Time for reading in test, predict redshifts and wrting them to files: ", datetime.now() - start, '\n')
TotalTime += datetime.now() - start ### Measuring time total

## PRINT TOTAL RUNTIME
print("#### Total time for running the whole program: ", TotalTime, "s")