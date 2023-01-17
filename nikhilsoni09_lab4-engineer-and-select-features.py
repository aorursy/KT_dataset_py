import pandas as pd

import numpy as np

# import the regressor 

from sklearn.tree import DecisionTreeRegressor  

from sklearn.metrics import mean_absolute_error
#dataframe = pd.read_csv("https://introtomlsampledata.blob.core.windows.net/data/bike-rental/bike-rental-hour.csv")

dataframe = pd.read_csv("../input/bikerentalhour/bike-rental-hour.csv")
dataframe.head()
dataframe.columns
dataframe.dtypes
categoryVariableList = ["season","weathersit"]

for var in categoryVariableList:

    dataframe[var] = dataframe[var].astype("category")

dataframe.dtypes
dataframe = dataframe.drop(["instant", "dteday", "casual" ,"registered"], axis = 1)
dataframe.head()
dataframesetA = dataframe.copy()

dataframesetA.columns
def azureml_main(dataframe1 = None, dataframe2 = None):



    # Execution logic goes here

    #print(f'Input pandas.DataFrame #1: {dataframe1}')



    # If a zip file is connected to the third input port,

    # it is unzipped under "./Script Bundle". This directory is added

    # to sys.path. Therefore, if your zip file contains a Python file

    # mymodule.py you can import it using:

    # import mymodule



    for i in np.arange(1, 13):

        prev_col_name = 'cnt' if i == 1 else 'Rentals in hour -{}'.format(i-1)

        new_col_name = 'Rentals in hour -{}'.format(i)



        dataframe1[new_col_name] = dataframe1[prev_col_name].shift(1).fillna(0)



    # Return value must be of a sequence of pandas.DataFrame

    # E.g.

    #   -  Single return value: return dataframe1,

    #   -  Two return values: return dataframe1, dataframe2

    return dataframe1,
dataframesetAB = azureml_main(dataframe)[0]
dataframesetAB.columns
dataframesetAB.head()
dataframesetAtest = dataframesetA[dataframesetA["yr"] == 0]

dataframesetAtrain = dataframesetA[dataframesetA["yr"] != 0]

dataframesetAtest.shape,dataframesetAtrain.shape
dataframesetABtest = dataframesetAB[dataframesetAB["yr"] == 0]

dataframesetABtrain = dataframesetAB[dataframesetAB["yr"] != 0]

dataframesetABtest.shape,dataframesetABtrain.shape
dataframesetAtest = dataframesetAtest.drop(["yr"],axis = 1)

dataframesetAtrain = dataframesetAtrain.drop(["yr"],axis = 1)

dataframesetABtest = dataframesetABtest.drop(["yr"],axis = 1)

dataframesetABtrain = dataframesetABtrain.drop(["yr"],axis = 1)
dataframesetAtest.shape,dataframesetAtrain.shape
dataframesetABtest.shape,dataframesetABtrain.shape
dataframesetAtestX = dataframesetAtest.drop(["cnt"],axis = 1)

dataframesetAtesty = dataframesetAtest["cnt"]

dataframesetAtestX.shape,dataframesetAtesty.shape
dataframesetAtrainX = dataframesetAtrain.drop(["cnt"],axis = 1)

dataframesetAtrainy = dataframesetAtrain["cnt"]

dataframesetAtrainX.shape,dataframesetAtrainy.shape
dataframesetABtestX = dataframesetABtest.drop(["cnt"],axis = 1)

dataframesetABtesty = dataframesetABtest["cnt"]

dataframesetABtestX.shape,dataframesetABtesty.shape
dataframesetABtrainX = dataframesetABtrain.drop(["cnt"],axis = 1)

dataframesetABtrainy = dataframesetABtrain["cnt"]

dataframesetABtrainX.shape,dataframesetABtrainy.shape
regressorA = DecisionTreeRegressor(random_state = 0)  

regressorA.fit(dataframesetAtrainX, dataframesetAtrainy) 
regressorAB = DecisionTreeRegressor(random_state = 0)  

regressorAB.fit(dataframesetABtrainX, dataframesetABtrainy) 
maesetA = mean_absolute_error(dataframesetAtesty, regressorA.predict(dataframesetAtestX))
maesetAB = mean_absolute_error(dataframesetABtesty,regressorAB.predict(dataframesetABtestX))
print("The Mean Absolute Error for set A: " + str(maesetA))

print("The Mean Absolute Error for set A + B: " + str(maesetAB))